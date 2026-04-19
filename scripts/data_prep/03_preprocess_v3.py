import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import anndata as ad
import scanpy as sc

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dpas.data.adt_names import clean_adt_varnames_inplace


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--registry_out", type=str, required=True)

    ap.add_argument("--gene_panel_path", type=str, required=True)
    ap.add_argument("--protein_panel_path", type=str, required=True)

    # filters / QC
    ap.add_argument("--only_in_tissue", action="store_true")
    ap.add_argument("--qc_min_genes", type=int, default=500)
    ap.add_argument("--qc_max_genes", type=int, default=0)
    ap.add_argument("--qc_max_mt_pct", type=float, default=35.0)

    # RNA transform (must remain nonnegative)
    ap.add_argument("--target_sum", type=float, default=1e4)
    ap.add_argument("--clip_max", type=float, default=0.0)
    ap.add_argument("--reset_from_raw", action="store_true", default=True)

    # ADT handling
    ap.add_argument("--skip_protein_cleaning", action="store_true",
                    help="If set, do NOT run clean_adt_varnames_inplace; assumes ADT var_names already cleaned.")
    ap.add_argument("--remove_isotype", action="store_true")
    ap.add_argument("--min_nt_len", type=int, default=6)
    ap.add_argument("--protein_missing_policy", choices=["error", "drop_missing"], default="error",
                    help="If some proteins in protein_panel_path are missing after (optional) cleaning: "
                         "'error' will raise; 'drop_missing' will keep only existing proteins (not recommended for mainline).")

    return ap.parse_args()


def _read_lines(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _ensure_raw_layer(adata: ad.AnnData, name: str) -> None:
    if "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()
    X = adata.layers["raw"].toarray() if hasattr(adata.layers["raw"], "toarray") else adata.layers["raw"]
    if float(np.min(X)) < -1e-8:
        raise ValueError(f"{name}: layers['raw'] has negative values.")


def _align_obs(rna: ad.AnnData, adt: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData]:
    common = rna.obs_names.intersection(adt.obs_names)
    if len(common) == 0:
        raise ValueError("RNA/ADT obs_names intersection is empty.")
    rna = rna[common].copy()
    adt = adt[common].copy()
    adt = adt[rna.obs_names].copy()
    return rna, adt


def _filter_in_tissue(rna: ad.AnnData, adt: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData]:
    if "in_tissue" not in rna.obs.columns or "in_tissue" not in adt.obs.columns:
        raise ValueError("missing obs['in_tissue']")
    mask = (rna.obs["in_tissue"].astype(int).values == 1)
    rna = rna[mask].copy()
    adt = adt[rna.obs_names].copy()
    return rna, adt


def _rna_qc_filter(rna: ad.AnnData, qc_min_genes: int, qc_max_genes: int, qc_max_mt_pct: float) -> ad.AnnData:
    # QC metrics must be computed on raw counts
    X_src = rna.layers["raw"] if "raw" in rna.layers else rna.X
    tmp = rna.copy()
    tmp.X = X_src

    tmp.var["mt"] = tmp.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(tmp, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    mask = (tmp.obs["n_genes_by_counts"].values >= qc_min_genes) & (tmp.obs["pct_counts_mt"].values <= qc_max_mt_pct)
    if qc_max_genes is not None and qc_max_genes > 0:
        mask &= (tmp.obs["n_genes_by_counts"].values <= qc_max_genes)

    return rna[mask].copy()


def _apply_rna_transform(rna: ad.AnnData, target_sum: float, clip_max: float, reset_from_raw: bool) -> None:
    if reset_from_raw and "raw" in rna.layers:
        rna.X = rna.layers["raw"].copy()

    sc.pp.normalize_total(rna, target_sum=target_sum)
    sc.pp.log1p(rna)

    if clip_max and clip_max > 0:
        if hasattr(rna.X, "toarray"):
            X = rna.X.toarray()
            X = np.clip(X, 0.0, float(clip_max)).astype(np.float32)
            rna.X = X
        else:
            rna.X = np.clip(rna.X, 0.0, float(clip_max)).astype(np.float32)


def _subset_and_reorder_adt_to_panel(
    adt: ad.AnnData,
    protein_panel: List[str],
    *,
    missing_policy: str = "error",
) -> Tuple[ad.AnnData, List[str]]:
    have = set(map(str, adt.var_names))
    missing = [p for p in protein_panel if p not in have]
    if missing:
        if missing_policy == "error":
            raise ValueError(
                f"ADT missing {len(missing)} proteins from protein_panel (show 10): {missing[:10]}\n"
                f"If you believe your ADT is already cleaned but names still mismatch, rerun WITHOUT --skip_protein_cleaning."
            )
        else:
            # drop missing (not recommended for mainline)
            proteins_used = [p for p in protein_panel if p in have]
            if len(proteins_used) == 0:
                raise ValueError("After drop_missing, no proteins remain. Check protein cleaning/panel.")
            adt = adt[:, proteins_used].copy()
            return adt, proteins_used

    adt = adt[:, protein_panel].copy()
    return adt, protein_panel


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_panel = _read_lines(args.gene_panel_path)
    protein_panel = _read_lines(args.protein_panel_path)

    specs = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
    if isinstance(specs, dict) and "specs" in specs:
        specs = specs["specs"]

    new_specs: List[Dict[str, object]] = []

    for s in specs:
        name = str(s["name"])
        rna = ad.read_h5ad(s["rna"])
        adt = ad.read_h5ad(s["adt"])

        _ensure_raw_layer(rna, f"{name}/RNA")
        _ensure_raw_layer(adt, f"{name}/ADT")

        # align by obs_names (do not trust row order)
        rna, adt = _align_obs(rna, adt)

        if args.only_in_tissue:
            rna, adt = _filter_in_tissue(rna, adt)

        # QC mask from RNA; apply to ADT by obs_names
        rna = _rna_qc_filter(rna, args.qc_min_genes, args.qc_max_genes, args.qc_max_mt_pct)
        adt = adt[rna.obs_names].copy()

        # subset RNA genes to fixed panel (strict; should be feasible given Stage2)
        missing_genes = [g for g in gene_panel if g not in set(rna.var_names)]
        if missing_genes:
            raise ValueError(f"{name}: missing {len(missing_genes)} genes from gene_panel (show 10): {missing_genes[:10]}")
        rna = rna[:, gene_panel].copy()

        _apply_rna_transform(rna, args.target_sum, args.clip_max, args.reset_from_raw)
        rna.uns["preprocessed_v3"] = True
        rna.uns["gene_panel_path"] = str(Path(args.gene_panel_path).as_posix())

        # ADT: optional cleaning, BUT do NOT recompute intersection here
        if not args.skip_protein_cleaning:
            rep = clean_adt_varnames_inplace(
                adt,
                remove_isotype=args.remove_isotype,
                min_nt_len=int(args.min_nt_len),
                keep_first_duplicate=True,
            )
            adt.uns["protein_clean_report"] = rep.__dict__
            adt.uns["protein_cleaning_applied"] = True
        else:
            adt.uns["protein_cleaning_applied"] = False

        # enforce Stage1 protein panel (fixed P + fixed order)
        adt, proteins_used = _subset_and_reorder_adt_to_panel(
            adt,
            protein_panel,
            missing_policy=args.protein_missing_policy,
        )

        # ensure raw layer still exists and is aligned
        if "raw" not in adt.layers:
            adt.layers["raw"] = adt.X.copy()

        adt.uns["preprocessed_v3"] = True
        adt.uns["protein_panel_path"] = str(Path(args.protein_panel_path).as_posix())
        adt.uns["proteins_used_size"] = int(len(proteins_used))

        # write
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        rna_out = d / "RNA_proc.h5ad"
        adt_out = d / "ADT_raw_clean.h5ad"
        rna.write_h5ad(str(rna_out))
        adt.write_h5ad(str(adt_out))

        new_specs.append({
            "name": name,
            "batch_id": int(s.get("batch_id", 0)),
            "rna": str(rna_out).replace("\\", "/"),
            "adt": str(adt_out).replace("\\", "/"),
        })

    Path(args.registry_out).write_text(json.dumps(new_specs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] wrote {len(new_specs)} processed pairs -> {args.registry_out}")
    print(f"[Done] used fixed protein panel -> {args.protein_panel_path}")
    print(f"[Done] used fixed gene panel -> {args.gene_panel_path}")


if __name__ == "__main__":
    main()