import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


def infer_dataset_name(p: Path):
    # 常见：<dataset>_RNA_raw.h5ad / <dataset>_ADT_raw.h5ad
    name = p.stem
    name = re.sub(r"_(RNA|ADT).*?$", "", name, flags=re.IGNORECASE)
    return name


def frac_negative(X):
    if X is None:
        return None
    if sp.issparse(X):
        return float((X.data < 0).mean()) if X.data.size else 0.0
    X = np.asarray(X)
    return float((X < 0).mean())


def integer_like_ratio(X, sample_n=20000, tol=1e-6):
    if X is None:
        return None
    if sp.issparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()

    if data.size == 0:
        return None
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    if data.size > sample_n:
        idx = np.random.choice(data.size, size=sample_n, replace=False)
        data = data[idx]
    frac = np.abs(data - np.round(data))
    return float(np.mean(frac < tol))


def check_one(path: Path, kind: str):
    a = ad.read_h5ad(path)

    problems = []
    warns = []

    # basic
    if a.n_obs == 0 or a.n_vars == 0:
        problems.append("empty AnnData")

    # raw layer
    has_raw = "raw" in a.layers
    if not has_raw:
        problems.append("missing layers['raw']")
    else:
        if a.layers["raw"].shape != a.shape:
            problems.append(f"layers['raw'].shape {a.layers['raw'].shape} != .shape {a.shape}")

    # nonneg + integer-like for raw
    if has_raw:
        neg = frac_negative(a.layers["raw"])
        if neg and neg > 0:
            problems.append(f"raw has negative values (frac={neg:.3g})")
        ilr = integer_like_ratio(a.layers["raw"])
        if ilr is not None and ilr < 0.99:
            warns.append(f"raw not integer-like (ratio={ilr:.3g})")

    # spatial
    if "spatial" not in a.obsm:
        problems.append("missing obsm['spatial']")
    else:
        S = np.asarray(a.obsm["spatial"])
        if S.ndim != 2 or S.shape[1] != 2 or S.shape[0] != a.n_obs:
            problems.append(f"obsm['spatial'] has shape {S.shape}, expected (n_obs,2)")

    # obs required
    for col in ["in_tissue", "array_row", "array_col"]:
        if col not in a.obs.columns:
            warns.append(f"missing obs['{col}']")
    # these are optional but strongly recommended for traceability
    for col in ["barcode_raw"]:
        if col not in a.obs.columns:
            warns.append(f"missing obs['{col}'] (recommended for traceability)")

    # pixel coords optional but useful
    for col in ["pxl_row_in_fullres", "pxl_col_in_fullres"]:
        if col not in a.obs.columns:
            warns.append(f"missing obs['{col}'] (pixel coords)")

    # var recommended
    # 00_build_raw_h5ad.py style usually: gene_ids + feature_types (+ genome)
    if not any(c in a.var.columns for c in ["gene_ids", "feature_id", "feature_ids"]):
        warns.append("missing var gene id column (gene_ids/feature_id)")
    if not any(c in a.var.columns for c in ["feature_types", "feature_type"]):
        warns.append("missing var feature type column (feature_types)")

    # obs_names
    if not a.obs_names.is_unique:
        problems.append("obs_names not unique")
    # kind-specific sanity
    if kind == "RNA":
        # genes usually large
        if a.n_vars < 5000:
            warns.append(f"RNA has only {a.n_vars} vars (unexpectedly small?)")
    if kind == "ADT":
        if a.n_vars < 10:
            warns.append(f"ADT has only {a.n_vars} vars (unexpectedly small?)")

    # uns recommended keys
    for k in ["dataset_name", "batch_id", "spatial_coord_kind"]:
        if k not in a.uns:
            warns.append(f"missing uns['{k}'] (recommended)")

    summary = {
        "path": str(path),
        "kind": kind,
        "shape": [int(a.n_obs), int(a.n_vars)],
        "X_neg_frac": frac_negative(a.X),
        "raw_neg_frac": frac_negative(a.layers["raw"]) if has_raw else None,
        "raw_integer_like_ratio": integer_like_ratio(a.layers["raw"]) if has_raw else None,
        "obsm_keys": list(a.obsm.keys()),
        "layers": list(a.layers.keys()),
        "obs_cols_head": list(a.obs.columns[:25]),
        "var_cols": list(a.var.columns),
        "uns_keys_head": list(a.uns.keys())[:25],
        "problems": problems,
        "warnings": warns,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r'folder containing h5ad files, e.g. G:\MODEL\Data\data\output')
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_json = Path(args.out_json) if args.out_json else root / "healthcheck_report.json"
    out_csv = Path(args.out_csv) if args.out_csv else root / "healthcheck_summary.csv"

    h5ads = sorted(root.glob("*.h5ad"))
    if not h5ads:
        raise FileNotFoundError(f"No .h5ad found in {root}")

    # group by dataset
    groups = {}
    for p in h5ads:
        ds = infer_dataset_name(p)
        s = p.stem.lower()
        kind = "RNA" if ("rna" in s and "adt" not in s and "protein" not in s) else ("ADT" if ("adt" in s or "protein" in s) else "UNK")
        groups.setdefault(ds, {}).setdefault(kind, []).append(p)

    report = {"root": str(root), "datasets": {}, "global_problems": [], "global_warnings": []}
    summary_rows = []

    for ds, kinds in groups.items():
        ds_entry = {"files": {k: [str(x) for x in v] for k, v in kinds.items()}, "checks": {}, "pairing": {}}

        # pairing checks
        if "RNA" not in kinds:
            ds_entry["pairing"]["problem"] = "missing RNA file"
            report["global_problems"].append(f"{ds}: missing RNA")
        if "ADT" not in kinds:
            ds_entry["pairing"]["problem"] = ds_entry["pairing"].get("problem", "") + " | missing ADT file"
            report["global_problems"].append(f"{ds}: missing ADT")

        # choose first if multiple
        rna_path = kinds.get("RNA", [None])[0]
        adt_path = kinds.get("ADT", [None])[0]

        if rna_path is not None:
            r = check_one(rna_path, "RNA")
            ds_entry["checks"]["RNA"] = r
            summary_rows.append({
                "dataset": ds, "kind": "RNA", "n_obs": r["shape"][0], "n_vars": r["shape"][1],
                "raw_neg_frac": r["raw_neg_frac"], "raw_integer_like_ratio": r["raw_integer_like_ratio"],
                "n_problems": len(r["problems"]), "n_warnings": len(r["warnings"]),
                "problems": "; ".join(r["problems"]), "warnings": "; ".join(r["warnings"]),
            })

        if adt_path is not None:
            r = check_one(adt_path, "ADT")
            ds_entry["checks"]["ADT"] = r
            summary_rows.append({
                "dataset": ds, "kind": "ADT", "n_obs": r["shape"][0], "n_vars": r["shape"][1],
                "raw_neg_frac": r["raw_neg_frac"], "raw_integer_like_ratio": r["raw_integer_like_ratio"],
                "n_problems": len(r["problems"]), "n_warnings": len(r["warnings"]),
                "problems": "; ".join(r["problems"]), "warnings": "; ".join(r["warnings"]),
            })

        # RNA/ADT alignment (if both exist)
        if (rna_path is not None) and (adt_path is not None):
            rna = ad.read_h5ad(rna_path)
            adt = ad.read_h5ad(adt_path)
            inter = rna.obs_names.intersection(adt.obs_names)
            ds_entry["pairing"].update({
                "n_rna": int(rna.n_obs),
                "n_adt": int(adt.n_obs),
                "n_intersection": int(len(inter)),
                "intersection_ratio_vs_rna": float(len(inter) / max(1, rna.n_obs)),
                "intersection_ratio_vs_adt": float(len(inter) / max(1, adt.n_obs)),
                "same_order": bool((rna.obs_names == adt.obs_names).all()) if rna.n_obs == adt.n_obs else False,
            })
            if len(inter) == 0:
                report["global_problems"].append(f"{ds}: RNA/ADT intersection=0")
            elif len(inter) != rna.n_obs or len(inter) != adt.n_obs:
                report["global_warnings"].append(f"{ds}: RNA/ADT intersection not full (check filtering)")

        report["datasets"][ds] = ds_entry

    # write outputs
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).sort_values(["dataset", "kind"]).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_csv}")

    # console summary: only show issues
    df = pd.DataFrame(summary_rows)
    bad = df[df["n_problems"] > 0]
    if len(bad):
        print("\n[PROBLEMS]")
        print(bad[["dataset","kind","n_problems","problems"]].to_string(index=False))
    warn = df[(df["n_problems"] == 0) & (df["n_warnings"] > 0)]
    if len(warn):
        print("\n[WARNINGS]")
        print(warn[["dataset","kind","n_warnings","warnings"]].to_string(index=False))


if __name__ == "__main__":
    main()

