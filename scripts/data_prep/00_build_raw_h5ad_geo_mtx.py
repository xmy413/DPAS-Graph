from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.io import mmread


def _sanitize_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _find_first(dir_path: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(dir_path.glob(pat))
        if hits:
            return hits[0]
    return None


def _read_features_tsv_gz(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, compression="gzip")
    if df.shape[1] >= 3:
        df = df.iloc[:, :3].copy()
        df.columns = ["feature_id", "feature_name", "feature_type"]
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ["feature_id", "feature_name"]
        df["feature_type"] = "Gene Expression"
    df["feature_id"] = df["feature_id"].astype(str)
    df["feature_name"] = df["feature_name"].astype(str)
    df["feature_type"] = df["feature_type"].astype(str)
    return df


def _read_positions_csv_gz(path: Path) -> pd.DataFrame:

    first = pd.read_csv(path, compression="gzip", nrows=1, header=None)
    first_row = first.iloc[0].astype(str).tolist()
    has_header = any("barcode" in s.lower() for s in first_row) or any("in_tissue" in s.lower() for s in first_row)

    if has_header:
        df = pd.read_csv(path, compression="gzip")
        df.columns = [c.strip() for c in df.columns]
        if "barcode" not in df.columns:
            raise ValueError(f"{path}: header positions but no barcode column")
    else:
        df = pd.read_csv(path, compression="gzip", header=None)
        if df.shape[1] < 6:
            raise ValueError(f"{path}: positions has <6 cols")
        df = df.iloc[:, :6].copy()
        df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]

    df["barcode"] = df["barcode"].astype(str)
    df["in_tissue"] = pd.to_numeric(df["in_tissue"], errors="coerce").fillna(0).astype(int)
    return df


def _pick_barcodes_for_matrix(
    M: sp.csr_matrix,
    n_features: int,
    posdf: pd.DataFrame,
    only_in_tissue: bool,
) -> Tuple[pd.DataFrame, str, str]:

    candidates = []
    if only_in_tissue:
        pos_tissue = posdf[posdf["in_tissue"] == 1].copy()
        candidates.append(("in_tissue==1", pos_tissue))
    candidates.append(("all_spots", posdf))

    for tag, dfcand in candidates:
        n_bc = dfcand.shape[0]
        if M.shape == (n_features, n_bc):
            return dfcand, tag, "feat_x_bc"
        if M.shape == (n_bc, n_features):
            return dfcand, tag, "bc_x_feat"

    raise ValueError(
        f"Matrix shape {M.shape} incompatible with features={n_features} "
        f"and candidates={[ (t, d.shape[0]) for t,d in candidates ]}. "
        f"通常是 matrix 与 positions 不属于同一次输出。"
    )


def _attach_spatial_from_positions(adata: ad.AnnData, pos_used: pd.DataFrame, coord: str, source: str) -> ad.AnnData:
    pos_used = pos_used.set_index("barcode").reindex(adata.obs_names)


    for c in ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]:
        if c in pos_used.columns:
            adata.obs[c] = pos_used[c].values


    coord = coord.lower().strip()
    if coord == "pixel":
        x = pd.to_numeric(pos_used["pxl_col_in_fullres"], errors="coerce").values.astype(np.float32)
        y = pd.to_numeric(pos_used["pxl_row_in_fullres"], errors="coerce").values.astype(np.float32)
    elif coord == "array":
        x = pd.to_numeric(pos_used["array_col"], errors="coerce").values.astype(np.float32)
        y = pd.to_numeric(pos_used["array_row"], errors="coerce").values.astype(np.float32)
    else:
        raise ValueError("--coord must be pixel or array")

    adata.obsm["spatial"] = np.stack([x, y], axis=1)
    adata.uns["spatial_source"] = source
    adata.uns["spatial_coord_kind"] = coord
    return adata


def _split_rna_adt_like_00(adata: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData, Dict[str, int]]:
    if "feature_types" not in adata.var.columns:
        raise ValueError("adata.var missing feature_types")

    ft_lc = adata.var["feature_types"].astype(str).str.lower()
    gene_mask = ft_lc.isin(["gene expression", "gene_expression", "gex"])
    prot_mask = ft_lc.isin(["antibody capture", "antibody_capture", "protein expression", "protein", "adt"])

    n_gene = int(gene_mask.sum())
    n_prot = int(prot_mask.sum())
    if n_gene == 0:
        raise ValueError("No Gene Expression features identified.")
    if n_prot == 0:

        non_gene = ~gene_mask
        if int(non_gene.sum()) == 0:
            raise ValueError("No ADT features identified, and no non-gene features exist.")
        prot_mask = non_gene
        n_prot = int(non_gene.sum())

    rna = adata[:, gene_mask].copy()
    adt = adata[:, prot_mask].copy()

    rna.layers["raw"] = rna.X.copy()
    adt.layers["raw"] = adt.X.copy()
    return rna, adt, {"n_gene": n_gene, "n_protein": n_prot}


def build_one_dataset(ds_dir: Path, ds_name: str, coord: str, only_in_tissue: bool) -> Tuple[ad.AnnData, ad.AnnData, Dict[str, Any]]:
    mtx_path = _find_first(ds_dir, ["*_matrix.mtx.gz", "*matrix.mtx.gz"])
    feat_path = _find_first(ds_dir, ["*_features.tsv.gz", "*features.tsv.gz"])
    pos_path = _find_first(ds_dir, ["*tissue_positions*.csv.gz"])

    if mtx_path is None or feat_path is None or pos_path is None:
        raise FileNotFoundError(f"{ds_dir}: need *_matrix.mtx.gz, *_features.tsv.gz, tissue_positions*.csv.gz")

    feats = _read_features_tsv_gz(feat_path)
    posdf = _read_positions_csv_gz(pos_path)
    M = mmread(mtx_path).tocsr()

    pos_used, tag, orient = _pick_barcodes_for_matrix(M, feats.shape[0], posdf, only_in_tissue=only_in_tissue)
    barcodes = pos_used["barcode"].astype(str).tolist()

    if orient == "feat_x_bc":
        X = M.T.tocsr()
    else:
        X = M.tocsr()

    adata = ad.AnnData(X=X)
    adata.obs_names = pd.Index(barcodes, dtype=str)
    adata.var_names = pd.Index(feats["feature_name"].astype(str).values, dtype=str)

    adata.var["gene_ids"] = feats["feature_id"].astype(str).values
    adata.var["feature_types"] = feats["feature_type"].astype(str).values
    adata.var["genome"] = "NA"


    adata.var_names_make_unique()


    adata = _attach_spatial_from_positions(
        adata,
        pos_used=pos_used,
        coord=coord,
        source=str(pos_path).replace("\\", "/"),
    )


    adata.obs["barcode_raw"] = adata.obs_names.astype(str)
    prefix = ds_name
    adata.obs_names = [f"{prefix}__{b}" for b in adata.obs["barcode_raw"].tolist()]
    adata.obs_names_make_unique()


    rna, adt, st = _split_rna_adt_like_00(adata)

    meta = {
        "name": ds_name,
        "raw_dir": str(ds_dir).replace("\\", "/"),
        "h5": str(mtx_path).replace("\\", "/"),                 
        "spatial_tar_gz": str(pos_path).replace("\\", "/"),     
        "mtx": str(mtx_path).replace("\\", "/"),
        "features": str(feat_path).replace("\\", "/"),
        "positions": str(pos_path).replace("\\", "/"),
        "barcode_source": tag,
        "orient": orient,
        **st,
    }
    return rna, adt, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build raw h5ad pairs from GEO MTX folders (00_build_raw_h5ad-compatible outputs)")
    p.add_argument("--root", type=str, required=True, help="root folder containing dataset subfolders (e.g. GSE263617)")
    p.add_argument("--out_dir", type=str, required=True, help="output dir for *_RNA_raw.h5ad and *_ADT_raw.h5ad")
    p.add_argument("--pairs_out", type=str, required=True, help="output pairs_raw_h5ad.json path")
    p.add_argument("--coord", type=str, default="pixel", choices=["pixel", "array"])
    p.add_argument("--only_in_tissue", action="store_true", help="only keep in_tissue==1 (recommended)")
    p.add_argument("--keep_off_tissue", action="store_true", help="keep off-tissue spots (not recommended)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    pairs_out = Path(args.pairs_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)

    only_in_tissue = True
    if args.keep_off_tissue:
        only_in_tissue = False
    if args.only_in_tissue:
        only_in_tissue = True

    ds_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if len(ds_dirs) == 0:
        raise RuntimeError(f"{root} has no dataset subfolders")

    specs: List[Dict[str, Any]] = []
    batch_id = 0

    print(f"[Root] {root}")
    print(f"[Out ] {out_dir}")
    print(f"[Pairs] {pairs_out}")
    print(f"[Coord] {args.coord} | only_in_tissue={only_in_tissue}")

    for ds in ds_dirs:
        ds_name = _sanitize_name(ds.name)
        print("\n" + "=" * 80)
        print(f"[DatasetDir] {ds} | name={ds_name}")

        rna, adt, meta = build_one_dataset(ds, ds_name=ds_name, coord=args.coord, only_in_tissue=only_in_tissue)
        print(f"  split: genes={meta['n_gene']} proteins={meta['n_protein']} (barcode_source={meta['barcode_source']})")

        rna_out = out_dir / f"{ds_name}_RNA_raw.h5ad"
        adt_out = out_dir / f"{ds_name}_ADT_raw.h5ad"


        rna.uns["dataset_name"] = ds_name
        adt.uns["dataset_name"] = ds_name
        rna.uns["batch_id"] = int(batch_id)
        adt.uns["batch_id"] = int(batch_id)
        rna.uns["source_h5"] = meta["h5"]
        adt.uns["source_h5"] = meta["h5"]
        rna.uns["source_spatial"] = meta["spatial_tar_gz"]
        adt.uns["source_spatial"] = meta["spatial_tar_gz"]

        rna.write_h5ad(str(rna_out))
        adt.write_h5ad(str(adt_out))
        print(f"  save:\n    RNA -> {rna_out}\n    ADT -> {adt_out}")

        specs.append(
            {
                "name": ds_name,
                "batch_id": int(batch_id),
                "rna": str(rna_out).replace("\\", "/"),
                "adt": str(adt_out).replace("\\", "/"),
                "raw_dir": meta["raw_dir"],
                "h5": meta["h5"],
                "spatial_tar_gz": meta["spatial_tar_gz"],
            }
        )
        batch_id += 1

    pairs_out.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[Done]")
    print(f"  wrote pairs json -> {pairs_out}")


if __name__ == "__main__":
    main()
