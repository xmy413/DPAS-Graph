from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise ImportError("Need pandas for read tissue_positions*.csv") from e

try:
    import anndata as ad
except Exception as e:
    raise ImportError("Need anndata") from e

try:
    import scanpy as sc
except Exception as e:
    raise ImportError(
        "Need scanpy to read 10x H5 (sc.read_10x_h5). Please install first: pip install scanpy or conda install -c conda-forge scanpy"
    ) from e


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


def _strip_suffixes(filename: str) -> str:
    name = filename
    for suf in [
        "_filtered_feature_bc_matrix.h5",
        "filtered_feature_bc_matrix.h5",
        ".h5",
    ]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return _sanitize_name(name)


@dataclass
class SpatialInfo:
    coords: np.ndarray  # (n,2)
    obs_df: pd.DataFrame  # index=barcode，
    coord_kind: str       # 'pixel' or 'array'
    source: str           # tissue_positions*.csv


def _read_tissue_positions_from_tar(tar_gz_path: Path) -> Tuple[pd.DataFrame, str]:
    with tarfile.open(tar_gz_path, mode="r:gz") as tf:
        members = tf.getmembers()

        cand = None
        for m in members:
            n = m.name.replace("\\", "/")
            if n.endswith("spatial/tissue_positions_list.csv") or n.endswith("tissue_positions_list.csv"):
                cand = m
                break
        if cand is None:
            for m in members:
                n = m.name.replace("\\", "/")
                if n.endswith("spatial/tissue_positions.csv") or n.endswith("tissue_positions.csv"):
                    cand = m
                    break
        if cand is None:
            names = [m.name for m in members]
            raise FileNotFoundError(
                f" {tar_gz_path} cannot find tissue_positions_list.csv / tissue_positions.csv。tar 内容示例：{names[:20]}"
            )

        f = tf.extractfile(cand)
        if f is None:
            raise RuntimeError(f"cannot open file in tar {cand.name}")

        raw = f.read()
        bio = io.BytesIO(raw)


        try:
            df = pd.read_csv(bio)
            if "barcode" not in df.columns:
                raise ValueError("no barcode col")
            src = cand.name
            return df, src
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio, header=None)
            if df.shape[1] < 6:
                raise ValueError(f"{cand.name} The number of columns is insufficient (expected >= 6), actual ={df.shape[1]}")
            df = df.iloc[:, :6].copy()
            df.columns = [
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ]
            src = cand.name
            return df, src


def _build_spatial_info(
    adata: "ad.AnnData",
    spatial_tar_gz: Path,
    coord: str = "pixel",
    only_in_tissue: bool = True,
) -> SpatialInfo:

    df, src = _read_tissue_positions_from_tar(spatial_tar_gz)


    col_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["barcode", "barcodes"]:
            col_map[c] = "barcode"
        elif lc in ["in_tissue"]:
            col_map[c] = "in_tissue"
        elif lc in ["array_row"]:
            col_map[c] = "array_row"
        elif lc in ["array_col"]:
            col_map[c] = "array_col"
        elif lc in ["pxl_row_in_fullres", "pxl_row_in_fullres "]:
            col_map[c] = "pxl_row_in_fullres"
        elif lc in ["pxl_col_in_fullres", "pxl_col_in_fullres "]:
            col_map[c] = "pxl_col_in_fullres"
        elif lc in ["pxl_row_in_fullres", "pxl_row_in_fullres", "pxl_row_in_fullres"]:
            col_map[c] = "pxl_row_in_fullres"
        elif lc in ["pxl_col_in_fullres", "pxl_col_in_fullres", "pxl_col_in_fullres"]:
            col_map[c] = "pxl_col_in_fullres"

    df = df.rename(columns=col_map).copy()

    req = ["barcode", "in_tissue"]
    for r in req:
        if r not in df.columns:
            raise ValueError(f"tissue positions 缺少必要列 {r}，实际列={list(df.columns)}")


    if "barcode" not in df.columns:
        df = df.rename(columns={df.columns[0]: "barcode"})

    df["barcode"] = df["barcode"].astype(str)
    df = df.set_index("barcode", drop=True)


    common = df.index.intersection(adata.obs_names)
    if len(common) == 0:

        ex1 = list(df.index[:5])
        ex2 = list(adata.obs_names[:5])
        raise ValueError(
            f"The barcodes in tissue_positions do not match any of the barcodes in the expression matrix at all. \n"
            f"  Example of tissue_positions: {ex1}\n"
            f"  Example of matrix barcodes : {ex2}\n"
            f"  Please check for any differences in the '-1' suffix or any abnormal reading methods."
        )

    df = df.loc[common].copy()


    df["in_tissue"] = pd.to_numeric(df["in_tissue"], errors="coerce").fillna(0).astype(int)
    if only_in_tissue:
        df = df[df["in_tissue"] == 1].copy()

    if df.shape[0] == 0:
        raise ValueError("After filtering in_tissue, there were no spots (it is possible that the in_tissue column of this file is abnormal).")

    coord = coord.lower().strip()
    if coord not in ["pixel", "array"]:
        raise ValueError("--coord can only be 'pixel' or 'array'")

    if coord == "pixel":
        need = ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"tissue positions lack {c}（coord=pixel），actual row={list(df.columns)}")
        #  (x,y) = (col,row)
        x = pd.to_numeric(df["pxl_col_in_fullres"], errors="coerce")
        y = pd.to_numeric(df["pxl_row_in_fullres"], errors="coerce")
    else:
        need = ["array_row", "array_col"]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"tissue positions lack {c}（coord=array ），actual row={list(df.columns)}")
        x = pd.to_numeric(df["array_col"], errors="coerce")
        y = pd.to_numeric(df["array_row"], errors="coerce")

    coords = np.stack([x.values, y.values], axis=1).astype(np.float32)

    return SpatialInfo(coords=coords, obs_df=df, coord_kind=coord, source=src)


def _read_10x_h5_gene_protein(h5_path: Path) -> ad.AnnData:
    """read 10x filtered_feature_bc_matrix.h5（ Gene + ADT）。"""
    adata = sc.read_10x_h5(str(h5_path), gex_only=False)
    adata.var_names_make_unique()
    return adata


def _infer_feature_types(adata: ad.AnnData) -> pd.Series:
    """back feature_types Series"""
    if "feature_types" in adata.var.columns:
        ft = adata.var["feature_types"]
    elif "feature_type" in adata.var.columns:
        ft = adata.var["feature_type"]
    else:
        raise ValueError(
            "adata.var cannot find feature_types/feature_type， cannot distinguish Gene vs ADT。"
        )
    return ft.astype(str)


def _split_rna_adt(adata: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData, Dict[str, int]]:

    ft = _infer_feature_types(adata)
    ft_lc = ft.str.lower()

    gene_mask = ft_lc.isin(["gene expression", "gene_expression", "gex"])
    prot_mask = ft_lc.isin(["antibody capture", "antibody_capture", "protein expression", "protein", "adt"])

    n_gene = int(gene_mask.sum())
    n_prot = int(prot_mask.sum())

    if n_gene == 0:
        raise ValueError("cannot identify Gene Expression features（feature_types 可能不一致）。")
    if n_prot == 0:
        non_gene = ~gene_mask
        n_non_gene = int(non_gene.sum())
        if n_non_gene == 0:
            raise ValueError("cannot identify ADT features，并且除 gene 外也没有其它 features。")
        prot_mask = non_gene
        n_prot = n_non_gene

    rna = adata[:, gene_mask].copy()
    adt = adata[:, prot_mask].copy()

    rna.layers["raw"] = rna.X.copy()
    adt.layers["raw"] = adt.X.copy()

    stats = {"n_gene": n_gene, "n_protein": n_prot}
    return rna, adt, stats


def _attach_spatial(adata: ad.AnnData, sp: SpatialInfo) -> ad.AnnData:

    common = adata.obs_names.intersection(sp.obs_df.index)
    adata = adata[common, :].copy()


    obs_df = sp.obs_df.loc[adata.obs_names].copy()


    for c in obs_df.columns:
        adata.obs[c] = obs_df[c].values


    if sp.coord_kind == "pixel":
        x = pd.to_numeric(obs_df["pxl_col_in_fullres"], errors="coerce").values.astype(np.float32)
        y = pd.to_numeric(obs_df["pxl_row_in_fullres"], errors="coerce").values.astype(np.float32)
        coords = np.stack([x, y], axis=1)
    else:
        x = pd.to_numeric(obs_df["array_col"], errors="coerce").values.astype(np.float32)
        y = pd.to_numeric(obs_df["array_row"], errors="coerce").values.astype(np.float32)
        coords = np.stack([x, y], axis=1)

    adata.obsm["spatial"] = coords
    adata.uns["spatial_source"] = sp.source
    adata.uns["spatial_coord_kind"] = sp.coord_kind
    return adata


def _discover_dataset_dirs(root: Path) -> List[Path]:

    if not root.exists():
        raise FileNotFoundError(f"--root 不存在：{root}")
    ds = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        ds.append(p)
    return ds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build raw h5ad pairs from 10x Spatial-CITE-seq folders")

    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="The root directory of the original data (containing multiple subfolders of data sets), for example G:/MODEL/Data/preprocessing/Spatial-CITE-seq",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="The directory to output raw h5ad files, for example G:/MODEL/Data/preprocessing/output",
    )
    p.add_argument(
        "--pairs_out",
        type=str,
        required=True,
        help="The path to output pairs_raw_h5ad.json, for example G:/MODEL/Data/preprocessing/pairs_raw_h5ad.json",
    )
    p.add_argument(
        "--coord",
        type=str,
        default="pixel",
        choices=["pixel", "array"],
        help="The type of spatial coordinates: pixel (default) uses fullres pixel coordinates; array uses grid coordinates (array_row, array_col). Generally, pixel coordinates are more commonly used and more consistent across datasets, while array coordinates may be missing or less consistent. Please check the tissue_positions*.csv files in your datasets to decide which one to use. If you choose pixel, the code will look for columns like pxl_row_in_fullres and pxl_col_in_fullres. If you choose array, the code will look for columns like array_row and array_col.",
    )

    p.add_argument("--only_in_tissue", action="store_true", help="Only keep spots with in_tissue==1 (recommended)")
    p.add_argument("--keep_off_tissue", action="store_true", help="Keep off-tissue spots (not recommended)")


    p.add_argument("--h5_pattern", type=str, default="*filtered_feature_bc_matrix.h5")
    p.add_argument("--spatial_pattern", type=str, default="*spatial.tar.gz")

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

    ds_dirs = _discover_dataset_dirs(root)
    if len(ds_dirs) == 0:
        raise RuntimeError(f"{root} without any dataset subdirectories.")

    specs: List[Dict[str, Any]] = []
    batch_id = 0

    print(f"[Root] {root}")
    print(f"[Out ] {out_dir}")
    print(f"[Pairs] {pairs_out}")
    print(f"[Coord] {args.coord} | only_in_tissue={only_in_tissue}")

    for ds in ds_dirs:
        print("\n" + "=" * 80)
        print(f"[DatasetDir] {ds}")

        h5_path = _find_first(ds, [args.h5_pattern, "*.h5"])
        tar_path = _find_first(ds, [args.spatial_pattern, "*.tar.gz"])

        if h5_path is None:
            print("  [Skip] no .h5 found")
            continue
        if tar_path is None:
            raise FileNotFoundError(f"  Found h5 but not spatial.tar.gz: {ds}")

        prefix = _strip_suffixes(h5_path.name)
        ds_name = _sanitize_name(ds.name)

        print(f"  name   : {ds_name}")
        print(f"  h5     : {h5_path}")
        print(f"  spatial: {tar_path}")
        print(f"  prefix : {prefix}")


        adata = _read_10x_h5_gene_protein(h5_path)
        print(f"  matrix : n_obs={adata.n_obs} n_var={adata.n_vars}")


        sp = _build_spatial_info(adata, tar_path, coord=args.coord, only_in_tissue=only_in_tissue)
        print(f"  spatial: n_positions={sp.obs_df.shape[0]} (source={sp.source})")


        adata = _attach_spatial(adata, sp)
        print(f"  after spatial align: n_obs={adata.n_obs} (has obsm['spatial']={ 'spatial' in adata.obsm })")

        adata.obs["barcode_raw"] = adata.obs_names.astype(str)

        prefix = ds_name  
        adata.obs_names = [f"{prefix}__{b}" for b in adata.obs["barcode_raw"].tolist()]

        adata.obs_names_make_unique()


        rna, adt, st = _split_rna_adt(adata)
        print(f"  split  : genes={st['n_gene']} proteins={st['n_protein']}")

        rna_out = out_dir / f"{prefix}_RNA_raw.h5ad"
        adt_out = out_dir / f"{prefix}_ADT_raw.h5ad"


        rna.uns["dataset_name"] = ds_name
        adt.uns["dataset_name"] = ds_name
        rna.uns["batch_id"] = int(batch_id)
        adt.uns["batch_id"] = int(batch_id)
        rna.uns["source_h5"] = str(h5_path)
        adt.uns["source_h5"] = str(h5_path)
        rna.uns["source_spatial"] = str(tar_path)
        adt.uns["source_spatial"] = str(tar_path)

        rna.write_h5ad(str(rna_out))
        adt.write_h5ad(str(adt_out))

        print(f"  save   :\n    RNA -> {rna_out}\n    ADT -> {adt_out}")

        specs.append(
            {
                "name": ds_name,
                "batch_id": int(batch_id),
                "rna": str(rna_out).replace("\\", "/"),
                "adt": str(adt_out).replace("\\", "/"),
                "raw_dir": str(ds).replace("\\", "/"),
                "h5": str(h5_path).replace("\\", "/"),
                "spatial_tar_gz": str(tar_path).replace("\\", "/"),
            }
        )
        batch_id += 1

    if len(specs) == 0:
        raise RuntimeError("No valid datasets found; please check if there are valid dataset folders under root.")

    pairs_out.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[Done]")
    print(f"  wrote pairs json -> {pairs_out}")
    print("  Next step:")
    pairs_out_posix = str(pairs_out).replace("\\", "/")
    print(f"    python scripts/01_preprocess.py --pairs_json \"{pairs_out_posix}\" ...")


if __name__ == "__main__":
    main()
