# scripts/01_stage2_make_hvgs_and_genes_model_v2.py  (Python 3.9)
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import anndata as ad

# sparse ops
from scipy import sparse


def load_specs(pairs_json: str) -> List[Dict]:
    obj = json.loads(Path(pairs_json).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "specs" in obj:
        obj = obj["specs"]
    if not isinstance(obj, list):
        raise ValueError("pairs_json must be a list or {specs:[...]}")

    specs = []
    for s in obj:
        name = s.get("name") or s.get("dataset") or s.get("id")
        if name is None:
            raise ValueError("each spec must have a 'name'")
        rna_path = s.get("rna") or s.get("rna_h5ad") or s.get("rna_path")
        if rna_path is None:
            raise ValueError(f"spec {name} missing rna path")
        specs.append({"name": str(name), "rna": str(rna_path)})
    return specs


def read_lines(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def write_lines(path: str, lines: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(list(lines)), encoding="utf-8")


def get_X_raw(rna: ad.AnnData, layer: str = "raw") -> sparse.csr_matrix:
    if layer is None:
        X = rna.X
    else:
        if layer not in rna.layers:
            raise ValueError(f"RNA h5ad missing layers['{layer}']")
        X = rna.layers[layer]
    if sparse.issparse(X):
        return X.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(X, dtype=np.float32))


def filter_in_tissue(rna: ad.AnnData) -> ad.AnnData:
    if "in_tissue" not in rna.obs.columns:
        raise ValueError("RNA obs missing 'in_tissue'")
    mask = (rna.obs["in_tissue"].astype(int).values == 1)
    return rna[mask].copy()


def normalize_total_and_log1p(X: sparse.csr_matrix, target_sum: float = 1e4, eps: float = 1e-12) -> sparse.csr_matrix:
    # library size per cell
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    lib[lib <= 0] = 1.0
    scale = (float(target_sum) / lib).astype(np.float32)  # (N,)
    Xn = X.multiply(scale[:, None])  # row scaling
    Xn.data = np.log1p(Xn.data)      # log1p on nonzero entries
    return Xn.tocsr()


def accumulate_sum_sumsq(
    X_log: sparse.csr_matrix,
    sum_vec: np.ndarray,
    sumsq_vec: np.ndarray,
) -> None:
    # sum over rows -> per gene
    s = np.asarray(X_log.sum(axis=0)).ravel()
    ss = np.asarray(X_log.power(2).sum(axis=0)).ravel()
    sum_vec += s
    sumsq_vec += ss


def seurat_dispersion_zscore(
    means: np.ndarray,
    vars_: np.ndarray,
    n_bins: int = 20,
    eps: float = 1e-12,
) -> np.ndarray:
    # dispersion
    disp = np.zeros_like(means, dtype=np.float64)
    m = means.astype(np.float64)
    v = vars_.astype(np.float64)
    ok = m > 0
    disp[ok] = v[ok] / (m[ok] + eps)

    log_disp = np.log(disp + eps)
    log_mean = np.log1p(m)

    # quantile bins on log_mean
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(log_mean, qs)
    # ensure non-decreasing edges
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    bin_id = np.digitize(log_mean, edges[1:-1], right=False)  # 0..n_bins-1

    z = np.zeros_like(log_disp, dtype=np.float64)
    for b in range(n_bins):
        idx = np.where(bin_id == b)[0]
        if idx.size == 0:
            continue
        mu = float(np.mean(log_disp[idx]))
        sd = float(np.std(log_disp[idx]))
        if sd < 1e-8:
            sd = 1.0
        z[idx] = (log_disp[idx] - mu) / sd
    return z


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_json", required=True)
    ap.add_argument("--topk", type=int, default=4000)
    ap.add_argument("--layer", type=str, default="raw")  # use RNA.layers['raw']
    ap.add_argument("--only_in_tissue", action="store_true")
    ap.add_argument("--target_sum", type=float, default=1e4)

    ap.add_argument("--hvgs_out", required=True)
    ap.add_argument("--genes_model_out", required=True)
    ap.add_argument("--stats_csv_out", default="")

    ap.add_argument("--keep_genes_path", default="")
    ap.add_argument("--keep_policy", choices=["error", "drop_missing"], default="error")  # missing in intersection

    ap.add_argument("--n_bins", type=int, default=20)
    args = ap.parse_args()

    specs = load_specs(args.pairs_json)

    # 1) compute intersection genes across datasets
    gene_inter: Optional[set] = None
    gene_order_ref: Optional[List[str]] = None
    for i, sp in enumerate(specs):
        rna = ad.read_h5ad(sp["rna"], backed="r")
        genes = list(map(str, rna.var_names))
        if i == 0:
            gene_order_ref = genes
            gene_inter = set(genes)
        else:
            gene_inter &= set(genes)

    if gene_inter is None or len(gene_inter) == 0:
        raise RuntimeError("No intersection genes across RNA datasets.")

    # Keep a deterministic order for arrays: use first dataset order
    genes_common = [g for g in gene_order_ref if g in gene_inter]
    Gc = len(genes_common)
    print(f"[Stage2] intersection genes: {Gc}")

    # 2) accumulate mean/var on log1p(normalize_total) space across all datasets
    sum_vec = np.zeros(Gc, dtype=np.float64)
    sumsq_vec = np.zeros(Gc, dtype=np.float64)
    n_total = 0

    name2pos = {g: i for i, g in enumerate(genes_common)}

    for sp in specs:
        name = sp["name"]
        rna = ad.read_h5ad(sp["rna"])
        if args.only_in_tissue:
            rna = filter_in_tissue(rna)

        # subset to genes_common (reorder)
        idx = rna.var_names.get_indexer(genes_common)
        idx = np.asarray(idx, dtype=np.int64)
        if (idx < 0).any():
            raise RuntimeError("Unexpected: gene missing after intersection check.")

        X = get_X_raw(rna, layer=args.layer)
        X = X[:, idx]  # (N, Gc)

        Xlog = normalize_total_and_log1p(X, target_sum=args.target_sum)

        accumulate_sum_sumsq(Xlog, sum_vec, sumsq_vec)
        n_total += Xlog.shape[0]
        print(f"  + {name}: N={Xlog.shape[0]}")

    if n_total <= 1:
        raise RuntimeError("Not enough total cells/spots to compute variance.")

    mean = sum_vec / n_total
    # unbiased variance
    var = (sumsq_vec - (sum_vec * sum_vec) / n_total) / max(n_total - 1, 1)
    var = np.maximum(var, 0.0)

    z = seurat_dispersion_zscore(mean, var, n_bins=int(args.n_bins))

    # 3) select topK by z-score
    topk = int(args.topk)
    if topk > Gc:
        raise ValueError(f"topk={topk} > intersection genes={Gc}")

    order = np.argsort(-z, kind="mergesort")
    hvgs = [genes_common[i] for i in order[:topk]]

    write_lines(args.hvgs_out, hvgs)
    print(f"[Stage2] wrote hvgs_top{topk}: -> {args.hvgs_out}")

    # optional stats csv
    if args.stats_csv_out:
        Path(args.stats_csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_csv_out, "w", encoding="utf-8") as f:
            f.write("gene,mean,var,z\n")
            for i in order[:min(Gc, 20000)]:  
                f.write(f"{genes_common[i]},{mean[i]},{var[i]},{z[i]}\n")
        print(f"[Stage2] wrote stats csv -> {args.stats_csv_out}")

    # 4) build genes_model.txt = hvgs + keep_genes (append unique)
    genes_model = list(hvgs)
    hvgs_set = set(hvgs)

    if args.keep_genes_path:
        keep = read_lines(args.keep_genes_path)
        missing = [g for g in keep if g not in gene_inter]
        if missing:
            if args.keep_policy == "error":
                raise ValueError(f"{len(missing)} keep genes not in intersection genes (show 10): {missing[:10]}")
            else:
                # drop missing
                keep = [g for g in keep if g in gene_inter]
                print(f"[Stage2] dropped {len(missing)} keep genes not in intersection (keep_policy=drop_missing)")

        for g in keep:
            if g not in hvgs_set and g not in genes_model:
                genes_model.append(g)

    write_lines(args.genes_model_out, genes_model)
    print(f"[Stage2] wrote genes_model: G={len(genes_model)} -> {args.genes_model_out}")
    print(f"[Stage2] note: HVGs are placed first; sim_gene_idx for Top4000 is 0..{topk-1}")


if __name__ == "__main__":
    main()
