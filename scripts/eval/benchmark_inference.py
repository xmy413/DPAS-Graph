import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata as ad
import torch
from torch_geometric.loader import DataLoader

from dpas.models.dpas_graph import DualAdaptiveEncoder, Decoder_Protein_MLP
from dpas.data.graph_dataset import MultiGraphDataset_for_no_protein


hidden_dim = 1024
dropout_rate = 0.3


def _to_dense_float32(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _rna_dpas_norm_inplace(rna: ad.AnnData):
    if rna.uns.get("_dpas_input_domain", "raw") == "proc":
        rna.X = _to_dense_float32(rna.X)
        return

    if "raw" in rna.layers:
        rna.X = rna.layers["raw"].copy()
    elif rna.raw is not None:
        rna.X = rna.raw.X.copy()

    rna.X = _to_dense_float32(rna.X)

    X = rna.X
    frac_nonint = float((np.abs(X - np.round(X)) > 1e-6).mean())
    if frac_nonint > 0.01:
        sc.pp.scale(rna, max_value=10)
        return

    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna, max_value=10)


def _ensure_edge_attr_2ch(edge_attr: torch.Tensor, key: str) -> torch.Tensor:
    if edge_attr is None:
        raise ValueError(
            f"Missing edge_attr for {key}. DualAdaptiveEncoder requires edge_attr with 2 channels [w_spatial, w_feat]."
        )
    if edge_attr.dim() != 2 or edge_attr.size(-1) != 2:
        raise ValueError(f"edge_attr for {key} must have shape (E,2). Got {tuple(edge_attr.shape)}")
    return edge_attr


def _count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _load_registry_sample(specs_json: str, name: str) -> ad.AnnData:
    specs = json.loads(Path(specs_json).read_text(encoding="utf-8"))
    hit = None
    for s in specs:
        if str(s["name"]) == str(name):
            hit = s
            break
    if hit is None:
        raise ValueError(f"Dataset name not found in specs_json: {name}")

    rna = sc.read_h5ad(hit["rna"])
    rna_path = str(hit["rna"]).lower()
    rna.uns["_dpas_input_domain"] = "proc" if "proc" in Path(rna_path).name else "raw"
    rna.uns["name"] = str(name)

    if "spatial" not in rna.obsm:
        raise ValueError(f"{name}: RNA missing obsm['spatial']")

    return rna


def _load_direct_rna(rna_h5ad: str) -> ad.AnnData:
    rna = sc.read_h5ad(rna_h5ad)
    rna_path = str(rna_h5ad).lower()
    rna.uns["_dpas_input_domain"] = "proc" if "proc" in Path(rna_path).name else "raw"
    rna.uns["name"] = Path(rna_h5ad).stem

    if "spatial" not in rna.obsm:
        raise ValueError(f"{rna_h5ad}: RNA missing obsm['spatial']")

    return rna


def _load_panel(ckpt_dir: str, common_gene_path: str = None, common_protein_path: str = None):
    ckpt_dir = Path(ckpt_dir)

    if common_gene_path is None:
        common_gene_path = ckpt_dir / "common_gene.txt"
    else:
        common_gene_path = Path(common_gene_path)

    if common_protein_path is None:
        common_protein_path = ckpt_dir / "common_protein.txt"
    else:
        common_protein_path = Path(common_protein_path)

    if not common_gene_path.exists():
        raise FileNotFoundError(f"common_gene file not found: {common_gene_path}")
    if not common_protein_path.exists():
        raise FileNotFoundError(f"common_protein file not found: {common_protein_path}")

    common_gene = [x.strip() for x in common_gene_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    common_protein = [x.strip() for x in common_protein_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    return common_gene, common_protein


def _resolve_ckpt_paths(ckpt_dir: str, ckpt_prefix: str):
    ckpt_dir = Path(ckpt_dir)
    if ckpt_prefix in ("", None):
        enc_path = ckpt_dir / "encoder_mRNA.pth"
        dec_path = ckpt_dir / "decoder_protein.pth"
    else:
        enc_path = ckpt_dir / f"{ckpt_prefix}_encoder_mRNA.pth"
        dec_path = ckpt_dir / f"{ckpt_prefix}_decoder_protein.pth"

    if not enc_path.exists():
        raise FileNotFoundError(f"encoder checkpoint not found: {enc_path}")
    if not dec_path.exists():
        raise FileNotFoundError(f"decoder checkpoint not found: {dec_path}")

    return enc_path, dec_path


def _prepare_graph_dataset(
    rna: ad.AnnData,
    common_gene,
    graph_cache_dir: str,
):
    graph_cache_dir = Path(graph_cache_dir)
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    missing = set(common_gene) - set(rna.var_names)
    if missing:
        raise ValueError(f"RNA is missing {len(missing)} genes required by checkpoint panel.")

    rna = rna[:, common_gene].copy()
    _rna_dpas_norm_inplace(rna)

    t0 = time.perf_counter()
    ds = MultiGraphDataset_for_no_protein([rna], save_dir=str(graph_cache_dir))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    graph_build_time = time.perf_counter() - t0

    return rna, ds, loader, graph_build_time


def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _single_forward(loader, encoder, decoder, device):
    encoder.eval()
    decoder.eval()

    pred_np = None
    with torch.no_grad():
        for data in loader:
            m_x = data["mRNA"].x.to(device)
            m_ei = data[("mRNA", "mRNA_knn", "mRNA")].edge_index.to(device)
            m_ea = _ensure_edge_attr_2ch(data[("mRNA", "mRNA_knn", "mRNA")].edge_attr.to(device), "mRNA")

            z = encoder(m_x, m_ei, m_ea)
            prot_pred = decoder(z)
            pred_np = prot_pred.detach().cpu().numpy()

    return pred_np


def benchmark_inference(
    loader,
    encoder,
    decoder,
    device,
    n_warmup: int,
    n_runs: int,
):
    times = []

    for _ in range(n_warmup):
        _sync_if_needed(device)
        _ = _single_forward(loader, encoder, decoder, device)
        _sync_if_needed(device)

    final_pred = None
    for _ in range(n_runs):
        _sync_if_needed(device)
        t0 = time.perf_counter()
        final_pred = _single_forward(loader, encoder, decoder, device)
        _sync_if_needed(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.asarray(times, dtype=np.float64)
    return {
        "prediction": final_pred,
        "forward_total_time_sec": float(times.sum()),
        "forward_mean_time_sec": float(times.mean()),
        "forward_std_time_sec": float(times.std(ddof=0)),
        "n_runs": int(n_runs),
        "n_warmup": int(n_warmup),
    }


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--specs_json", type=str, default=None, help="Registry json path")
    source.add_argument("--rna_h5ad", type=str, default=None, help="Direct RNA h5ad path")

    ap.add_argument("--name", type=str, default=None, help="Dataset name when using --specs_json")
    ap.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory for one holdout")
    ap.add_argument("--ckpt_prefix", type=str, default="best", help="best / last / empty")
    ap.add_argument("--graph_cache_dir", type=str, required=True, help="Cache dir for graph building")
    ap.add_argument("--common_gene_path", type=str, default=None)
    ap.add_argument("--common_protein_path", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_runs", type=int, default=20)
    ap.add_argument("--save_pred", action="store_true")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    if args.specs_json is not None and not args.name:
        raise ValueError("When using --specs_json, you must also provide --name")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.specs_json is not None:
        rna = _load_registry_sample(args.specs_json, args.name)
        sample_name = str(args.name)
    else:
        rna = _load_direct_rna(args.rna_h5ad)
        sample_name = str(rna.uns.get("name", "sample"))

    common_gene, common_protein = _load_panel(
        ckpt_dir=args.ckpt_dir,
        common_gene_path=args.common_gene_path,
        common_protein_path=args.common_protein_path,
    )

    enc_path, dec_path = _resolve_ckpt_paths(args.ckpt_dir, args.ckpt_prefix)

    encoder = DualAdaptiveEncoder(
        in_channels=len(common_gene),
        hidden_dim=hidden_dim,
        dropout=dropout_rate,
    ).to(device)

    decoder = Decoder_Protein_MLP(
        hidden_dim=hidden_dim,
        proteins_or_dim=common_protein,
    ).to(device)

    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    decoder.load_state_dict(torch.load(dec_path, map_location=device))

    enc_total, enc_trainable = _count_parameters(encoder)
    dec_total, dec_trainable = _count_parameters(decoder)

    rna, ds, loader, graph_build_time = _prepare_graph_dataset(
        rna=rna,
        common_gene=common_gene,
        graph_cache_dir=args.graph_cache_dir,
    )

    result = benchmark_inference(
        loader=loader,
        encoder=encoder,
        decoder=decoder,
        device=device,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
    )

    n_spots = int(rna.n_obs)
    mean_t = result["forward_mean_time_sec"]

    summary = {
        "sample_name": sample_name,
        "device": str(device),
        "ckpt_dir": str(args.ckpt_dir),
        "ckpt_prefix": str(args.ckpt_prefix),
        "n_spots": n_spots,
        "n_genes": int(len(common_gene)),
        "n_proteins": int(len(common_protein)),
        "graph_build_time_sec": float(graph_build_time),
        "forward_total_time_sec": float(result["forward_total_time_sec"]),
        "forward_mean_time_sec": float(result["forward_mean_time_sec"]),
        "forward_std_time_sec": float(result["forward_std_time_sec"]),
        "ms_per_spot": float(mean_t * 1000.0 / max(1, n_spots)),
        "spots_per_sec": float(n_spots / max(1e-12, mean_t)),
        "n_warmup": int(result["n_warmup"]),
        "n_runs": int(result["n_runs"]),
        "encoder_total_params": int(enc_total),
        "encoder_trainable_params": int(enc_trainable),
        "decoder_total_params": int(dec_total),
        "decoder_trainable_params": int(dec_trainable),
        "model_total_params": int(enc_total + dec_total),
        "model_trainable_params": int(enc_trainable + dec_trainable),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_pred:
        pred = result["prediction"]
        if pred is not None:
            np.save(out_json.with_suffix(".pred.npy"), pred.astype(np.float32))
            (out_json.with_suffix(".protein_names.txt")).write_text(
                "\n".join(common_protein) + "\n",
                encoding="utf-8",
            )
            (out_json.with_suffix(".spot_names.txt")).write_text(
                "\n".join([str(x) for x in rna.obs_names]) + "\n",
                encoding="utf-8",
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()