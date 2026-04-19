import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


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


def _adt_clr_inplace(adt: ad.AnnData):
    if adt.uns.get("_dpas_input_domain", "raw") == "proc":
        adt.X = _to_dense_float32(adt.X)
        return

    if "raw" in adt.layers:
        adt.X = adt.layers["raw"].copy()

    X = _to_dense_float32(adt.X)
    X = np.log1p(X)
    X = X - X.mean(axis=1, keepdims=True)
    adt.X = X.astype(np.float32)


def _load_from_registry(spec):
    rna = sc.read_h5ad(spec["rna"])
    adt = sc.read_h5ad(spec["adt"])
    name = str(spec["name"])

    rna_path = str(spec["rna"]).lower()
    adt_path = str(spec["adt"]).lower()

    rna.uns["_dpas_input_domain"] = "proc" if "proc" in Path(rna_path).name else "raw"
    adt.uns["_dpas_input_domain"] = "proc" if ("proc" in Path(adt_path).name or "clr" in Path(adt_path).name) else "raw"

    rna.uns["name"] = name
    adt.uns["name"] = name

    if "spatial" not in rna.obsm:
        raise ValueError(f"{name}: RNA missing obsm['spatial']")
    if "spatial" not in adt.obsm:
        adt.obsm["spatial"] = rna.obsm["spatial"].copy()

    common = rna.obs_names.intersection(adt.obs_names)
    rna = rna[common].copy()
    adt = adt[common].copy()
    return name, rna, adt


def _count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
    }


def _bundle_param_summary(encoder_mrna, decoder_mrna, decoder_protein):
    enc = _count_parameters(encoder_mrna)
    dec_rna = _count_parameters(decoder_mrna)
    dec_prot = _count_parameters(decoder_protein)

    total_all = enc["total"] + dec_rna["total"] + dec_prot["total"]
    trainable_all = enc["trainable"] + dec_rna["trainable"] + dec_prot["trainable"]

    return {
        "encoder_mRNA_total": enc["total"],
        "encoder_mRNA_trainable": enc["trainable"],
        "decoder_mRNA_total": dec_rna["total"],
        "decoder_mRNA_trainable": dec_rna["trainable"],
        "decoder_protein_total": dec_prot["total"],
        "decoder_protein_trainable": dec_prot["trainable"],
        "model_total": int(total_all),
        "model_trainable": int(trainable_all),
    }


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs_json", required=True)
    ap.add_argument("--names", default="", help="comma-separated; empty means use all names in json")
    ap.add_argument("--out_root", required=True, help="root dir for graphs/ and ckpt/")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr_scale", type=float, default=1.0)
    ap.add_argument("--init_ckpt_root", default=None, help="optional root with per-holdout ckpt dirs")
    ap.add_argument("--fixed_panel_dir", default=None)
    ap.add_argument("--log_mode", default="compact", choices=["compact", "full", "silent"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    specs = json.loads(Path(args.specs_json).read_text(encoding="utf-8"))
    name2spec = {str(s["name"]): s for s in specs}

    if args.names.strip():
        names = [x.strip() for x in args.names.split(",") if x.strip()]
    else:
        names = list(name2spec.keys())

    graph_seed = int(os.environ.get("DPAS_GRAPH_SEED", args.seed))
    os.environ["DPAS_GRAPH_SEED"] = str(graph_seed)

    out_root = Path(args.out_root)
    graphs_root = out_root / "graphs"
    ckpt_root = out_root / "ckpt"
    graphs_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    from dpas.training.trainer import train_and_evaluate_fold

    print(f"[Init] seed={args.seed} | DPAS_GRAPH_SEED={graph_seed}")

    rows = []
    for test_name in names:
        train_names = [n for n in names if n != test_name]
        print(f"=== Holdout: {test_name} | Train: {len(train_names)} ===")

        train_adata_list, train_pdata_list = [], []
        for n in train_names:
            _, rna, adt = _load_from_registry(name2spec[n])
            _rna_dpas_norm_inplace(rna)
            _adt_clr_inplace(adt)
            train_adata_list.append(rna)
            train_pdata_list.append(adt)

        _, test_rna, test_adt = _load_from_registry(name2spec[test_name])
        _rna_dpas_norm_inplace(test_rna)
        _adt_clr_inplace(test_adt)

        processed_data_dir = str(graphs_root / f"holdout_{test_name}")
        save_ckpt_dir = str(ckpt_root / f"holdout_{test_name}")
        os.makedirs(processed_data_dir, exist_ok=True)
        os.makedirs(save_ckpt_dir, exist_ok=True)

        init_ckpt_dir = None
        if args.init_ckpt_root is not None:
            cand = Path(args.init_ckpt_root) / f"holdout_{test_name}"
            if cand.exists():
                init_ckpt_dir = str(cand)

        ret = train_and_evaluate_fold(
            train_adata_list=train_adata_list,
            test_adata=test_rna,
            train_pdata_list=train_pdata_list,
            test_pdata=test_adt,
            test_sample_name=test_name,
            processed_data_dir=processed_data_dir,
            init_ckpt_dir=init_ckpt_dir,
            save_ckpt_dir=save_ckpt_dir,
            fixed_panel_dir=args.fixed_panel_dir,
            epochs=args.epochs,
            lr_scale=args.lr_scale,
            log_mode=args.log_mode,
            seed=args.seed,
        )

        m = ret["metrics"]
        param_summary = _bundle_param_summary(
            encoder_mrna=ret["encoder_mRNA"],
            decoder_mrna=ret["decoder_mRNA"],
            decoder_protein=ret["decoder_protein"],
        )

        holdout_row = {
            "test_name": m["test_name"],
            "G": m["G"],
            "P": m["P"],
            "seed": m.get("seed", args.seed),
            "graph_seed": graph_seed,
            "best_epoch": m["best"]["epoch"],
            "best_rmse_global": m["best"]["test_rmse_global"],
            "best_mae_global": m["best"]["test_mae_global"],
            "best_rmse_pro_mean": m["best"]["test_rmse_pro_mean"],
            "best_pcc_pro_median": m["best"]["test_pcc_pro_median"],
            "best_pcc_spot_median": m["best"]["test_pcc_spot_median"],
            "best_sp_pro_mean": m["best"].get("test_sp_pro_mean", 0.0),
            "final_epoch": m["final"]["epoch"],
            "final_loss": m["final"]["test_loss"],
            "final_rmse_global": m["final"]["rmse_global"],
            "final_mae_global": m["final"]["mae_global"],
            "final_rmse_pro_mean": m["final"]["rmse_pro_mean"],
            "final_rmse_pro_median": m["final"]["rmse_pro_median"],
            "final_pcc_pro_mean": m["final"]["pcc_pro_mean"],
            "final_pcc_pro_median": m["final"]["pcc_pro_median"],
            "final_pcc_spot_mean": m["final"]["pcc_spot_mean"],
            "final_pcc_spot_median": m["final"]["pcc_spot_median"],
            "final_sp_pro_mean": m["final"].get("sp_pro_mean", 0.0),
            "final_sp_pro_median": m["final"].get("sp_pro_median", 0.0),
            "selection_metric": m.get("selection_metric", "test_rmse_global"),
            **param_summary,
        }
        rows.append(holdout_row)

        pd.DataFrame([holdout_row]).to_csv(Path(save_ckpt_dir) / "lodo_row_summary.csv", index=False)

        _write_json(Path(save_ckpt_dir) / "model_param_summary.json", param_summary)

        print(
            f"[DONE] holdout={test_name} "
            f"best_rmse={holdout_row['best_rmse_global']:.4f} "
            f"final_rmse={holdout_row['final_rmse_global']:.4f} "
            f"final_mae={holdout_row['final_mae_global']:.4f} "
            f"pcc_pro={holdout_row['final_pcc_pro_median']:.4f} "
            f"pcc_spot={holdout_row['final_pcc_spot_median']:.4f} "
            f"params={holdout_row['model_total']}"
        )

    df = pd.DataFrame(rows)

    
    df = df.sort_values(["best_rmse_global", "final_rmse_global"], ascending=[True, True])

    df.to_csv(out_root / "lodo_summary.csv", index=False)

    overall = {
        "n_holdouts": int(len(df)),
        "seed": int(args.seed),
        "graph_seed": int(graph_seed),
        "mean_best_rmse_global": float(df["best_rmse_global"].mean()),
        "mean_best_mae_global": float(df["best_mae_global"].mean()),
        "mean_best_pcc_pro_median": float(df["best_pcc_pro_median"].mean()),
        "mean_best_pcc_spot_median": float(df["best_pcc_spot_median"].mean()),
        "mean_final_rmse_global": float(df["final_rmse_global"].mean()),
        "mean_final_mae_global": float(df["final_mae_global"].mean()),
        "mean_final_pcc_pro_median": float(df["final_pcc_pro_median"].mean()),
        "mean_final_pcc_spot_median": float(df["final_pcc_spot_median"].mean()),
        "median_final_rmse_global": float(df["final_rmse_global"].median()),
        "median_final_mae_global": float(df["final_mae_global"].median()),
        "median_final_pcc_pro_median": float(df["final_pcc_pro_median"].median()),
        "median_final_pcc_spot_median": float(df["final_pcc_spot_median"].median()),
    }
    _write_json(out_root / "lodo_overall_summary.json", overall)

    print("=== LoDO done ===")
    print(df[[
        "test_name",
        "best_epoch",
        "best_rmse_global",
        "best_mae_global",
        "best_pcc_pro_median",
        "best_pcc_spot_median",
        "model_total",
    ]])
    print("[Overall]")
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
