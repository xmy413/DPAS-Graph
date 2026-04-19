import os
import json
import random
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dpas.data.graph_dataset import MultiGraphDataset_mRNA_with_protein_target, MultiGraphDataset_for_no_protein
from dpas.eval.metrics import evaluate_prediction_arrays, save_prediction_artifacts
from dpas.models.dpas_graph import DualAdaptiveEncoder, Decoder_mRNA_MaskedSmall, Decoder_Protein_MLP

# ==========================
# Global hyper-parameters
# ==========================
lambda_pred = 8.0
lambda_rna = 1.0
hidden_dim = 1024
dropout_rate = 0.3
batch_size = 1


def rmse_loss(pred, target, eps=1e-8):
    return torch.sqrt(nn.MSELoss()(pred, target) + eps)


def _ensure_edge_attr_2ch(edge_attr: torch.Tensor, key: str) -> torch.Tensor:
    if edge_attr is None:
        raise ValueError(
            f"Missing edge_attr for {key}. DualAdaptiveEncoder requires edge_attr with 2 channels [w_spatial, w_feat]."
        )
    if edge_attr.dim() != 2 or edge_attr.size(-1) != 2:
        raise ValueError(f"edge_attr for {key} must have shape (E,2). Got {tuple(edge_attr.shape)}")
    return edge_attr


def _write_json(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_state_bundle(save_dir, prefix, encoder_mRNA, decoder_mRNA, decoder_protein):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder_mRNA.state_dict(), os.path.join(save_dir, f"{prefix}_encoder_mRNA.pth"))
    torch.save(decoder_mRNA.state_dict(), os.path.join(save_dir, f"{prefix}_decoder_mRNA.pth"))
    torch.save(decoder_protein.state_dict(), os.path.join(save_dir, f"{prefix}_decoder_protein.pth"))


def _load_state_bundle(load_dir, prefix, encoder_mRNA, decoder_mRNA, decoder_protein, device):
    encoder_mRNA.load_state_dict(torch.load(os.path.join(load_dir, f"{prefix}_encoder_mRNA.pth"), map_location=device))
    decoder_mRNA.load_state_dict(torch.load(os.path.join(load_dir, f"{prefix}_decoder_mRNA.pth"), map_location=device))
    decoder_protein.load_state_dict(torch.load(os.path.join(load_dir, f"{prefix}_decoder_protein.pth"), map_location=device))


def _capture_state_bundle(encoder_mRNA, decoder_mRNA, decoder_protein):
    return {
        "encoder_mRNA": deepcopy({k: v.detach().cpu() for k, v in encoder_mRNA.state_dict().items()}),
        "decoder_mRNA": deepcopy({k: v.detach().cpu() for k, v in decoder_mRNA.state_dict().items()}),
        "decoder_protein": deepcopy({k: v.detach().cpu() for k, v in decoder_protein.state_dict().items()}),
    }


def _restore_state_bundle(state_bundle, encoder_mRNA, decoder_mRNA, decoder_protein, device):
    encoder_mRNA.load_state_dict({k: v.to(device) for k, v in state_bundle["encoder_mRNA"].items()})
    decoder_mRNA.load_state_dict({k: v.to(device) for k, v in state_bundle["decoder_mRNA"].items()})
    decoder_protein.load_state_dict({k: v.to(device) for k, v in state_bundle["decoder_protein"].items()})


def _save_history_csv(save_ckpt_dir, history_rows):
    if save_ckpt_dir is None:
        return
    df = pd.DataFrame(history_rows)
    df.to_csv(os.path.join(save_ckpt_dir, "metrics_history.csv"), index=False)


def _set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _resolve_init_ckpt_tag(init_ckpt_dir: str) -> str:
    init_ckpt_dir = str(init_ckpt_dir)
    preferred = ["best", "last"]
    required = ["encoder_mRNA", "decoder_mRNA", "decoder_protein"]
    for tag in preferred:
        ok = all(Path(init_ckpt_dir, f"{tag}_{name}.pth").exists() for name in required)
        if ok:
            return tag

    legacy = all(Path(init_ckpt_dir, f"{name}.pth").exists() for name in required)
    if legacy:
        return "legacy"

    raise FileNotFoundError(f"No compatible checkpoint bundle found in {init_ckpt_dir}")


def _copy_bundle_to_alias(save_dir: str, tag: str):
    names = ["encoder_mRNA", "decoder_mRNA", "decoder_protein"]
    for name in names:
        src = Path(save_dir) / f"{tag}_{name}.pth"
        if not src.exists():
            raise FileNotFoundError(f"Missing checkpoint for alias export: {src}")
        shutil.copyfile(src, Path(save_dir) / f"{name}.pth")


def _run_test_inference(
    test_loader,
    encoder_mRNA,
    decoder_mRNA,
    decoder_protein,
    export_attention: bool = False,
):
    encoder_mRNA.eval()
    decoder_mRNA.eval()
    decoder_protein.eval()

    huber = nn.SmoothL1Loss(beta=1.0)
    total_loss = 0.0
    nb = 0
    all_protein_pred, all_protein_true = [], []
    all_attention = [] if export_attention else None

    with torch.no_grad():
        for data in test_loader:
            nb += 1
            device = next(encoder_mRNA.parameters()).device

            m_x = data["mRNA"].x.to(device)
            m_ei = data[("mRNA", "mRNA_knn", "mRNA")].edge_index.to(device)
            m_ea = _ensure_edge_attr_2ch(data[("mRNA", "mRNA_knn", "mRNA")].edge_attr.to(device), "mRNA")
            p_x = data.protein_target.to(device)

            z_m = encoder_mRNA(m_x, m_ei, m_ea)
            rna_recon = decoder_mRNA(z_m)
            if export_attention:
                prot_pred, attn_w = decoder_protein(z_m, return_attn=True)
                if attn_w is not None:
                    all_attention.append(attn_w.detach().cpu().numpy())
            else:
                prot_pred = decoder_protein(z_m)

            loss_rna = rmse_loss(rna_recon, m_x)
            loss_prot_pred = huber(prot_pred, p_x)
            total = lambda_pred * loss_prot_pred + lambda_rna * loss_rna
            total_loss += float(total.detach().cpu().item())

            all_protein_pred.append(prot_pred.detach().cpu().numpy())
            all_protein_true.append(p_x.detach().cpu().numpy())

    y_pred = np.vstack(all_protein_pred).astype(np.float32)
    y_true = np.vstack(all_protein_true).astype(np.float32)
    attn = None if (all_attention is None or len(all_attention) == 0) else np.vstack(all_attention).astype(np.float32)

    return {
        "test_loss": float(total_loss / max(1, nb)),
        "y_true": y_true,
        "y_pred": y_pred,
        "attention": attn,
    }


def _save_prediction_bundle_h5ad(
    out_dir,
    test_adata: ad.AnnData,
    eval_dict: dict,
    test_sample_name: str,
    protein_names,
    bundle_info: dict = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(eval_dict["y_true"], dtype=np.float32)
    y_pred = np.asarray(eval_dict["y_pred"], dtype=np.float32)
    per_spot_df = eval_dict["per_spot_df"].copy()
    summary = dict(eval_dict["summary"])

    bundle = test_adata.copy()

    if bundle.n_obs != y_true.shape[0]:
        raise ValueError(
            f"prediction bundle obs mismatch: bundle.n_obs={bundle.n_obs}, "
            f"y_true rows={y_true.shape[0]}"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    if len(per_spot_df) != bundle.n_obs:
        raise ValueError(
            f"per_spot_df length mismatch: len(per_spot_df)={len(per_spot_df)}, "
            f"bundle.n_obs={bundle.n_obs}"
        )

    protein_names = [str(x) for x in protein_names]
    if len(protein_names) != y_true.shape[1]:
        raise ValueError(
            f"protein_names length mismatch: {len(protein_names)} vs n_proteins={y_true.shape[1]}"
        )

    # Core prediction matrices
    bundle.obsm["protein_true"] = y_true
    bundle.obsm["protein_pred"] = y_pred

    # Spot-level summary metrics
    bundle.obs["spot_rmse"] = per_spot_df["rmse"].to_numpy(dtype=np.float32)
    bundle.obs["spot_mae"] = per_spot_df["mae"].to_numpy(dtype=np.float32)
    bundle.obs["spot_pcc"] = per_spot_df["pcc"].to_numpy(dtype=np.float32)

    # Metadata for later plotting/traceability
    bundle.uns["protein_names"] = protein_names
    bundle.uns["test_name"] = str(test_sample_name)
    bundle.uns["rmse_global"] = float(summary["rmse_global"])
    bundle.uns["mae_global"] = float(summary["mae_global"])
    bundle.uns["pcc_pro_median"] = float(summary["pcc_pro_median"])
    bundle.uns["pcc_spot_median"] = float(summary["pcc_spot_median"])

    if "sp_pro_mean" in summary:
        bundle.uns["sp_pro_mean"] = float(summary["sp_pro_mean"])
    if "sp_pro_median" in summary:
        bundle.uns["sp_pro_median"] = float(summary["sp_pro_median"])

    if bundle_info is not None:
        for k, v in bundle_info.items():
            if isinstance(v, (np.floating, float)):
                bundle.uns[str(k)] = float(v)
            elif isinstance(v, (np.integer, int)):
                bundle.uns[str(k)] = int(v)
            elif isinstance(v, str):
                bundle.uns[str(k)] = v
            else:
                # keep metadata simple and robust for h5ad
                bundle.uns[str(k)] = str(v)

    out_path = out_dir / f"{test_sample_name}__prediction_bundle.h5ad"
    bundle.write_h5ad(out_path)


def evaluate_testset(
    test_adata,
    test_pdata,
    test_loader,
    encoder_mRNA,
    decoder_mRNA,
    decoder_protein,
    test_sample_name="test",
    out_dir=None,
    export_artifacts: bool = False,
    export_attention: bool = False,
    export_prediction_bundle_h5ad: bool = False,
    bundle_info: dict = None,
    include_spearman: bool = True,
    log_mode: str = "compact",
):
    infer = _run_test_inference(
        test_loader=test_loader,
        encoder_mRNA=encoder_mRNA,
        decoder_mRNA=decoder_mRNA,
        decoder_protein=decoder_protein,
        export_attention=export_attention,
    )

    eval_dict = evaluate_prediction_arrays(
        y_true=infer["y_true"],
        y_pred=infer["y_pred"],
        protein_names=list(test_pdata.var_names),
        spot_names=list(test_pdata.obs_names),
        include_spearman=include_spearman,
        include_spot_spearman=False,
    )
    metrics = dict(eval_dict["summary"])
    metrics["test_loss"] = float(infer["test_loss"])

    if export_artifacts and out_dir is not None:
        save_prediction_artifacts(
            out_dir=out_dir,
            eval_dict=eval_dict,
            sample_name=test_sample_name,
            protein_names=list(test_pdata.var_names),
            spot_names=list(test_pdata.obs_names),
            attention=infer["attention"] if export_attention else None,
            summary_extra={
                "test_loss": float(infer["test_loss"]),
            },
        )

        # keep in-memory prediction for downstream use
        test_adata.obsm["protein_predict"] = infer["y_pred"]

        # save a plotting-friendly h5ad bundle for paper figures
        if export_prediction_bundle_h5ad:
            _save_prediction_bundle_h5ad(
                out_dir=out_dir,
                test_adata=test_adata,
                eval_dict=eval_dict,
                test_sample_name=test_sample_name,
                protein_names=list(test_pdata.var_names),
                bundle_info=bundle_info,
            )

    if log_mode == "full":
        print(
            f"[Test {test_sample_name}] "
            f"rmse_global={metrics['rmse_global']:.4f} "
            f"mae_global={metrics['mae_global']:.4f} "
            f"pcc_pro_median={metrics['pcc_pro_median']:.4f} "
            f"pcc_spot_median={metrics['pcc_spot_median']:.4f} "
            f"sp_pro_mean={metrics.get('sp_pro_mean', 0.0):.4f} "
            f"loss={metrics['test_loss']:.4f}"
        )
    elif log_mode == "compact":
        print(
            f"[Test {test_sample_name}] "
            f"rmse={metrics['rmse_global']:.4f} "
            f"mae={metrics['mae_global']:.4f} "
            f"pcc_pro={metrics['pcc_pro_median']:.4f} "
            f"pcc_spot={metrics['pcc_spot_median']:.4f}"
        )

    return metrics


def train_and_evaluate_fold(
    train_adata_list,
    test_adata,
    train_pdata_list,
    test_pdata,
    test_sample_name,
    processed_data_dir,
    init_ckpt_dir=None,
    save_ckpt_dir=None,
    fixed_panel_dir=None,
    epochs=100,
    lr_scale=1.0,
    log_mode: str = "compact",
    seed: int = 0,
):
    _set_global_seed(seed)

    gene_list = [set(adata.var_names) for adata in train_adata_list + [test_adata]]
    common_gene = sorted(list(gene_list[0].intersection(*gene_list[1:])))
    protein_list = [set(pdata.var_names) for pdata in train_pdata_list + [test_pdata]]
    common_protein = sorted(list(protein_list[0].intersection(*protein_list[1:])))

    if fixed_panel_dir is not None:
        gfile = Path(fixed_panel_dir) / "common_gene.txt"
        pfile = Path(fixed_panel_dir) / "common_protein.txt"
        fixed_gene = [x.strip() for x in gfile.read_text(encoding="utf-8").splitlines() if x.strip()]
        fixed_prot = [x.strip() for x in pfile.read_text(encoding="utf-8").splitlines() if x.strip()]
        missing_g = set(fixed_gene) - set(common_gene)
        missing_p = set(fixed_prot) - set(common_protein)
        if missing_g or missing_p:
            raise ValueError(f"fixed_panel_dir has items not in current intersection: genes={len(missing_g)}, prots={len(missing_p)}")
        common_gene = fixed_gene
        common_protein = fixed_prot

    panel_dir = Path(save_ckpt_dir) if save_ckpt_dir is not None else Path(processed_data_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    Path(panel_dir / f"common_gene_{len(common_gene)}.txt").write_text("\n".join(common_gene) + "\n", encoding="utf-8")
    Path(panel_dir / f"common_protein_{len(common_protein)}.txt").write_text("\n".join(common_protein) + "\n", encoding="utf-8")

    train_adata_list = [a[:, common_gene].copy() for a in train_adata_list]
    test_adata = test_adata[:, common_gene].copy()
    train_pdata_list = [p[:, common_protein].copy() for p in train_pdata_list]
    test_pdata = test_pdata[:, common_protein].copy()

    dataset = MultiGraphDataset_mRNA_with_protein_target(train_adata_list, train_pdata_list, save_dir=processed_data_dir)
    test_dataset = MultiGraphDataset_mRNA_with_protein_target([test_adata], [test_pdata], save_dir=processed_data_dir)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=_make_loader_generator(seed))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log_mode != "silent":
        print(f"[Init] device={device}")
        print(f"[Init] panels: G={len(common_gene)}, P={len(common_protein)}")
        print(f"[Init] seed={seed}")
        print(f"[Init] loss weights: lambda_pred={lambda_pred}, lambda_rna={lambda_rna}")

    encoder_mRNA = DualAdaptiveEncoder(in_channels=len(common_gene), hidden_dim=hidden_dim, dropout=dropout_rate).to(device)
    decoder_mRNA = Decoder_mRNA_MaskedSmall(hidden_dim=hidden_dim, output_dim=len(common_gene)).to(device)
    decoder_protein = Decoder_Protein_MLP(
        hidden_dim=hidden_dim,
        proteins_or_dim=common_protein,).to(device)

    init_ckpt_tag = None
    if init_ckpt_dir is not None:
        init_ckpt_dir = str(init_ckpt_dir)
        init_ckpt_tag = _resolve_init_ckpt_tag(init_ckpt_dir)
        if init_ckpt_tag == "legacy":
            encoder_mRNA.load_state_dict(torch.load(os.path.join(init_ckpt_dir, "encoder_mRNA.pth"), map_location=device))
            decoder_mRNA.load_state_dict(torch.load(os.path.join(init_ckpt_dir, "decoder_mRNA.pth"), map_location=device))
            decoder_protein.load_state_dict(torch.load(os.path.join(init_ckpt_dir, "decoder_protein.pth"), map_location=device))
        else:
            _load_state_bundle(init_ckpt_dir, init_ckpt_tag, encoder_mRNA, decoder_mRNA, decoder_protein, device)
        if log_mode != "silent":
            print(f"[Init] loaded ckpt from {init_ckpt_dir} (tag={init_ckpt_tag})")

    huber = nn.SmoothL1Loss(beta=1.0)

    optim_enc_m = torch.optim.Adam(encoder_mRNA.parameters(), lr=5e-4 * lr_scale, weight_decay=2e-5)
    optim_dec_m = torch.optim.Adam(decoder_mRNA.parameters(), lr=1e-4 * lr_scale, weight_decay=1e-5)
    optim_dec_p = torch.optim.Adam(decoder_protein.parameters(), lr=1e-4 * lr_scale, weight_decay=1e-5)

    history = []
    best = {
        "epoch": 0,
        "test_rmse_global": float("inf"),
        "test_mae_global": float("inf"),
        "test_pcc_pro_median": -1.0,
        "test_pcc_spot_median": -1.0,
        "test_rmse_pro_mean": float("inf"),
        "test_sp_pro_mean": 0.0,
    }
    best_state = _capture_state_bundle(encoder_mRNA, decoder_mRNA, decoder_protein)

    if save_ckpt_dir is not None:
        os.makedirs(save_ckpt_dir, exist_ok=True)

    last_epoch_completed = 0

    for epoch in range(1, epochs + 1):
        encoder_mRNA.train()
        decoder_mRNA.train()
        decoder_protein.train()

        accum = {
            "total": 0.0,
            "rna": 0.0,
            "prot_pred": 0.0,
        }
        nb = 0

        for data in train_loader:
            nb += 1
            m_x = data["mRNA"].x.to(device)
            m_ei = data[("mRNA", "mRNA_knn", "mRNA")].edge_index.to(device)
            m_ea = _ensure_edge_attr_2ch(data[("mRNA", "mRNA_knn", "mRNA")].edge_attr.to(device), "mRNA")
            p_x = data.protein_target.to(device)

            z_m = encoder_mRNA(m_x, m_ei, m_ea)
            rna_recon = decoder_mRNA(z_m)
            prot_pred = decoder_protein(z_m)

            loss_rna = rmse_loss(rna_recon, m_x)
            loss_prot_pred = huber(prot_pred, p_x)
            total = lambda_pred * loss_prot_pred + lambda_rna * loss_rna

            optim_enc_m.zero_grad(set_to_none=True)
            optim_dec_m.zero_grad(set_to_none=True)
            optim_dec_p.zero_grad(set_to_none=True)

            total.backward()
            torch.nn.utils.clip_grad_norm_(encoder_mRNA.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder_mRNA.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder_protein.parameters(), 1.0)

            optim_enc_m.step()
            optim_dec_m.step()
            optim_dec_p.step()

            accum["total"] += float(total.detach().cpu().item())
            accum["rna"] += float(loss_rna.detach().cpu().item())
            accum["prot_pred"] += float(loss_prot_pred.detach().cpu().item())

        train_avg = {k: (v / max(1, nb)) for k, v in accum.items()}

        test_metrics = evaluate_testset(
            test_adata=test_adata,
            test_pdata=test_pdata,
            test_loader=test_loader,
            encoder_mRNA=encoder_mRNA,
            decoder_mRNA=decoder_mRNA,
            decoder_protein=decoder_protein,
            test_sample_name=test_sample_name,
            out_dir=None,
            export_artifacts=False,
            export_attention=False,
            include_spearman=True,
            log_mode="silent",
        )

        is_best = False
        if test_metrics["rmse_global"] < best["test_rmse_global"]:
            best = {
                "epoch": epoch,
                "test_rmse_global": test_metrics["rmse_global"],
                "test_mae_global": test_metrics["mae_global"],
                "test_pcc_pro_median": test_metrics["pcc_pro_median"],
                "test_pcc_spot_median": test_metrics["pcc_spot_median"],
                "test_rmse_pro_mean": test_metrics["rmse_pro_mean"],
                "test_sp_pro_mean": test_metrics.get("sp_pro_mean", 0.0),
            }
            best_state = _capture_state_bundle(encoder_mRNA, decoder_mRNA, decoder_protein)
            is_best = True
            if save_ckpt_dir is not None:
                _save_state_bundle(save_ckpt_dir, "best", encoder_mRNA, decoder_mRNA, decoder_protein)

        row = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_rna_recon": train_avg["rna"],
            "train_prot_pred": train_avg["prot_pred"],
            "test_loss": test_metrics["test_loss"],
            "test_rmse_global": test_metrics["rmse_global"],
            "test_mae_global": test_metrics["mae_global"],
            "test_rmse_pro_mean": test_metrics["rmse_pro_mean"],
            "test_rmse_pro_median": test_metrics["rmse_pro_median"],
            "test_pcc_pro_mean": test_metrics["pcc_pro_mean"],
            "test_pcc_pro_median": test_metrics["pcc_pro_median"],
            "test_pcc_spot_mean": test_metrics["pcc_spot_mean"],
            "test_pcc_spot_median": test_metrics["pcc_spot_median"],
            "test_sp_pro_mean": test_metrics.get("sp_pro_mean", 0.0),
            "test_sp_pro_std": test_metrics.get("sp_pro_std", 0.0),
            "test_sp_pro_median": test_metrics.get("sp_pro_median", 0.0),
            "best_rmse_global_so_far": best["test_rmse_global"],
            "best_epoch_so_far": best["epoch"],
            "is_best": is_best,
        }
        history.append(row)
        last_epoch_completed = epoch

        if save_ckpt_dir is not None:
            _save_history_csv(save_ckpt_dir, history)
            _write_json(os.path.join(save_ckpt_dir, "metrics.json"), {
                "test_name": str(test_sample_name),
                "G": int(len(common_gene)),
                "P": int(len(common_protein)),
                "best": best,
                "latest_epoch": epoch,
                "history": history,
            })

        if log_mode == "full":
            flag = " [BEST]" if is_best else ""
            print(
                f"Epoch [{epoch:03d}/{epochs}]{flag} "
                f"| train_total={row['train_total']:.4f} "
                f"| train_rna={row['train_rna_recon']:.4f} "
                f"| train_prot_pred={row['train_prot_pred']:.4f} "
                f"| test_rmse={row['test_rmse_global']:.4f} "
                f"| test_mae={row['test_mae_global']:.4f} "
                f"| pcc_pro={row['test_pcc_pro_median']:.4f} "
                f"| pcc_spot={row['test_pcc_spot_median']:.4f} "
                f"| best_rmse={row['best_rmse_global_so_far']:.4f}@{row['best_epoch_so_far']}"
            )
        elif log_mode == "compact":
            flag = " [BEST]" if is_best else ""
            print(
                f"Epoch [{epoch:03d}/{epochs}]{flag} "
                f"| train_total={row['train_total']:.4f} "
                f"| train_prot_pred={row['train_prot_pred']:.4f} "
                f"| test_rmse={row['test_rmse_global']:.4f} "
                f"| test_mae={row['test_mae_global']:.4f} "
                f"| pcc_pro={row['test_pcc_pro_median']:.4f} "
                f"| best_rmse={row['best_rmse_global_so_far']:.4f}@{row['best_epoch_so_far']}"
            )

        if epoch % 10 == 0:
            for opt in [optim_enc_m, optim_dec_m, optim_dec_p]:
                for pg in opt.param_groups:
                    pg["lr"] *= 0.8
            if log_mode == "full":
                print(f"[LR] epoch={epoch}, multiplied all learning rates by 0.8")

    if save_ckpt_dir is not None:
        _save_state_bundle(save_ckpt_dir, "last", encoder_mRNA, decoder_mRNA, decoder_protein)

    _restore_state_bundle(best_state, encoder_mRNA, decoder_mRNA, decoder_protein, device)

    final_out_dir = save_ckpt_dir if save_ckpt_dir is not None else processed_data_dir
    final_metrics = evaluate_testset(
        test_adata=test_adata,
        test_pdata=test_pdata,
        test_loader=test_loader,
        encoder_mRNA=encoder_mRNA,
        decoder_mRNA=decoder_mRNA,
        decoder_protein=decoder_protein,
        test_sample_name=test_sample_name,
        out_dir=final_out_dir,
        export_artifacts=True,
        export_attention=False,
        export_prediction_bundle_h5ad=True,
        bundle_info={
            "best_epoch": int(best["epoch"]),
            "selection_metric": "test_rmse_global",
            "seed": int(seed),
        },
        include_spearman=True,
        log_mode="silent",
    )

    metrics_summary = {
        "test_name": str(test_sample_name),
        "G": int(len(common_gene)),
        "P": int(len(common_protein)),
        "selection_metric": "test_rmse_global",
        "best": best,
        "seed": int(seed),
        "init_ckpt_tag": init_ckpt_tag,
        "loss_weights": {
            "lambda_pred": float(lambda_pred),
            "lambda_rna": float(lambda_rna),
        },
        "final": {
            "epoch": int(best["epoch"]),
            "test_loss": final_metrics["test_loss"],
            "rmse_global": final_metrics["rmse_global"],
            "mae_global": final_metrics["mae_global"],
            "rmse_pro_mean": final_metrics["rmse_pro_mean"],
            "rmse_pro_median": final_metrics["rmse_pro_median"],
            "pcc_pro_mean": final_metrics["pcc_pro_mean"],
            "pcc_pro_median": final_metrics["pcc_pro_median"],
            "pcc_spot_mean": final_metrics["pcc_spot_mean"],
            "pcc_spot_median": final_metrics["pcc_spot_median"],
            "sp_pro_mean": final_metrics.get("sp_pro_mean", 0.0),
            "sp_pro_std": final_metrics.get("sp_pro_std", 0.0),
            "sp_pro_median": final_metrics.get("sp_pro_median", 0.0),
        },
        "history": history,
    }

    if save_ckpt_dir is not None:
        _copy_bundle_to_alias(save_ckpt_dir, "best")
        Path(os.path.join(save_ckpt_dir, "common_gene.txt")).write_text("\n".join(common_gene) + "\n", encoding="utf-8")
        Path(os.path.join(save_ckpt_dir, "common_protein.txt")).write_text("\n".join(common_protein) + "\n", encoding="utf-8")
        _save_history_csv(save_ckpt_dir, history)
        _write_json(os.path.join(save_ckpt_dir, "metrics.json"), metrics_summary)

    if log_mode != "silent":
        print(
            f"[Holdout {test_sample_name}] "
            f"best@{best['epoch']}: rmse={best['test_rmse_global']:.4f} "
            f"mae={best['test_mae_global']:.4f} "
            f"pcc_pro={best['test_pcc_pro_median']:.4f} "
            f"pcc_spot={best['test_pcc_spot_median']:.4f}"
        )

    return {
        "encoder_mRNA": encoder_mRNA,
        "decoder_mRNA": decoder_mRNA,
        "decoder_protein": decoder_protein,
        "metrics": metrics_summary,
    }


def predict_data(test_adata, predict_loader, ae_mRNA, decoder_protein, common_protein):
    ae_mRNA.eval()
    decoder_protein.eval()
    device_m = next(ae_mRNA.parameters()).device
    device_p = next(decoder_protein.parameters()).device

    with torch.no_grad():
        for data in predict_loader:
            m_x = data["mRNA"].x.to(device_m)
            m_ei = data[("mRNA", "mRNA_knn", "mRNA")].edge_index.to(device_m)
            m_ea = _ensure_edge_attr_2ch(data[("mRNA", "mRNA_knn", "mRNA")].edge_attr.to(device_m), "mRNA")

            z = ae_mRNA(m_x, m_ei, m_ea)
            prot_pred = decoder_protein(z.to(device_p)).detach().cpu().numpy()
            test_adata.obsm["protein_predict"] = prot_pred

    return get_activity(test_adata, key="protein_predict", protein_names=common_protein)


def get_activity(adata, key="protein_predict", protein_names=None):
    import anndata as ad

    X = adata.obsm[key]
    obs = adata.obs.copy()
    uns = dict(adata.uns)

    s = key[:2]
    obsm = {k: v for k, v in adata.obsm.items() if s not in k}
    layers = {k: v for k, v in getattr(adata, "layers", {}).items() if s in k}

    adata_pt = ad.AnnData(X=X, obs=obs, uns=uns, obsm=obsm, layers=layers)
    if protein_names is not None:
        adata_pt.var_names = list(protein_names)
    return adata_pt


def protein_predict(model_save_dir, adata, common_gene, common_protein, ckpt_prefix: str = "last"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = DualAdaptiveEncoder(in_channels=len(common_gene), hidden_dim=hidden_dim, dropout=dropout_rate).to(device)
    dec = Decoder_Protein_MLP(
        hidden_dim=hidden_dim,
        proteins_or_dim=common_protein,).to(device)

    if ckpt_prefix in ("", None):
        enc_path = os.path.join(model_save_dir, "encoder_mRNA.pth")
        dec_path = os.path.join(model_save_dir, "decoder_protein.pth")
    else:
        enc_path = os.path.join(model_save_dir, f"{ckpt_prefix}_encoder_mRNA.pth")
        dec_path = os.path.join(model_save_dir, f"{ckpt_prefix}_decoder_protein.pth")

    enc.load_state_dict(torch.load(enc_path, map_location=device))
    dec.load_state_dict(torch.load(dec_path, map_location=device))

    ds = MultiGraphDataset_for_no_protein([adata], save_dir=model_save_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    return predict_data(adata, loader, enc, dec, common_protein)