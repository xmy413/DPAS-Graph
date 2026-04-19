import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _as_float64_2d(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def _safe_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _safe_mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.abs(x - y)))


def _safe_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= eps:
        return 0.0

    r = float(np.sum(x * y) / denom)
    if not np.isfinite(r):
        return 0.0
    return max(-1.0, min(1.0, r))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    r = spearmanr(x, y)
    corr = r.correlation if hasattr(r, "correlation") else r[0]
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(corr)


def _ensure_names(names: Optional[Sequence[str]], n: int, prefix: str) -> list:
    if names is None:
        return [f"{prefix}_{i}" for i in range(n)]
    names = list(names)
    if len(names) != n:
        raise ValueError(f"Length mismatch for {prefix} names: expected {n}, got {len(names)}")
    return [str(x) for x in names]


def build_per_protein_df(
    y_true,
    y_pred,
    protein_names: Optional[Sequence[str]] = None,
    include_spearman: bool = True,
) -> pd.DataFrame:
    y_true = _as_float64_2d(y_true)
    y_pred = _as_float64_2d(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    n_spots, n_proteins = y_true.shape
    protein_names = _ensure_names(protein_names, n_proteins, "protein")

    rows = []
    for j in range(n_proteins):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        row = {
            "protein": protein_names[j],
            "rmse": _safe_rmse(yp, yt),
            "mae": _safe_mae(yp, yt),
            "pcc": _safe_pearson(yp, yt),
            "mean_true": float(np.mean(yt)),
            "std_true": float(np.std(yt)),
            "mean_pred": float(np.mean(yp)),
            "std_pred": float(np.std(yp)),
            "n_spots": int(n_spots),
        }
        if include_spearman:
            row["spearman"] = _safe_spearman(yp, yt)
        rows.append(row)

    return pd.DataFrame(rows)


def build_per_spot_df(
    y_true,
    y_pred,
    spot_names: Optional[Sequence[str]] = None,
    include_spearman: bool = False,
) -> pd.DataFrame:
    y_true = _as_float64_2d(y_true)
    y_pred = _as_float64_2d(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    n_spots, n_proteins = y_true.shape
    spot_names = _ensure_names(spot_names, n_spots, "spot")

    rows = []
    for i in range(n_spots):
        yt = y_true[i, :]
        yp = y_pred[i, :]
        row = {
            "spot": spot_names[i],
            "rmse": _safe_rmse(yp, yt),
            "mae": _safe_mae(yp, yt),
            "pcc": _safe_pearson(yp, yt),
            "n_proteins": int(n_proteins),
        }
        if include_spearman:
            row["spearman"] = _safe_spearman(yp, yt)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_prediction_metrics(
    y_true,
    y_pred,
    per_protein_df: pd.DataFrame,
    per_spot_df: pd.DataFrame,
) -> Dict[str, float]:
    y_true = _as_float64_2d(y_true)
    y_pred = _as_float64_2d(y_pred)

    summary = {
        "rmse_global": _safe_rmse(y_pred, y_true),
        "mae_global": _safe_mae(y_pred, y_true),
        "rmse_pro_mean": float(per_protein_df["rmse"].mean()),
        "rmse_pro_median": float(per_protein_df["rmse"].median()),
        "mae_pro_mean": float(per_protein_df["mae"].mean()),
        "mae_pro_median": float(per_protein_df["mae"].median()),
        "pcc_pro_mean": float(per_protein_df["pcc"].mean()),
        "pcc_pro_median": float(per_protein_df["pcc"].median()),
        "pcc_pro_std": float(per_protein_df["pcc"].std(ddof=0)),
        "pcc_spot_mean": float(per_spot_df["pcc"].mean()),
        "pcc_spot_median": float(per_spot_df["pcc"].median()),
        "pcc_spot_std": float(per_spot_df["pcc"].std(ddof=0)),
    }

    if "spearman" in per_protein_df.columns:
        summary.update({
            "sp_pro_mean": float(per_protein_df["spearman"].mean()),
            "sp_pro_median": float(per_protein_df["spearman"].median()),
            "sp_pro_std": float(per_protein_df["spearman"].std(ddof=0)),
        })

    if "spearman" in per_spot_df.columns:
        summary.update({
            "sp_spot_mean": float(per_spot_df["spearman"].mean()),
            "sp_spot_median": float(per_spot_df["spearman"].median()),
            "sp_spot_std": float(per_spot_df["spearman"].std(ddof=0)),
        })

    return summary


def evaluate_prediction_arrays(
    y_true,
    y_pred,
    protein_names: Optional[Sequence[str]] = None,
    spot_names: Optional[Sequence[str]] = None,
    include_spearman: bool = True,
    include_spot_spearman: bool = False,
) -> Dict[str, object]:
    y_true = _as_float64_2d(y_true)
    y_pred = _as_float64_2d(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    per_protein_df = build_per_protein_df(
        y_true=y_true,
        y_pred=y_pred,
        protein_names=protein_names,
        include_spearman=include_spearman,
    )

    per_spot_df = build_per_spot_df(
        y_true=y_true,
        y_pred=y_pred,
        spot_names=spot_names,
        include_spearman=include_spot_spearman,
    )

    summary = summarize_prediction_metrics(
        y_true=y_true,
        y_pred=y_pred,
        per_protein_df=per_protein_df,
        per_spot_df=per_spot_df,
    )
    summary.update({
        "n_spots": int(y_true.shape[0]),
        "n_proteins": int(y_true.shape[1]),
    })

    return {
        "summary": summary,
        "per_protein_df": per_protein_df,
        "per_spot_df": per_spot_df,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def summarize_attention(attn, protein_names: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    attn = np.asarray(attn, dtype=np.float64)
    if attn.ndim != 3:
        raise ValueError(f"Expected attention with shape (N, P, K), got {attn.shape}")

    n_spots, n_proteins, n_slots = attn.shape
    protein_names = _ensure_names(protein_names, n_proteins, "protein")
    slot_cols = [f"slot_{k}" for k in range(n_slots)]

    mean_pk = attn.mean(axis=0)
    protein_slot_df = pd.DataFrame(mean_pk, columns=slot_cols)
    protein_slot_df.insert(0, "protein", protein_names)

    eps = 1e-12
    entropy = -(attn * np.log(attn + eps)).sum(axis=2)
    protein_entropy_df = pd.DataFrame({
        "protein": protein_names,
        "attn_entropy_mean": entropy.mean(axis=0),
        "attn_entropy_std": entropy.std(axis=0),
        "attn_max_mean": attn.max(axis=2).mean(axis=0),
        "n_spots": int(n_spots),
    })

    slot_usage_df = pd.DataFrame({
        "slot": slot_cols,
        "mean_attention": attn.mean(axis=(0, 1)),
        "std_attention": attn.std(axis=(0, 1)),
        "n_spots": int(n_spots),
        "n_proteins": int(n_proteins),
    })

    return {
        "protein_slot_df": protein_slot_df,
        "protein_entropy_df": protein_entropy_df,
        "slot_usage_df": slot_usage_df,
    }


def save_prediction_artifacts(
    out_dir,
    eval_dict: Dict[str, object],
    sample_name: str,
    protein_names: Optional[Sequence[str]] = None,
    spot_names: Optional[Sequence[str]] = None,
    attention: Optional[np.ndarray] = None,
    summary_extra: Optional[Dict[str, object]] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(eval_dict["y_true"], dtype=np.float32)
    y_pred = np.asarray(eval_dict["y_pred"], dtype=np.float32)
    per_protein_df = eval_dict["per_protein_df"].copy()
    per_spot_df = eval_dict["per_spot_df"].copy()
    summary = dict(eval_dict["summary"])

    np.save(out_dir / "protein_true.npy", y_true)
    np.save(out_dir / "protein_pred.npy", y_pred)

    protein_names = _ensure_names(protein_names, y_true.shape[1], "protein")
    spot_names = _ensure_names(spot_names, y_true.shape[0], "spot")
    (out_dir / "protein_names.txt").write_text("\n".join(protein_names) + "\n", encoding="utf-8")
    (out_dir / "spot_names.txt").write_text("\n".join(spot_names) + "\n", encoding="utf-8")

    per_protein_df.sort_values(["rmse", "pcc"], ascending=[False, True]).to_csv(
        out_dir / f"{sample_name}__per_protein.csv", index=False
    )
    per_protein_df.to_csv(out_dir / "per_protein_metrics.csv", index=False)
    per_spot_df.to_csv(out_dir / "per_spot_metrics.csv", index=False)

    summary_payload = {
        "test_name": str(sample_name),
        **summary,
    }
    if summary_extra is not None:
        summary_payload.update(summary_extra)

    pd.DataFrame([summary_payload]).to_csv(out_dir / "metrics_summary.csv", index=False)
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if attention is not None:
        attention = np.asarray(attention, dtype=np.float32)
        np.save(out_dir / "attention.npy", attention)

        attn_summary = summarize_attention(attention, protein_names=protein_names)
        attn_summary["protein_slot_df"].to_csv(out_dir / "attention_protein_slot_mean.csv", index=False)
        attn_summary["protein_entropy_df"].to_csv(out_dir / "attention_protein_entropy.csv", index=False)
        attn_summary["slot_usage_df"].to_csv(out_dir / "attention_slot_usage.csv", index=False)