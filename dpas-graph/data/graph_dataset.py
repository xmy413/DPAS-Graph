from __future__ import annotations

import os
import zlib
from typing import Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import coalesce

from dpas.data.io_utils import adata_to_df


DPAS_EDGE_WEIGHT_MODE = os.getenv("DPAS_EDGE_WEIGHT_MODE", "rbf")
DPAS_EDGE_FILTER_Q = float(os.getenv("DPAS_EDGE_FILTER_Q", "0.0"))
DPAS_EDGE_DROP_BASE = float(os.getenv("DPAS_EDGE_DROP_BASE", "0.0"))
DPAS_EDGE_DROP_GAMMA = float(os.getenv("DPAS_EDGE_DROP_GAMMA", "2.0"))
DPAS_W_SPATIAL = float(os.getenv("DPAS_W_SPATIAL", "1.0"))
DPAS_W_FEAT = float(os.getenv("DPAS_W_FEAT", "1.0"))
DPAS_GRAPH_SEED = int(os.getenv("DPAS_GRAPH_SEED", "0"))

DPAS_EDGE_ATTR_DIM_ENV = os.getenv("DPAS_EDGE_ATTR_DIM", None)
if DPAS_EDGE_ATTR_DIM_ENV is not None and int(DPAS_EDGE_ATTR_DIM_ENV) != 2:
    raise ValueError(
        f"DPAS_EDGE_ATTR_DIM={DPAS_EDGE_ATTR_DIM_ENV} is incompatible with DualAdaptiveEncoder; "
        "edge_attr must have 2 channels [w_spatial, w_feat]."
    )
DPAS_EDGE_ATTR_DIM = 2


def _stable_name_seed(name: str) -> int:
    name = str(name or "")
    return zlib.adler32(name.encode("utf-8")) & 0x7FFFFFFF


def cfg_tag() -> str:
    return (
        f"ed{DPAS_EDGE_ATTR_DIM}_wm{DPAS_EDGE_WEIGHT_MODE}_q{DPAS_EDGE_FILTER_Q}"
        f"_db{DPAS_EDGE_DROP_BASE}_g{DPAS_EDGE_DROP_GAMMA}"
        f"_ws{DPAS_W_SPATIAL}_wf{DPAS_W_FEAT}"
    )


def build_knn_adj(features: np.ndarray, k: int, apply_pca: bool = False, variance: float = 0.85) -> torch.Tensor:
    if apply_pca and features.shape[1] > 1500:
        pca = PCA(n_components=variance, svd_solver="full")
        features = pca.fit_transform(features)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(features)
    _, indices = nbrs.kneighbors(features)

    n = features.shape[0]
    adjacency = torch.zeros((n, n), dtype=torch.float32)
    for i, neighbors in enumerate(indices):
        adjacency[i, neighbors] = 1.0
    adjacency += torch.eye(n, dtype=torch.float32)
    return adjacency


def adjacency_to_edge_index(adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = adjacency.nonzero(as_tuple=False).t().long()
    edge_weight = adjacency[edge_index[0], edge_index[1]].to(torch.float32)
    return edge_index, edge_weight


def _coalesce_compat(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        return coalesce(edge_index, edge_attr, num_nodes=num_nodes, reduce="add")
    except TypeError:
        return coalesce(edge_index, edge_attr, num_nodes, "add")


def build_knn_edges(
    features: np.ndarray,
    k: int,
    apply_pca: bool = False,
    variance: float = 0.85,
    weight_mode: str = "rbf",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if apply_pca and features.shape[1] > 1500:
        pca = PCA(n_components=variance, svd_solver="full")
        features = pca.fit_transform(features)

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(features)
    distances, indices = nbrs.kneighbors(features)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    n = features.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = indices.reshape(-1)
    d = distances.reshape(-1).astype(np.float32)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    if weight_mode == "binary":
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    elif weight_mode == "invdist":
        edge_weight = 1.0 / (torch.tensor(d, dtype=torch.float32) + 1e-6)
    else:
        dd = torch.tensor(d, dtype=torch.float32)
        pos = dd[dd > 0]
        sigma = torch.median(pos) + 1e-6 if pos.numel() > 0 else torch.tensor(1.0)
        edge_weight = torch.exp(-(dd * dd) / (2.0 * sigma * sigma))

    self_idx = torch.arange(n, dtype=torch.long)
    self_edges = torch.stack([self_idx, self_idx], dim=0)
    self_w = torch.ones(n, dtype=torch.float32)

    edge_index = torch.cat([edge_index, self_edges], dim=1)
    edge_weight = torch.cat([edge_weight, self_w], dim=0)
    return edge_index, edge_weight


def _normalize_per_channel(edge_attr: torch.Tensor) -> torch.Tensor:
    if edge_attr.numel() == 0:
        return edge_attr
    mx = edge_attr.max(dim=0, keepdim=True).values
    return edge_attr / (mx + 1e-8)


def _edge_score(edge_attr: torch.Tensor) -> torch.Tensor:
    if edge_attr.dim() == 1:
        return edge_attr
    return edge_attr.sum(dim=1)


def _sanitize_edge_attr(edge_attr: torch.Tensor) -> torch.Tensor:
    if edge_attr.numel() == 0:
        return edge_attr
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(edge_attr, min=0.0)


def apply_weight_filter(edge_index: torch.Tensor, edge_attr: torch.Tensor, q: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if q <= 0.0:
        return edge_index, edge_attr

    is_self = (edge_index[0] == edge_index[1])
    score = _edge_score(edge_attr)

    score_nonself = score[~is_self]
    if score_nonself.numel() == 0:
        return edge_index, edge_attr

    thr = torch.quantile(score_nonself, q)
    keep = is_self | (score >= thr)
    return edge_index[:, keep], edge_attr[keep]


def apply_weight_dropout(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    drop_base: float,
    gamma: float,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if drop_base <= 0.0:
        return edge_index, edge_attr

    is_self = (edge_index[0] == edge_index[1])
    score = _edge_score(edge_attr)

    smin = score.min()
    smax = score.max()
    s01 = (score - smin) / (smax - smin + 1e-8)

    keep_prob = 1.0 - drop_base * (1.0 - (s01 ** gamma))
    keep_prob = torch.clamp(keep_prob, 0.0, 1.0)

    g = torch.Generator(device=keep_prob.device)
    g.manual_seed(int(seed))
    r = torch.rand(keep_prob.shape, generator=g, device=keep_prob.device)

    keep = (r < keep_prob) | is_self
    if int(keep.sum()) < 10:
        return edge_index, edge_attr

    return edge_index[:, keep], edge_attr[keep]


def preprocess_data(adata, pdata_df):
    X_mRNA = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    X_mRNA = np.asarray(X_mRNA, dtype=np.float32)
    X_protein = np.asarray(pdata_df.values, dtype=np.float32)
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    return X_mRNA, X_protein, spatial


def preprocess_data_no_protein(adata_p):
    X_mRNA = adata_p.X.toarray() if not isinstance(adata_p.X, np.ndarray) else adata_p.X
    X_mRNA = np.asarray(X_mRNA, dtype=np.float32)
    spatial = np.asarray(adata_p.obsm["spatial"], dtype=np.float32)
    return X_mRNA, spatial


def preprocess_data_mrna_with_target(adata, pdata_df):
    X_mRNA = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    X_mRNA = np.asarray(X_mRNA, dtype=np.float32)
    protein_target = np.asarray(pdata_df.values, dtype=np.float32)
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    return X_mRNA, protein_target, spatial


def create_pyg_data(X_mRNA: np.ndarray, X_protein: np.ndarray, spatial: np.ndarray, device: str = "cpu", sample_name: str = "") -> HeteroData:
    n = spatial.shape[0]

    e_sp, w_sp = build_knn_edges(spatial, k=6, apply_pca=False, weight_mode=DPAS_EDGE_WEIGHT_MODE)
    e_m, w_m = build_knn_edges(X_mRNA, k=10, apply_pca=True, variance=0.85, weight_mode=DPAS_EDGE_WEIGHT_MODE)
    e_p, w_p = build_knn_edges(X_protein, k=10, apply_pca=False, weight_mode=DPAS_EDGE_WEIGHT_MODE)

    e_m_all = torch.cat([e_sp, e_m], dim=1)
    attr_sp = torch.stack([DPAS_W_SPATIAL * w_sp, torch.zeros_like(w_sp)], dim=1)
    attr_m = torch.stack([torch.zeros_like(w_m), DPAS_W_FEAT * w_m], dim=1)
    a_m_all = torch.cat([attr_sp, attr_m], dim=0)
    e_m_all, a_m_all = _coalesce_compat(e_m_all, a_m_all, num_nodes=n)
    a_m_all = _sanitize_edge_attr(a_m_all)
    a_m_all = _normalize_per_channel(a_m_all)

    e_p_all = torch.cat([e_sp, e_p], dim=1)
    attr_sp = torch.stack([DPAS_W_SPATIAL * w_sp, torch.zeros_like(w_sp)], dim=1)
    attr_p = torch.stack([torch.zeros_like(w_p), DPAS_W_FEAT * w_p], dim=1)
    a_p_all = torch.cat([attr_sp, attr_p], dim=0)
    e_p_all, a_p_all = _coalesce_compat(e_p_all, a_p_all, num_nodes=n)
    a_p_all = _sanitize_edge_attr(a_p_all)
    a_p_all = _normalize_per_channel(a_p_all)

    if DPAS_EDGE_FILTER_Q > 0.0:
        e_m_all, a_m_all = apply_weight_filter(e_m_all, a_m_all, DPAS_EDGE_FILTER_Q)
        e_p_all, a_p_all = apply_weight_filter(e_p_all, a_p_all, DPAS_EDGE_FILTER_Q)

    if DPAS_EDGE_DROP_BASE > 0.0:
        seed = DPAS_GRAPH_SEED + _stable_name_seed(sample_name)
        e_m_all, a_m_all = apply_weight_dropout(e_m_all, a_m_all, DPAS_EDGE_DROP_BASE, DPAS_EDGE_DROP_GAMMA, seed=seed)
        e_p_all, a_p_all = apply_weight_dropout(e_p_all, a_p_all, DPAS_EDGE_DROP_BASE, DPAS_EDGE_DROP_GAMMA, seed=seed)

    if e_m_all.numel() == 0:
        raise ValueError("edge_index_mRNA is empty.")
    if e_p_all.numel() == 0:
        raise ValueError("edge_index_protein is empty.")

    data = HeteroData()
    data["mRNA"].x = torch.tensor(X_mRNA, dtype=torch.float32)
    data["protein"].x = torch.tensor(X_protein, dtype=torch.float32)
    data["mRNA"].pos = torch.tensor(spatial, dtype=torch.float32)
    data["protein"].pos = torch.tensor(spatial, dtype=torch.float32)

    data["mRNA", "mRNA_knn", "mRNA"].edge_index = e_m_all
    data["protein", "protein_knn", "protein"].edge_index = e_p_all
    data["mRNA", "mRNA_knn", "mRNA"].edge_attr = a_m_all
    data["protein", "protein_knn", "protein"].edge_attr = a_p_all
    return data


def create_pyg_data_no_protein(X_mRNA: np.ndarray, spatial: np.ndarray, device: str = "cpu", sample_name: str = "") -> HeteroData:
    n = spatial.shape[0]

    e_sp, w_sp = build_knn_edges(spatial, k=6, apply_pca=False, weight_mode=DPAS_EDGE_WEIGHT_MODE)
    e_m, w_m = build_knn_edges(X_mRNA, k=10, apply_pca=True, variance=0.85, weight_mode=DPAS_EDGE_WEIGHT_MODE)

    e_all = torch.cat([e_sp, e_m], dim=1)
    attr_sp = torch.stack([DPAS_W_SPATIAL * w_sp, torch.zeros_like(w_sp)], dim=1)
    attr_m = torch.stack([torch.zeros_like(w_m), DPAS_W_FEAT * w_m], dim=1)
    a_all = torch.cat([attr_sp, attr_m], dim=0)
    e_all, a_all = _coalesce_compat(e_all, a_all, num_nodes=n)
    a_all = _sanitize_edge_attr(a_all)
    a_all = _normalize_per_channel(a_all)

    if DPAS_EDGE_FILTER_Q > 0.0:
        e_all, a_all = apply_weight_filter(e_all, a_all, DPAS_EDGE_FILTER_Q)

    if DPAS_EDGE_DROP_BASE > 0.0:
        seed = DPAS_GRAPH_SEED + _stable_name_seed(sample_name)
        e_all, a_all = apply_weight_dropout(e_all, a_all, DPAS_EDGE_DROP_BASE, DPAS_EDGE_DROP_GAMMA, seed=seed)

    if e_all.numel() == 0:
        raise ValueError("edge_index_mRNA is empty.")

    data = HeteroData()
    data["mRNA"].x = torch.tensor(X_mRNA, dtype=torch.float32)
    data["mRNA"].pos = torch.tensor(spatial, dtype=torch.float32)
    data["mRNA", "mRNA_knn", "mRNA"].edge_index = e_all
    data["mRNA", "mRNA_knn", "mRNA"].edge_attr = a_all
    return data


def create_pyg_data_mrna_with_target(
    X_mRNA: np.ndarray,
    protein_target: np.ndarray,
    spatial: np.ndarray,
    device: str = "cpu",
    sample_name: str = "",
) -> HeteroData:
    data = create_pyg_data_no_protein(X_mRNA, spatial, device=device, sample_name=sample_name)
    data.protein_target = torch.tensor(protein_target, dtype=torch.float32)
    return data


def save_pyg_data(pyg_data: HeteroData, filepath: str) -> None:
    torch.save(pyg_data, filepath)


def load_pyg_data(filepath: str) -> HeteroData:
    return torch.load(filepath)


class MultiGraphDataset(Dataset):
    def __init__(self, adata_list, pdata_list, device: str = "cuda", save_dir: str = "processed_data"):
        super().__init__()
        self.data_list = []
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for adata, pdata in zip(adata_list, pdata_list):
            try:
                sample_name = list(adata.uns["spatial"].keys())[0]
            except Exception:
                sample_name = adata.uns["name"]

            filepath = os.path.join(self.save_dir, f"{sample_name}_{adata.shape[1]}_{cfg_tag()}.pth")

            if os.path.exists(filepath):
                print(f"Loading preprocessed data for sample '{sample_name}' from '{filepath}'")
                pyg_data = load_pyg_data(filepath)
            else:
                print(f"Creating and saving preprocessed data for sample '{sample_name}'")
                pdata_df = adata_to_df(pdata)
                X_mRNA, X_protein, spatial = preprocess_data(adata, pdata_df)
                pyg_data = create_pyg_data(X_mRNA, X_protein, spatial, device=device, sample_name=sample_name)
                save_pyg_data(pyg_data, filepath)

            self.data_list.append(pyg_data)

        print("Dataset ready")

    def len(self):
        return len(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class MultiGraphDataset_mRNA_with_protein_target(Dataset):
    def __init__(self, adata_list, pdata_list, device: str = "cuda", save_dir: str = "processed_data"):
        super().__init__()
        self.data_list = []
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for adata, pdata in zip(adata_list, pdata_list):
            try:
                sample_name = list(adata.uns["spatial"].keys())[0]
            except Exception:
                sample_name = adata.uns["name"]

            filepath = os.path.join(
                self.save_dir,
                f"{sample_name}_{adata.shape[1]}_{pdata.shape[1]}_{cfg_tag()}_mrna_target.pth",
            )

            if os.path.exists(filepath):
                print(f"Loading preprocessed data for sample '{sample_name}' from '{filepath}'")
                pyg_data = load_pyg_data(filepath)
            else:
                print(f"Creating and saving preprocessed data for sample '{sample_name}'")
                pdata_df = adata_to_df(pdata)
                X_mRNA, protein_target, spatial = preprocess_data_mrna_with_target(adata, pdata_df)
                pyg_data = create_pyg_data_mrna_with_target(
                    X_mRNA,
                    protein_target,
                    spatial,
                    device=device,
                    sample_name=sample_name,
                )
                save_pyg_data(pyg_data, filepath)

            self.data_list.append(pyg_data)

        print("Dataset ready")

    def len(self):
        return len(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class MultiGraphDataset_for_no_protein(Dataset):
    def __init__(self, adata_list, device: str = "cuda", save_dir: str = "processed_data"):
        super().__init__()
        self.data_list = []
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for adata_p in adata_list:
            try:
                sample_name = list(adata_p.uns["spatial"].keys())[0]
            except Exception:
                sample_name = adata_p.uns["name"]

            filepath = os.path.join(self.save_dir, f"{sample_name}_{adata_p.shape[1]}_{cfg_tag()}_no_protein.pth")

            if os.path.exists(filepath):
                print(f"Loading preprocessed data for sample '{sample_name}' from '{filepath}'")
                pyg_data = load_pyg_data(filepath)
            else:
                print(f"Creating and saving preprocessed data for sample '{sample_name}'")
                X_mRNA, spatial = preprocess_data_no_protein(adata_p)
                pyg_data = create_pyg_data_no_protein(X_mRNA, spatial, device=device, sample_name=sample_name)
                save_pyg_data(pyg_data, filepath)

            self.data_list.append(pyg_data)

        print("Dataset ready")

    def len(self):
        return len(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]