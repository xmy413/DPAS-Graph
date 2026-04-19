import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


## Internal auxiliary module：Reusable edge scoring module ##
class _EdgeMLP(nn.Module):
    '''
    Small MLP that maps 5 edge descriptors to one edge score
    '''
    def __init__(self, out_mode: str, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        assert out_mode in ("sigmoid", "softplus") # Limit the design space
        self.out_mode = out_mode
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat5: torch.Tensor) -> torch.Tensor:
        y = self.net(feat5)  # (E,1)
        if self.out_mode == "sigmoid":
            return torch.sigmoid(y)
        return F.softplus(y)

def _edge_feat5(h: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    '''Build 5 summary features for each edge from current node states and edge weight.'''
    src = edge_index[0]
    dst = edge_index[1]
    hs = h[src]
    hd = h[dst]

    # L2norms
    ns = torch.linalg.norm(hs, dim=1)
    nd = torch.linalg.norm(hd, dim=1)

    # cosine similarity
    dot = (hs * hd).sum(dim=1)
    cos = dot / (ns * nd + eps)

    # scaled difference norm
    diff = torch.linalg.norm(hs - hd, dim=1) / (h.shape[1] ** 0.5)

    return torch.stack([cos, diff, ns, nd, w], dim=1)
## ------------------------------------------------------- ##


## Model-Core: Encoder ##
class DualAdaptiveEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        dropout: float = 0.4,
        lambda_patch: float = 0.1,
        beta_cross: float = 0.2,
        prune_ratio_sp: float = 0.10,
        edge_mlp_hidden: int = 64,
        edge_mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.lambda_patch = float(lambda_patch)
        self.beta_cross = float(beta_cross)
        self.prune_ratio_sp = float(prune_ratio_sp)

        dims = [2048, 1024, int(hidden_dim)]
        in_dims = [int(in_channels)] + dims[:-1]
        out_dims = dims

        self.conv_sp = nn.ModuleList()
        self.conv_ft = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.cross_sp = nn.ModuleList()
        self.cross_ft = nn.ModuleList()
        self.edge_calib_sp = nn.ModuleList() # Spatial edge calibrator
        self.edge_calib_ft = nn.ModuleList() # Feature edge calibrator
        self.edge_patch_ft = nn.ModuleList() # Feature edge patcher

        for din, dout in zip(in_dims, out_dims):
            '''
            1.One edge-calibration module per layer for spatial edges;
            2.One edge-calibration module per layer for feature edges;
            3.A non-negative patch module is used only for the feature channel;
            4.These edge modules are layer-specific rather than shared across layers.
            '''
            self.conv_sp.append(GATv2Conv(din, dout, heads=1, concat=True, edge_dim=2, add_self_loops=False))
            self.conv_ft.append(GATv2Conv(din, dout, heads=1, concat=True, edge_dim=2, add_self_loops=False))
            self.skip.append(nn.Linear(din, dout) if din != dout else nn.Identity())
            self.norm.append(nn.LayerNorm(dout))
            self.cross_sp.append(nn.Linear(dout, dout))
            self.cross_ft.append(nn.Linear(dout, dout))
            self.edge_calib_sp.append(_EdgeMLP("sigmoid", hidden=edge_mlp_hidden, dropout=edge_mlp_dropout))
            self.edge_calib_ft.append(_EdgeMLP("sigmoid", hidden=edge_mlp_hidden, dropout=edge_mlp_dropout))
            self.edge_patch_ft.append(_EdgeMLP("softplus", hidden=edge_mlp_hidden, dropout=edge_mlp_dropout))

        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _ensure_edge_attr(edge_attr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ## Ensure edge attributes always have shape (E, 2) with channels [w_sp, w_ft]
        if edge_attr is None:
            raise ValueError("DualAdaptiveEncoder requires edge_attr with 2 columns: [w_sp, w_ft].")
        if edge_attr.dim() == 1:
            edge_attr = torch.stack([edge_attr, torch.zeros_like(edge_attr)], dim=1)
        if edge_attr.size(1) != 2:
            raise ValueError(f"edge_attr must have shape (E,2), got {tuple(edge_attr.shape)}")
        return edge_attr.to(dtype=x.dtype, device=x.device)

    @staticmethod
    def _global_prune(w: torch.Tensor, prune_ratio: float) -> torch.Tensor:
        ## Globally suppress the weakest spatial edges by zeroing their weights.
        if prune_ratio <= 0.0:
            return w
        E = w.numel()
        if E < 8:
            return w
        k = int(prune_ratio * E)
        if k <= 0:
            return w
        if k >= E:
            return torch.zeros_like(w)
        # kthvalue uses 1-indexed k
        kth = torch.kthvalue(w, k=max(1, k)).values
        return torch.where(w >= kth, w, torch.zeros_like(w))

    def forward(self, x, edge_index, edge_weight=None, return_debug: bool = False):
        edge_attr = self._ensure_edge_attr(edge_weight, x)
        w_sp0 = edge_attr[:, 0] # base edge weights
        w_ft0 = edge_attr[:, 1] # base edge weights

        debug_layers = []
        h = x

        for l in range(len(self.conv_sp)):
            # layer-wise recalibration around a fixed base
            # per-edge calibration
            feat_sp = _edge_feat5(h, edge_index, w_sp0)
            feat_ft = _edge_feat5(h, edge_index, w_ft0)

            r_sp = self.edge_calib_sp[l](feat_sp).squeeze(1)  # (E,)
            r_ft = self.edge_calib_ft[l](feat_ft).squeeze(1)

            # Calibrate the right behind
            w_sp = w_sp0 * r_sp
            w_ft = w_ft0 * r_ft

            # learned patch (feature channel only)
            if self.lambda_patch > 0:
                patch = self.edge_patch_ft[l](feat_ft).squeeze(1)
                w_ft = w_ft + self.lambda_patch * patch
            else:
                patch = torch.zeros_like(w_ft)

            # prune weakest spatial edges (optional)
            w_sp = self._global_prune(w_sp, self.prune_ratio_sp)

            # two message passing runs with the same edge_index
            edge_attr_sp = torch.stack([w_sp, torch.zeros_like(w_sp)], dim=1)
            edge_attr_ft = torch.stack([torch.zeros_like(w_ft), w_ft], dim=1)

            u_sp = self.conv_sp[l](h, edge_index, edge_attr_sp)
            u_ft = self.conv_ft[l](h, edge_index, edge_attr_ft)

            # cross-fusion residual correction (small)
            if self.beta_cross != 0.0:
                u_sp = u_sp + self.beta_cross * self.cross_sp[l](u_ft - u_sp)
                u_ft = u_ft + self.beta_cross * self.cross_ft[l](u_sp - u_ft)

            # merge + skip
            h = self.skip[l](h) + (u_sp + u_ft)
            h = self.norm[l](h)
            h = self.act(h)
            h = self.drop(h)

            if return_debug:
                with torch.no_grad():
                    # summarize (avoid large transfers)
                    def _summ(v):
                        v = v.detach()
                        return {
                            "mean": float(v.mean().item()),
                            "p10": float(torch.quantile(v, 0.10).item()),
                            "p50": float(torch.quantile(v, 0.50).item()),
                            "p90": float(torch.quantile(v, 0.90).item()),
                            "lt0.1": float((v < 0.1).float().mean().item()),
                            "gt0.9": float((v > 0.9).float().mean().item()),
                        }
                    debug_layers.append({
                        "r_sp": _summ(r_sp),
                        "r_ft": _summ(r_ft),
                        "patch": {
                            "mean": float(patch.detach().mean().item()),
                            "p90": float(torch.quantile(patch.detach(), 0.90).item()),
                        },
                        "pruned_sp_ratio": float((w_sp.detach() == 0).float().mean().item()),
                        "u_sp_norm": float(torch.linalg.norm(u_sp.detach(), dim=1).mean().item()),
                        "u_ft_norm": float(torch.linalg.norm(u_ft.detach(), dim=1).mean().item()),
                    })

        if return_debug:
            return h, {"layers": debug_layers}
        return h
## ------------------- ##

## Model-Core: Decoder ##
class Decoder_mRNA_MaskedSmall(nn.Module):
    '''
    Lightweight RNA reconstruction head used as an auxiliary constraint
    '''
    def __init__(self, hidden_dim: int, output_dim: int, hidden_small: int = 512, dropout: float = 0.1):
        super().__init__()
        hidden_small = int(min(hidden_small, max(128, hidden_dim)))  # keep it modest
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim), hidden_small),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_small, self.output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @staticmethod
    def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        m = mask.to(dtype=pred.dtype, device=pred.device)
        se = (pred - target) ** 2
        num = (se * m).sum()
        den = m.sum().clamp_min(eps)
        return num / den

class Decoder_Protein_MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        proteins_or_dim,
        hidden_mid: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        if isinstance(proteins_or_dim, (list, tuple)):
            self.proteins = list(proteins_or_dim)
            num_proteins = len(self.proteins)
        else:
            self.proteins = None
            num_proteins = int(proteins_or_dim)

        self.num_proteins = int(num_proteins)
        hidden_mid = int(min(hidden_mid, max(128, hidden_dim)))

        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim), hidden_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid, self.num_proteins),
        )

    def forward(self, z: torch.Tensor, return_attn: bool = False):
        y = self.net(z)
        if return_attn:
            return y, None
        return y