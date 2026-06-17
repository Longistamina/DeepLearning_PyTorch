import torch
from torch import Tensor
import torch.nn.functional as F

######################################
## Create BigBird mask (edge_index) ##
######################################

def create_sparse_mask(counts: Tensor, max_length: int, window_size: int, num_random: int) -> Tensor:
    B = counts.shape[0]
    device = counts.device
    r_idx = torch.arange(max_length, device=device).view(1, max_length, 1)
    c_idx = torch.arange(max_length, device=device).view(1, 1, max_length)
    valid_row = r_idx < counts.view(B, 1, 1)
    valid_col = c_idx < counts.view(B, 1, 1)
    valid_area = valid_row & valid_col  # [B, max_length, max_length]
    mask = torch.zeros((B, max_length, max_length), dtype=torch.bool, device=device)
    mask[:, 0, :] = True
    mask[:, :, 0] = True
    last_idx = torch.clamp(counts - 1, min=0).view(B, 1, 1)
    mask = mask | (r_idx == last_idx) | (c_idx == last_idx)
    window_mask = torch.abs(r_idx - c_idx) <= (window_size // 2)
    mask = mask | window_mask
    if num_random > 0:
        rand_scores = torch.rand((B, max_length, max_length), device=device)
        rand_scores.masked_fill_(~valid_col, -1.0)
        k = min(num_random, max_length)
        if k > 0:
            _, topk_idx = torch.topk(rand_scores, k, dim=2)
            rand_mask = torch.zeros((B, max_length, max_length), dtype=torch.bool, device=device)
            rand_mask.scatter_(2, topk_idx, True)
            mask = mask | rand_mask
    mask = mask & valid_area
    return mask

def mask2edge(mask: Tensor, counts: Tensor) -> Tensor:
    b, src, dst = mask.nonzero(as_tuple=True)
    offsets = F.pad(counts[:-1], pad=(1, 0), value=0).cumsum(dim=0)
    flat_src = src + offsets[b]
    flat_dst = dst + offsets[b]
    edge_index = torch.stack([flat_src, flat_dst], dim=0) # [2, E]
    return edge_index

#################
## rels2coords ##
#################

def rels2coords(rels: Tensor, graph_idx: Tensor, B: Tensor) -> Tensor:
    device = rels.device
    graph_idx_exp = graph_idx.unsqueeze(1).expand_as(rels)
    centers = torch.zeros((B, 3), device=device)
    centers.scatter_reduce_(dim=0, index=graph_idx_exp, src=rels, reduce='mean')
    rels = rels - centers[graph_idx] # Center the rel_coords (rels) to eliminate the RandomWalk
    coords = torch.cumsum(rels, dim=0)   # [N_total, 3] — global cumsum first
    centers = torch.zeros((B, 3), device=device)
    centers.scatter_reduce_(dim=0, index=graph_idx_exp, src=coords, reduce='mean')
    coords = coords - centers[graph_idx]
    return coords
