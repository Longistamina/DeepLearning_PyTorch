import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import math

from e3nn import (
    o3,
    nn as enn,
    math as emath
)

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

#-------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ SE3-Transformer --------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#

########################
## Step 0: the inputs ##
########################

hidden_dim = 32
L = 100
B = torch.tensor(2)
graph_idx = torch.tensor([0]*60 + [1]*40)
counts = torch.tensor([60, 40])

torch.manual_seed(42)
rels = torch.randn(L, 3)
pos = rels2coords(rels, graph_idx, B)

route_encoder = RouteFractionalEncoding(d_model=32)
route = route_encoder(graph_idx, counts, "cpu")

time_embedder = SinusoidalPositionEmbedding(dim=hidden_dim)
t = torch.randint(low=1, high=501, size=(B,))
t_emb = time_embedder(t)[graph_idx]

node_features = torch.cat((route, t_emb), dim=-1)

#############################################
## Step 1: Geometry and Graph Construction ##
#############################################

mask = create_sparse_mask(counts, max_length=counts.max().item(), window_size=8, num_random=16)
remove_selfloop = torch.eye(counts.max().item(), dtype=torch.bool, device="cpu").logical_not()
mask = mask & remove_selfloop
edge_index = mask2edge(mask , counts)
edge_src, edge_dst = edge_index[0], edge_index[1]

edge_vec = pos[edge_dst] - pos[edge_src]
edge_dist = edge_vec.norm(dim=-1, keepdim=True)

irreps_sh = o3.Irreps.spherical_harmonics(2)
edge_sh = o3.spherical_harmonics(l=irreps_sh, x=edge_vec, normalize=True, normalization="component")

######################################################
## Step 2: The Core SE(3) Attention Block (Q, K, V) ##
######################################################
'''
To do standard dot-product attention (α=softmax(Q⋅K), the dot product must result in a rotation-invariant scalar.
Therefore, Q and K must have the exact same geometric shape (irreps).

In Layer 1, your node features are just 1D sequence/time embeddings. These are pure Scalars (0e).
Because your nodes only contain Scalars, your Queries (Q) can only be Scalars.
Therefore, your Keys (K) must also be Scalars.

##################

But what about Values (V)?
Values don't participate in the dot product. They are just multiplied by the invariant attention weight (α)
and passed to the next node. Multiplying a 3D vector by an invariant scalar keeps it a 3D vector!

Therefore, Values (V) can (and should) contain 3D geometry (1o, 2e)
-> so the network can pass spatial information forward.
'''

# Q, querries
irreps_node_features = o3.Irreps(f"{node_features.shape[-1]}x0e")
h_q = o3.Linear(irreps_node_features, irreps_node_features)
q = h_q(node_features)

# K, V
tp = o3.FullyConnectedTensorProduct(
    irreps_in1=irreps_node_features,
    irreps_in2=irreps_sh,
    irreps_out=f"{hidden_dim}x0e + {hidden_dim}x(0e + 1o + 2e)"
)
# out shape will be [Num_Edges, Total_Dimensions]
# Keys: 32 channels of 0e = 32 dims
# Values: 32*(1 + 3 + 5) = 32 * 9 = 288 dims
# Total out dim = 32 + 288 = 320

num_weights_needed = tp.weight_numel

radial_mlp = RadialMLP(rbf_max_freq=32, time_emb_dim=64, num_weights=num_weights_needed)


#--------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ Helper utilities --------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------#

class RadialMLP(nn.Module):
    def __init__(self, rbf_max_freq, time_emb_dim, num_weights):
        super().__init__()

        # 1. RBF encoder: tunrs 1 distance scalar into 'rbf_max_freq' scalars
        self.rbf = lambda dist: emath.soft_one_hot_linspace(
            x=dist,
            start=0.0,
            end=500,
            number=rbf_max_freq,
            basis="gaussian",
            cutoff=False
        )

        # 2. Standard PyTorch MLP
        in_dim = rbf_max_freq + time_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, num_weights) # Output exactly what the TP needs!
        )

    def forward(self, edge_dist, time_emb):
        # edge_dist shape: [Num_Edges, 1]
        # time_emb shape:  [Num_Edges, time_emb_dim]
        # (Note: if time_emb is global per-graph, you must expand it to match edges first)

        rbf_feats = self.rbf(edge_dist) # Shape: [Num_Edges, rbf_max_freq]

        # Concatenate geometry (distance) and diffusion state (time)
        x = torch.cat([rbf_feats, time_emb], dim=-1)

        # Generate the weights
        weights = self.mlp(x) # Shape: [Num_Edges, num_weights_needed]
        return weights

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000.):
        super().__init__()
        self.dim  = dim
        self.base = base

    def forward(self, x):
        device   = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.base)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        if x.dim() == 1:
            emb = x[:, None].float() * emb[None, :]
        else:
            emb = x.unsqueeze(-1).float() * emb.unsqueeze(0).unsqueeze(0)

        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class RouteFractionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, f"d_model must be an even number, but got {d_model}"

    def forward(self, graph_idx, counts, device):
        N = graph_idx.shape[0]

        # 1. Calculate node count offsets
        offsets = F.pad(counts[:-1], pad=(1, 0), value=0).cumsum(dim=0)

        # 2. Get local integer positions (0, 1, 2, ..., L-1)
        global_positions = torch.arange(N, device=device)
        local_positions = global_positions - offsets[graph_idx]

        # 3. roll local_positions
        # shifts = torch.randint(low=0, high=20_000, size=counts.size()).to(device) * self.roll
        # shifts = shifts % counts
        # local_positions = (local_positions + shifts[graph_idx]) % counts[graph_idx]

        # 4. Convert to fractional positions (0.0 up to ~0.99)
        lengths_per_node = counts[graph_idx].float()
        fractional_positions = local_positions.float() / lengths_per_node

        # 5. Perfect Fourier Route Math
        half_dim = self.d_model // 2

        # Create integer harmonics: [1.0, 2.0, 3.0, ..., half_dim]
        # This guarantees that sin(2 * pi * 1.0 * k) is exactly 0.0, matching the start of the loop!
        harmonics = torch.arange(1, half_dim + 1, device=device).float()

        # Calculate angles: 2 * pi * fraction * harmonic
        # Shape: [N, half_dim]
        angles = 2 * math.pi * fractional_positions.unsqueeze(1) * harmonics.unsqueeze(0)

        pos_enc = torch.zeros(N, self.d_model, device=device)
        pos_enc[:, 0::2] = torch.sin(angles)
        pos_enc[:, 1::2] = torch.cos(angles)

        return pos_enc

class CyclicRoPE(nn.Module):
    """
    Vectorized Rotary Position Embedding for cyclic topologies.
    Uses integer harmonics to guarantee perfect 2π wrapping at boundary L.
    Zero Python loops.
    """
    def __init__(self, head_dim):
        super().__init__()
        k_freqs = torch.linspace(1, head_dim//2, steps=head_dim // 2).round().float()
        self.register_buffer('k_freqs', k_freqs)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # Faster and more memory efficient than slicing and stacking
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, src_counts, src_pad_mask):
        """
        q, k:         [B, H, L, head_dim]
        src_counts:   [B]  — actual chain length per graph
        src_pad_mask: [B, L] bool
        """
        B, H, L, D = q.shape
        device = q.device

        # 1. Node positions [1, L]
        positions = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)

        # 2. Cycle scalar per graph: (2 * pi) / N -> [B, 1]
        cycle_scalar = (2 * math.pi) / src_counts.unsqueeze(1).float()

        # 3. Base angles: (m * 2pi / N)  -> [B, L]
        base_angles = positions * cycle_scalar

        # 4. Multiply by integer frequencies -> [B, L, D/2]
        angles = base_angles.unsqueeze(-1) * self.k_freqs.unsqueeze(0).unsqueeze(0) + 1e-6

        # 5. Duplicate for both halves -> [B, L, D]
        angles = torch.cat([angles, angles], dim=-1)

        # 6. Broadcast to match [B, H, L, D] -> [B, 1, L, D]
        angles = angles.unsqueeze(1)

        # 7. Apply rotation (Padding areas will rotate, but SDPA mask ignores them later)
        cos = angles.cos()
        sin = angles.sin()

        q_out = q * cos + self._rotate_half(q) * sin
        k_out = k * cos + self._rotate_half(k) * sin

        return q_out, k_out
