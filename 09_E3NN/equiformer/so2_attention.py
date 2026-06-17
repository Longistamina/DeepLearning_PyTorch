import torch
import torch.nn as nn
import math

# Import the math engine we built in the previous step
from so2_math import (
    get_alignment_angles,
    rotate_to_local_frame,
    rotate_to_global_frame,
    SO2_Mixer
)

class SO2EquivariantGraphAttention(nn.Module):
    def __init__(self, lmax, channels_in, channels_q, channels_kv):
        super().__init__()
        self.lmax = lmax
        self.channels_in = channels_in
        self.channels_q = channels_q
        self.channels_kv = channels_kv

        # 1. Query Projection (Standard MLP on scalar channels)
        # We only use the l=0 (scalar) channels of the destination node to pose the Query.
        # This guarantees the resulting attention logits are strictly SE(3)-invariant.
        self.q_proj = nn.Sequential(
            nn.Linear(channels_in, channels_q),
            nn.SiLU(),
            nn.Linear(channels_q, channels_q)
        )

        # 2. Key Projection (Standard MLP on scalar channels of the mixed features)
        self.k_proj = nn.Sequential(
            nn.Linear(channels_kv, channels_q),
            nn.SiLU(),
            nn.Linear(channels_q, channels_q)
        )

        # 3. The SO(2) Mixer (Replaces the Tensor Product!)
        # It outputs channels_kv * 2 (half for Keys, half for Values)
        self.kv_mixer = SO2_Mixer(lmax, channels_in, channels_kv * 2)

    def split_by_degree(self, flat_features, channels):
        """
        Splits a flat e3nn-style tensor [N, C * (L+1)^2] into a list of [N, C, 2l+1]
        """
        features_by_l = []
        idx = 0
        for l in range(self.lmax + 1):
            dim = channels * (2 * l + 1)
            f_l = flat_features[:, idx : idx + dim]
            f_l = f_l.view(-1, channels, 2 * l + 1)
            features_by_l.append(f_l)
            idx += dim
        return features_by_l

    def flatten_degrees(self, features_by_l, channels):
        """
        Flattens a list of [N, C, 2l+1] back to e3nn-style [N, C * (L+1)^2]
        """
        flat = []
        for l in range(self.lmax + 1):
            f_l = features_by_l[l].view(-1, channels * (2 * l + 1))
            flat.append(f_l)
        return torch.cat(flat, dim=-1)

    def forward(self, node_features, edge_src, edge_dst, edge_vec, radial_weights):
        """
        Args:
            node_features: [N, C_in * (L+1)^2] (Flat equivariant features)
            edge_src:      [E]
            edge_dst:      [E]
            edge_vec:      [E, 3] (Raw 3D vectors, NO spherical harmonics needed!)
            radial_weights:[E, (L+1)^2, C_kv * 2] (Output of RadialFunction)
        """
        N = node_features.shape[0]

        # ==========================================
        # STEP 1: The SO(2) Reduction (Local Frame)
        # ==========================================
        src_feats = node_features[edge_src]
        src_feats_by_l = self.split_by_degree(src_feats, self.channels_in)

        # Calculate Euler angles to align edge_vec to the Z-axis
        alpha, beta, gamma = get_alignment_angles(edge_vec)

        # Rotate source features into the local edge frame
        local_feats_by_l = rotate_to_local_frame(src_feats_by_l, alpha, beta, gamma, self.lmax)

        # ==========================================
        # STEP 2: Channel-wise Scaling (The Mixer)
        # ==========================================
        # Apply the radial weights. Because we are in the Z-aligned frame,
        # this simple multiplication mathematically replaces the Clebsch-Gordan tables!
        mixed_local_feats_by_l = self.kv_mixer(local_feats_by_l, radial_weights)

        # ==========================================
        # STEP 3: Global Restoration
        # ==========================================
        # Rotate the mixed features back to the global 3D coordinate system
        global_feats_by_l = rotate_to_global_frame(mixed_local_feats_by_l, alpha, beta, gamma, self.lmax)

        # Flatten back to standard tensor and split into Keys and Values
        kv_flat = self.flatten_degrees(global_feats_by_l, self.channels_kv * 2)

        half_dim = kv_flat.shape[-1] // 2
        k_flat = kv_flat[:, :half_dim]  # [E, C_kv * (L+1)^2]
        v_flat = kv_flat[:, half_dim:]  # [E, C_kv * (L+1)^2]

        # ==========================================
        # STEP 4: Invariant Attention Logits
        # ==========================================
        # Extract ONLY the l=0 (scalar) channels for Q and K to guarantee invariance
        # The first `channels_in` elements of node_features are the l=0 scalars.
        dst_scalars = node_features[edge_dst, :self.channels_in]
        k_scalars = k_flat[:, :self.channels_kv]

        q = self.q_proj(dst_scalars) # [E, C_q]
        k = self.k_proj(k_scalars)   # [E, C_q]

        # Invariant Dot Product
        logits = (q * k).sum(dim=-1) / math.sqrt(self.channels_q) # [E]

        # Scatter Softmax (Group by destination node)
        logits_max = torch.zeros(N, device=logits.device).scatter_reduce_(0, edge_dst, logits, reduce='amax')
        logits_exp = torch.exp(logits - logits_max[edge_dst])
        sum_exp = torch.zeros(N, device=logits.device).scatter_reduce_(0, edge_dst, logits_exp, reduce='sum')
        alpha_attn = logits_exp / (sum_exp[edge_dst] + 1e-6) # [E]

        # ==========================================
        # STEP 5: Equivariant Aggregation
        # ==========================================
        # Weight the full equivariant Values (scalars + vectors + tensors)
        weighted_v = alpha_attn.unsqueeze(-1) * v_flat # [E, D_v]

        # Scatter Add back to destination nodes
        agg_messages = torch.zeros(N, v_flat.shape[-1], device=v_flat.device)
        idx = edge_dst.view(-1, 1).expand_as(weighted_v)
        agg_messages.scatter_add_(0, idx, weighted_v) # [N, D_v]

        return agg_messages
