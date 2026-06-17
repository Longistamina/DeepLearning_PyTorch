import torch
import torch.nn as nn
import torch.nn.functional as F

from so2_attention import SO2EquivariantGraphAttention
from equivariant_norms import EquivariantLayerNorm

##############################################

class EquivariantFFN(nn.Module):
    """
    Pure PyTorch Equivariant Feed-Forward Network (Gated MLP).
    Replaces e3nn.nn.Gate + o3.Linear.
    """
    def __init__(self, lmax, channels_in, channels_hidden, channels_out):
        super().__init__()
        self.lmax = lmax
        self.channels_in = channels_in
        self.channels_hidden = channels_hidden
        self.channels_out = channels_out

        # 1. Scalar MLP
        # We need to output `channels_hidden` new scalars, PLUS exactly enough
        # gates to scale every single tensor channel.
        # There are `channels_in` tensor channels per degree l>0.
        num_gates = channels_in * lmax
        scalar_out_dim = channels_hidden + num_gates

        self.scalar_mlp = nn.Sequential(
            nn.Linear(channels_in, channels_hidden * 2),
            nn.SiLU(),
            nn.Linear(channels_hidden * 2, scalar_out_dim)
        )

        # 2. Channel-mixing weights for tensors (Self-Interaction)
        # One weight matrix per degree l > 0 to mix C_in -> C_out channels
        self.tensor_linears = nn.ModuleList([
            nn.Linear(channels_in, channels_out, bias=False)
            for _ in range(lmax) # l=1 to lmax
        ])

        # 3. Channel-mixing for the final scalars
        self.scalar_out_linear = nn.Linear(channels_hidden, channels_out, bias=True)

        # 4. Precompute the gate expansion index (The Magic Trick)
        # Instead of looping over tensors to apply gates, we precompute an index array
        # that duplicates the gates to match the exact flat tensor shape.
        expand_index = []
        for l in range(1, lmax + 1):
            start_idx = (l - 1) * channels_in
            indices = torch.arange(start_idx, start_idx + channels_in)
            # Repeat each gate index (2l+1) times to match the m-components
            indices = indices.repeat_interleave(2 * l + 1)
            expand_index.append(indices)

        self.register_buffer('gate_expand_index', torch.cat(expand_index))

    def forward(self, x):
        """
        x: [N, C_in * (L+1)^2] flat equivariant tensor
        Returns: [N, C_out * (L+1)^2] flat equivariant tensor
        """
        N = x.shape[0]

        # ==========================================
        # STEP 1: Split Scalars and Tensors
        # ==========================================
        # The first C_in elements are strictly l=0 (scalars)
        scalars_in = x[:, :self.channels_in]
        tensors_in = x[:, self.channels_in:]

        # ==========================================
        # STEP 2: Generate Gates and New Scalars
        # ==========================================
        h = self.scalar_mlp(scalars_in)

        scalars_hidden = h[:, :self.channels_hidden]
        gates = h[:, self.channels_hidden:]

        # Apply non-linearities
        scalars_act = F.silu(scalars_hidden)
        gates_act = torch.sigmoid(gates) # Sigmoid bounds gates between 0 and 1

        # ==========================================
        # STEP 3: Gate the Tensors (Equivariant Non-Linearity)
        # ==========================================
        # Expand gates to match the exact flat shape of tensors_in
        expanded_gates = gates_act[:, self.gate_expand_index.long()]

        # Element-wise multiplication safely scales vectors without changing direction
        gated_tensors = tensors_in * expanded_gates

        # ==========================================
        # STEP 4: Channel Mixing (Self-Interaction)
        # ==========================================
        tensors_out = []
        idx = 0
        for l in range(1, self.lmax + 1):
            dim = self.channels_in * (2 * l + 1)
            t_l = gated_tensors[:, idx : idx + dim]

            # Reshape to [N, C_in, 2l+1]
            t_l = t_l.view(N, self.channels_in, 2 * l + 1)

            # Apply Linear on the channel dimension (transpose -> linear -> transpose)
            t_l_T = t_l.transpose(1, 2)
            t_l_out_T = self.tensor_linears[l-1](t_l_T)
            t_l_out = t_l_out_T.transpose(1, 2)

            tensors_out.append(t_l_out.reshape(N, -1))
            idx += dim

        tensors_out = torch.cat(tensors_out, dim=-1)

        # Mix the scalar channels
        scalars_out = self.scalar_out_linear(scalars_act)

        # ==========================================
        # STEP 5: Recombine
        # ==========================================
        out = torch.cat([scalars_out, tensors_out], dim=-1)
        return out

############################################################################

class EquiformerBlock(nn.Module):
    def __init__(self, lmax, channels, channels_q, channels_hidden):
        super().__init__()

        # 1. Attention Branch
        self.norm1 = EquivariantLayerNorm(lmax, channels)
        self.attn = SO2EquivariantGraphAttention(
            lmax=lmax,
            channels_in=channels,
            channels_q=channels_q,
            channels_kv=channels
        )

        # 2. FFN Branch
        self.norm2 = EquivariantLayerNorm(lmax, channels)
        self.ffn = EquivariantFFN(
            lmax=lmax,
            channels_in=channels,
            channels_hidden=channels_hidden,
            channels_out=channels
        )

    def forward(self, x, edge_src, edge_dst, edge_vec, radial_weights):
        """
        x: [N, C * (L+1)^2]
        """
        # Residual 1: Attention
        x_norm1 = self.norm1(x)
        attn_out = self.attn(x_norm1, edge_src, edge_dst, edge_vec, radial_weights)
        x = x + attn_out

        # Residual 2: FFN
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out

        return x
