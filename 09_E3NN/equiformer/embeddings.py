import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#################################
## SinusoidalPositionEmbedding ##
#################################

class SinusoidalPositionEmbedding(nn.Module):
    """
    Used only for timestep encoding (t → diffusion step embedding).
    This is correct usage of sinusoidal encoding because the timestep
    is an absolute scalar index, not a sequence position.
    Do NOT use this for basepair sequence positions.
    """
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


########################
## Position Embedding ##
########################

class LogBinEmbedding(nn.Module):
    def __init__(self, num_bins=512, max_dist=10000, embed_dim=48):
        super().__init__()
        self.num_bins = num_bins
        self.embed_dim = embed_dim

        # Log‑spaced bin boundaries for magnitude
        log_bounds = torch.linspace(0, torch.log(torch.tensor(max_dist, dtype=torch.float32)),
                                    steps=num_bins)
        self.register_buffer('bin_edges', torch.exp(log_bounds))  # [num_bins]

        # Embedding for magnitude bins
        self.mag_embed = nn.Embedding(num_bins, embed_dim)
        nn.init.normal_(self.mag_embed.weight, std=0.02)

        # Embedding for sign: 0 for zero, 1 for positive, 2 for negative
        self.sign_embed = nn.Embedding(3, embed_dim)
        nn.init.normal_(self.sign_embed.weight, std=0.02)

    def forward(self, signed_dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signed_dist: [E] integer tensor, signed cyclic distance in [-L//2, L//2]
        Returns:
            [E, embed_dim] combined positional embedding
        """
        # Magnitude binning
        mag = signed_dist.abs().float()
        mag_bin = torch.bucketize(mag, self.bin_edges).clamp(max=self.num_bins - 1)  # [E]

        # Sign encoding: 0 for exact zero, 1 for positive, 2 for negative
        sign = torch.where(signed_dist > 0, torch.tensor(1, device=signed_dist.device),
                           torch.where(signed_dist < 0, torch.tensor(2, device=signed_dist.device),
                                       torch.tensor(0, device=signed_dist.device)))

        mag_emb = self.mag_embed(mag_bin)
        sign_emb = self.sign_embed(sign)

        return mag_emb + sign_emb   # or torch.cat([mag_emb, sign_emb], dim=-1) if you want to double dim

class RouteFractionalEncoding(nn.Module):
    def __init__(self, d_model, roll=True):
        super().__init__()
        self.d_model = d_model
        self.roll = roll
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
