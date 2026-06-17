import torch
import torch.nn as nn

######################################################

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

######################################################

class RadialMLP(nn.Module):
    def __init__(self, rbf_max_freq, features_dim, num_weights):
        super().__init__()

        # REPLACEMENT: Use GaussianSmearing instead of soft_one_hot_linspace
        self.rbf = GaussianSmearing(
            start=0.0,
            stop=50.0,          # Max expected stretched DNA bond
            num_gaussians=rbf_max_freq,
            basis_width_scalar=1.0
        )

        in_dim = rbf_max_freq + features_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, features_dim*2),
            nn.SiLU(),
            nn.Linear(features_dim*2, features_dim*2),
            nn.SiLU(),
            nn.Linear(features_dim*2, num_weights)
        )

    def forward(self, edge_dist, features):
        if edge_dist.dim() == 2 and edge_dist.shape[-1] == 1:
            edge_dist = edge_dist.squeeze(-1)

        rbf_feats = self.rbf(edge_dist) # Shape: [E, rbf_max_freq]

        x = torch.cat([rbf_feats, features], dim=-1)
        return self.mlp(x)

# radial_weights = radial_weights_flat.view(E, (self.lmax + 1)**2, self.channels_kv * 2)
