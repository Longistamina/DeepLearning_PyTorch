import torch
import torch.nn as nn

##########################
## EquivariantLayerNorm ##
##########################

class EquivariantLayerNorm(nn.Module):
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            # One weight per degree l, applied to all C channels
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: [N, C * (L+1)^2] flat layout
        N = x.shape[0]
        out = []
        idx = 0

        for l in range(self.lmax + 1):
            dim = self.num_channels * (2 * l + 1)
            f_l = x[:, idx : idx + dim]

            # Reshape to [N, Channels, m_components]
            f_l = f_l.view(N, self.num_channels, 2 * l + 1)

            if l == 0:
                # ==========================================
                # SCALARS (l=0): Standard LayerNorm
                # Normalize across the channel dimension (dim=1)
                # ==========================================
                mean = f_l.mean(dim=1, keepdim=True)
                var = f_l.var(dim=1, keepdim=True, unbiased=False)
                f_l = (f_l - mean) / torch.sqrt(var + self.eps)

                if self.affine:
                    weight = self.affine_weight[l].view(1, self.num_channels, 1)
                    bias = self.affine_bias.view(1, self.num_channels, 1)
                    f_l = (f_l * weight) + bias

            else:
                # ==========================================
                # VECTORS/TENSORS (l>0): Equivariant RMSNorm
                # Normalize across the spatial m-components (dim=2)
                # ==========================================
                feature_norm = f_l.pow(2).mean(dim=2, keepdim=True) # [N, C, 1]
                feature_norm = (feature_norm + self.eps).pow(-0.5)

                if self.affine:
                    weight = self.affine_weight[l].view(1, self.num_channels, 1)
                    feature_norm = feature_norm * weight

                f_l = f_l * feature_norm

            out.append(f_l.view(N, -1))
            idx += dim

        return torch.cat(out, dim=-1)

###############################
## EquivariantMergeLayerNorm ##
###############################

class EquivariantMergeLayerNorm(torch.nn.Module):
    """
        1. Use `expand_index` to skip for loop during affine transformation.
        2.  Different from `EquivariantSeparableLayerNorm`, we normalize over all degrees L >= 0.
        3.  If `centering == False`, this becomes RMSNorm for all degrees.
    """
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', std_balance_degrees=True, centering=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.centering = centering

        if self.affine:
            self.affine_weight = torch.nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long() # L >= 0
                                                                        # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ....])
            for l in range(self.lmax + 1):
                start_idx = l**2
                length = 2*l + 1
                expand_index[start_idx : (start_idx + length)] = l # [0, 1, 1, 1, 2, 2, 2, 2, 2, ...]

            self.register_buffer('expand_index', expand_index)

            if self.centering:
                self.affine_bias = torch.nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter('affine_bias', None)
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length) # [1, 1/3, 1/3, 1/3, 1/5, 1/5, 1/5, 1/5, 1/5, ...], [I, 1]
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            balance_degree_weight = balance_degree_weight.permute((1, 0)) # [1, I]
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees}, centering={self.centering})"

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, inputs):
        """
            1.  `inputs` shape: (num_nodes, (self.lmax + 1) ** 2, self.num_channels)
        """
        # for L = 0 (scalars)
        if self.centering:
            scalars = inputs.narrow(1, 0, 1)
            scalars_mean = scalars.mean(dim=2, keepdim=True) # [N, 1, 1]
            scalars = scalars - scalars_mean
            inputs = torch.cat([scalars, inputs.narrow(1, 1, inputs.shape[1] - 1)], dim=1)

        # for L >= 0 (vectors, tensors)
        feature_norm = inputs.pow(2)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True) # [N, (L_max +1)**2, 1]
        if self.normalization == 'norm':
            feature_norm = feature_norm.sum(dim=1, keepdim=True)
        elif self.normalization == 'component':
            if self.std_balance_degrees:
                feature_norm = torch.einsum('ai, nic -> nac', self.balance_degree_weight, feature_norm) # [N, 1, 1]
            else:
                feature_norm = feature_norm.mean(dim=1, keepdim=True) # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(1, (self.lmax + 1), self.num_channels)
            weight = torch.index_select(weight, dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight
        outputs = inputs * feature_norm

        if self.affine and self.centering:
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)

        return outputs
