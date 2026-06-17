import torch
from functools import partial

_NORM_TYPE_LIST = [
    'equivariant_layer_norm',
    'merge_layer_norm',
    'merge_layer_norm_attn_rms_norm',   # Use `EquivariantMergeLayerNorm` for the pre-norm layer
                                        # and `RMSNorm` for attention re-normalization
]

def get_normalization_layer(norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
    assert norm_type in _NORM_TYPE_LIST
    if norm_type == 'equivariant_layer_norm':
        norm_class = EquivariantLayerNorm
    elif norm_type in ['merge_layer_norm', 'merge_layer_norm_attn_rms_norm']:
        norm_class = EquivariantMergeLayerNorm
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)

##########################
## EquivariantLayerNorm ##
##########################

class EquivariantLayerNorm(torch.nn.Module):
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = torch.nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            self.affine_bias = torch.nn.Parameter(torch.zeros(self.num_channels))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, inputs):
        """1.   `inputs` shape: (num_nodes, (self.lmax + 1) ** 2, self.num_channels)"""

        outputs = []

        for l in range(self.lmax + 1):
            start_idx = l**2
            length = 2*l + 1

            feature = inputs.narrow(1, start_idx, length)

            # For scalrs, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True) # [N, 1, C]
            elif self.normalization == 'component':
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True) # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True) # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1) # [1, C]
                weight = weight.view(1, 1, -1)              # [1, 1, C]
                feature_norm = feature_norm * weight        # [N, 1, C]

            feautre = feature * feature_norm

            if self.affine and (l == 0):
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias

            outputs.append(feature)

        outputs = torch.cat(outputs, dim=-1)

        return outputs

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

    @torch.cuda.amp.autocast(enabled=False)
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
