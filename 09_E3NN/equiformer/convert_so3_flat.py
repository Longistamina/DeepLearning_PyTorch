import torch

def flat_to_so3(flat_features, channels, lmax):
    """Converts [N, C * (L+1)^2] to [N, (L+1)^2, C]"""
    N = flat_features.shape[0]
    so3_features = []
    idx = 0
    for l in range(lmax + 1):
        dim = channels * (2 * l + 1)
        f_l = flat_features[:, idx : idx + dim]
        f_l = f_l.view(N, channels, 2 * l + 1)
        f_l = f_l.transpose(1, 2) # [N, 2l+1, C]
        so3_features.append(f_l)
        idx += dim
    return torch.cat(so3_features, dim=1) # [N, (L+1)^2, C]

def so3_to_flat(so3_features, channels, lmax):
    """Converts [N, (L+1)^2, C] back to [N, C * (L+1)^2]"""
    N = so3_features.shape[0]
    flat_features = []
    idx = 0
    for l in range(lmax + 1):
        length = 2 * l + 1
        f_l = so3_features[:, idx : idx + length, :]
        f_l = f_l.transpose(1, 2) # [N, C, 2l+1]
        f_l = f_l.reshape(N, channels * length)
        flat_features.append(f_l)
        idx += length
    return torch.cat(flat_features, dim=-1) # [N, C * (L+1)^2]
