import torch
import torch.nn as nn
from e3nn_wigner import wigner_D

def get_alignment_angles(edge_vec: torch.Tensor, edge_dist: torch.Tensor):
    """
    Computes the Z-Y-Z Euler angles required to rotate an arbitrary 3D edge
    vector so that it points straight up the Z-axis (0, 0, r).

    Args:
        edge_vec: [E, 3] tensor of (x, y, z) edge vectors
    Returns:
        alpha, beta, gamma: [E] tensors of Euler angles
    """
    r = edge_dist
    r = torch.clamp(r, min=1e-8) # Prevent division by zero

    x, y, z = edge_vec[:, 0], edge_vec[:, 1], edge_vec[:, 2]

    # Spherical coordinates
    theta = torch.acos(z / r)  # Polar angle [0, pi]
    phi = torch.atan2(y, x)    # Azimuthal angle [-pi, pi]

    # To rotate the vector TO the Z-axis, we apply the inverse of the
    # rotation that takes the Z-axis to the vector.
    # Z -> vec is R_z(phi) @ R_y(theta).
    # vec -> Z is R_y(-theta) @ R_z(-phi).
    # In Z-Y-Z Euler angles, this corresponds to:
    alpha = torch.zeros_like(phi)
    beta = -theta
    gamma = -phi

    return alpha, beta, gamma

def rotate_to_local_frame(features_by_l, alpha, beta, gamma, lmax):
    """
    Rotates node features into the local edge frame (Z-axis aligned).
    features_by_l: list of tensors, where features_by_l[l] has shape [E, C, 2l+1]
    """
    rotated_features = []
    for l in range(lmax + 1):
        # Get Wigner-D matrix for this degree: [E, 2l+1, 2l+1]
        D = wigner_D(l, alpha, beta, gamma)

        f_l = features_by_l[l] # Shape: [E, C, 2l+1]

        # Apply rotation via einsum:
        # (E, channels, m_in) x (E, m_out, m_in) -> (E, channels, m_out)
        rotated_f_l = torch.einsum('ecj,eij->eci', f_l, D)
        rotated_features.append(rotated_f_l)

    return rotated_features

def rotate_to_global_frame(features_by_l, alpha, beta, gamma, lmax):
    """
    Rotates features back to the global 3D frame.
    The inverse rotation uses angles (-gamma, -beta, -alpha).
    """
    rotated_features = []
    inv_alpha = -gamma
    inv_beta = -beta
    inv_gamma = -alpha

    for l in range(lmax + 1):
        D_inv = wigner_D(l, inv_alpha, inv_beta, inv_gamma)
        f_l = features_by_l[l]
        rotated_f_l = torch.einsum('ecj,eij->eci', f_l, D_inv)
        rotated_features.append(rotated_f_l)

    return rotated_features

####################################################

class SO2_Mixer(nn.Module):
    def __init__(self, lmax, channels_in, channels_out):
        super().__init__()
        self.lmax = lmax

        # One simple Linear layer per degree l.
        # No CG coefficients needed!
        self.linears = nn.ModuleList([
            nn.Linear(channels_in, channels_out, bias=False)
            for _ in range(lmax + 1)
        ])

    def forward(self, local_features_by_l, radial_weights):
        """
        local_features_by_l: list of [E, C_in, 2l+1]
        radial_weights: [E, total_m_components, C_out] from RadialFunction
        """
        out_features = []
        weight_idx = 0

        for l in range(self.lmax + 1):
            f_l = local_features_by_l[l] # [E, C_in, 2l+1]

            # 1. Mix the channels (C_in -> C_out)
            # Transpose to [E, 2l+1, C_in] to apply Linear on the last dim
            f_l_T = f_l.transpose(1, 2)
            mixed = self.linears[l](f_l_T) # [E, 2l+1, C_out]
            mixed = mixed.transpose(1, 2)  # Back to [E, C_out, 2l+1]

            # 2. Apply the Radial Weights (Channel-wise scaling for m-components)
            num_m = 2 * l + 1
            # Extract the weights for this specific l and its m-components
            w = radial_weights[:, weight_idx : weight_idx + num_m, :] # [E, 2l+1, C_out]
            weight_idx += num_m

            # Transpose w to match mixed: [E, C_out, 2l+1]
            w_T = w.transpose(1, 2)

            # Element-wise multiplication (The collapsed Tensor Product!)
            scaled = mixed * w_T
            out_features.append(scaled)

        return out_features
