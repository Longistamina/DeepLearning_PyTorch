# e3nn_wigner.py
import math
import torch

def explicit_default_types(dtype=None, device=None):
    """Simplified type resolver replacing e3nn's version"""
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.empty(0).device
    return dtype, device

def su2_generators(j: int) -> torch.Tensor:
    m = torch.arange(-j, j)
    raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    m = torch.arange(-j + 1, j + 1)
    lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    m = torch.arange(-j, j + 1)
    return torch.stack(
        [
            0.5 * (raising + lowering),  # x (usually)
            torch.diag(1j * m),  # z (usually)
            -0.5j * (raising - lowering),  # -y (usually)
        ],
        dim=0,
    )

def change_basis_real_to_complex(l: int, dtype=None, device=None) -> torch.Tensor:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
        q[l + m, l - abs(m)] = 1j * (-1) ** m / 2**0.5
    q = (-1j) ** l * q

    dtype, device = explicit_default_types(dtype, device)
    dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[dtype]
    return q.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)

def so3_generators(l) -> torch.Tensor:
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    X = torch.conj(Q.T) @ X @ Q
    assert torch.all(torch.abs(torch.imag(X)) < 1e-5)
    return torch.real(X)

def wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    r"""Wigner D matrix representation of SO(3).

    Note: In e3nn's convention, these correspond to:
    alpha: Rotation around Y axis, applied third.
    beta: Rotation around X axis, applied second.
    gamma: Rotation around Y axis, applied first.
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = so3_generators(l)
    return torch.matrix_exp(alpha * X[1]) @ torch.matrix_exp(beta * X[0]) @ torch.matrix_exp(gamma * X[1])
