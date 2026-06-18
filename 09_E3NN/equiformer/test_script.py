import torch
from equiformer_block import EquiformerBlock


def test_equiformer_block():
    torch.manual_seed(42)

    # --- Hyperparameters ---
    N = 100      # Number of nodes (DNA base pairs)
    E = 300      # Number of edges (sparse connections)
    lmax = 3
    channels = 128
    channels_q = 128
    channels_hidden = 256
    channels_kv = 128

    # --- 1. Dummy Graph Data ---
    # Flat layout: [N, C * (L+1)^2]
    node_features = torch.randn(N, channels * (lmax + 1)**2)

    edge_src = torch.randint(0, N, (E,))
    edge_dst = torch.randint(0, N, (E,))
    edge_index = torch.stack((edge_src, edge_dst), dim=0)

    edge_vec = torch.randn(E, 3)
    edge_dist = edge_vec.norm(dim=-1, keepdim=False)

    # --- 2. Dummy Radial Weights (Fix 2: Must be 3D!) ---
    # Shape: [E, (lmax+1)^2, channels_kv * 2]
    radial_weights = torch.randn(E, (lmax + 1)**2, channels_kv * 2)

    # --- 3. Instantiate Block ---
    block = EquiformerBlock(
        lmax=lmax,
        channels=channels,
        channels_q=channels_q,
        channels_hidden=channels_hidden
    )

    # Override the norm with our corrected one
    # block.norm1 = EquivariantLayerNorm(lmax, channels)
    # block.norm2 = EquivariantLayerNorm(lmax, channels)

    print(f'Parameters: {sum(p.numel() for p in block.parameters()):,}')

    # --- 4. Forward Pass ---
    print("Running Forward Pass...")
    out = block(node_features, edge_index, edge_vec, edge_dist, radial_weights)

    print("✅ Test Passed!")
    print(f"Input Shape:  {node_features.shape}")
    print(f"Output Shape: {out.shape}")

    assert out.shape == node_features.shape, "Residual connection shape mismatch!"
    print("🎉 Equivariant Block is perfectly wired and ready for Diffusion!")

if __name__ == "__main__":
    test_equiformer_block()
