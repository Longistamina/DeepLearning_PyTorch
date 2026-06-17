This is the perfect mindset. Stripping away the "general-purpose" bloat and the PyG dependencies to build a lean, pure-PyTorch EquiformerV3 specifically for **DNA Diffusion Noise Prediction** is exactly how you master the architecture. 

Since your task is highly specific (Hamiltonian path, `rels` diffusion, predicting `1o` noise), we can aggressively simplify the EquiformerV3 codebase. 

Here is your modular blueprint, the exact folder structure, and the step-by-step execution order to build this from the ground up.

---

### 📂 The Folder Structure (8 Scripts)

Create a clean directory and separate your code into these 8 distinct modules. This prevents the "spaghetti code" trap that plagues most equivariant repositories.

1. **`graph_utils.py`**: Handles the BigBird sparse masks, `rels` $\rightarrow$ `pos` reconstruction, and `edge_src`/`edge_dst` generation.
2. **`embeddings.py`**: Handles the 1D Sequence Embeddings (Route/Fractional), Diffusion Timestep Embeddings, and Relative Positional Encodings (LogBin).
3. **`radial_basis.py`**: Contains `GaussianSmearing` (or Bessel) and the `RadialFunction` (the MLP that generates channel-wise scaling weights).
4. **`so2_math.py`**: The raw math engine. Contains Wigner-D matrix calculations, Z-axis alignment logic, and SO(2) phase shifts (sines/cosines).
5. **`equivariant_norms.py`**: Contains `EquiformerLayerNorm` (normalizes vector magnitudes) and `RMSNorm` (for Q/K stabilization).
6. **`so2_attention.py`**: The core engine. Wires `so2_math`, `radial_basis`, and `equivariant_norms` together to generate Q, K, V, compute invariant attention, and aggregate messages.
7. **`equiformer_block.py`**: The Equivariant Feed-Forward Network (Gated MLP) that processes the aggregated messages before the next layer.
8. **`model_diffusion.py`**: The main `nn.Module` that stacks the blocks, adds the skip connections, and outputs the final `1o` noise prediction.

---

### 🗺️ The Execution Order (What to write first, then what!!)

Do not write the model first. You must build the mathematical primitives and test them on dummy tensors before wiring them together. 

#### Phase 1: The Inputs & The Fuel (Day 1)
*Goal: Get your sparse graph and radial weights working without any equivariant math.*
1. **Write `graph_utils.py`**: Build your BigBird mask and `rels2coords` function. Test it by generating dummy `rels`, converting to `pos`, and verifying your `edge_src`/`edge_dst` arrays.
2. **Write `embeddings.py`**: Build your Time and Sequence embeddings. These are just pure scalars (`0e`).
3. **Write `radial_basis.py`**: Implement `GaussianSmearing` and the `RadialFunction`. 
   * *Test:* Pass dummy edge distances through it and verify the output shape matches the `expand_index` logic (e.g., outputting exactly one weight per $m$-component).

#### Phase 2: The Stabilizers (Day 2)
*Goal: Ensure your network won't explode when you stack 10 layers.*
1. **Write `equivariant_norms.py`**: 
   * Implement `EquiformerLayerNorm`. It must extract the magnitudes of the `1o` and `2e` channels, pass them through a standard `nn.LayerNorm`, and rescale the original vectors.
   * Implement `RMSNorm` for the scalar channels.
   * *Test:* Create a dummy tensor with massive vector values, pass it through the norm, and verify the directions remain unchanged while the magnitudes normalize to ~1.

#### Phase 3: The Math Engine (Day 3)
*Goal: Master the SO(2) reduction. This is the hardest part of EquiformerV2/V3.*
1. **Write `so2_math.py`**: 
   * Write a function that takes an `edge_vec` `[E, 3]` and computes the polar/azimuthal angles ($\theta, \phi$).
   * Write the Wigner-D rotation logic that "aligns" a feature tensor to the Z-axis based on those angles.
   * Write the SO(2) phase-shift logic (multiplying the $m$-channels by $e^{im\phi}$).
   * *Test:* Take a dummy vector, rotate it to the Z-axis, apply a weight, and rotate it back. Verify the math is perfectly equivariant.

#### Phase 4: The Core Block (Day 4-5)
*Goal: Build the layer that replaces the `e3nn` Tensor Product.*
1. **Write `so2_attention.py`**: This is your `SO2EquivariantGraphAttention` class.
   * **Input:** `node_features`, `edge_src`, `edge_dst`, `edge_vec`, `radial_weights`.
   * **Step A:** Use `so2_math` to rotate source node features into the local edge frame (Z-axis aligned).
   * **Step B:** Multiply by the `radial_weights` (channel-wise scaling).
   * **Step C:** Rotate back to the global frame to get **Keys** and **Values**.
   * **Step D:** Generate **Queries** from destination nodes. Apply `RMSNorm` to Q and K.
   * **Step E:** Compute the invariant dot-product, `scatter_softmax`, and `scatter_add` to aggregate.
2. **Write `equivariant_ffn.py`**: Build the Gated MLP. It takes the aggregated features, uses scalars to gate the vectors, and applies SiLU to the scalars.

#### Phase 5: The Diffusion Model (Day 6)
*Goal: Wire it all together for the final task.*
1. **Write `model_diffusion.py`**:
   * **Input Head:** Map the raw `rels` (or `pos`) to initial node features.
   * **The Stack:** Loop through $N$ layers of `[EquiformerLayerNorm -> SO2Attention -> EquivariantFFN]`.
   * **Output Head:** A simple `o3.Linear` (or custom SO(2) linear layer) that projects the final node features strictly to `1x1o` (the predicted noise vector).
2. **Write the Training Loop**:
   * Sample clean `rels`.
   * Add Gaussian noise based on timestep $t$ (The DeNS objective!).
   * Pass through `model_diffusion`.
   * Compute `F.mse_loss(predicted_noise, true_noise)`.

---

### 🧬 DNA-Specific Tweaks to Keep in Mind

Since you are focusing purely on DNA diffusion, you can make these specific optimizations in your scripts:

1. **In `radial_basis.py` (Bond-Type Conditioning):** 
   EquiformerV3 uses atom-types to condition the radial weights. For DNA, you should condition on **Sequence Distance**. Pass your `LogBinEmbedding` (Relative Positional Encoding) directly into the `RadialFunction` alongside the 3D distance. This tells the MLP: *"A 5Å stretch is normal if they are 2 steps apart in the sequence, but catastrophic if they are 50 steps apart."*
2. **In `so2_attention.py` (No Self-Attention):** 
   Because DNA is a polymer, a node doesn't need to attend to itself to learn its local geometry. You can safely enforce `remove_selfloop` in your sparse mask, saving compute.
3. **In `model_diffusion.py` (The Output Head):**
   Your target is the noise added to `rels`. Since `rels` are translation-invariant vectors, your output head **must** be strictly equivariant (`1o`). Do not use a standard `nn.Linear` at the very end; use an equivariant projection so the predicted noise perfectly rotates with the input structure.

### Summary of the Paradigm Shift
By following this order, you are completely abandoning the `e3nn.FullyConnectedTensorProduct`. 
* **Old Way:** Node Features $\times$ Spherical Harmonics $\rightarrow$ Clebsch-Gordan Tables $\rightarrow$ Mixed Features.
* **Your New Way:** Node Features $\xrightarrow{\text{Wigner-D}}$ Z-Aligned Features $\xrightarrow{\text{Radial Weights}}$ Scaled Features $\xrightarrow{\text{Inverse Wigner-D}}$ Global Keys/Values.

Start with **Phase 1** today. Get your sparse edges and radial weights printing the correct shapes. Once the "plumbing" is verified, the heavy SO(2) math will be much easier to debug!
