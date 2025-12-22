'''
TENSOR EXPLAINED SIMPLY

1. MATHEMATICS (The "Multilinear Machine")
Wiki: "An algebraic object describing a multilinear relationship."
- Simple View: A tensor is a "machine" that takes in vectors and outputs a value.
- The Rule: It is "multilinear," meaning it processes each input independently and proportionally.
- Basis Independence: The tensor is a geometric reality that exists on its own. 
  It does not depend on the coordinate system (the "basis") you choose to describe it.

2. PHYSICS (The "Universal Truth")
Wiki: "A concise framework for solving mechanics and general relativity."
- Simple View: Tensors describe physical properties (like stress, gravity, or magnetism) that stay the same even if you rotate your perspective.
- Tensor Fields: In the real world, tensors often change from point to point (like the stress throughout a bridge). 
  This "map" of tensors is called a Tensor Field.
- Why it matters: It allows physicists to write one equation that works regardless of how an observer is moving or tilted.

3. DEEP LEARNING / PYTORCH (The "Component Array")
Wiki: "Components form an array... thought of as a high-dimensional matrix."
- Simple View: PyTorch deals with the *components* of a tensor. 
  When we write down the numbers of a mathematical tensor into a grid, that grid is what we call a "Tensor" in code.
  Works like list, vector, matrix, etc.
- Hierarchy:
    - Rank 0: Scalar (Magnitude only)
    - Rank 1: Vector (Magnitude and Direction)
    - Rank 2: Matrix (Components of a linear map)
    - Rank n: High-dimensional data (e.g., video frames or neural weights)

SUMMARY TABLE:
- Math: A multilinear map between vector spaces.
- Physics: A coordinate-independent physical property (often a Field).
- PyTorch: An optimized array of components for high-speed calculation.
'''