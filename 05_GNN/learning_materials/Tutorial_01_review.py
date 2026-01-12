'''
Officical website: https://antoniolonga.github.io/Pytorch_geometric_tutorials/index.html

github: https://github.com/AntonioLonga/PytorchGeometricTutorial/tree/main?tab=readme-ov-file

Tutorial_01:
https://www.youtube.com/watch?v=JtDgmmQ60x8&list=PLGMXrbDNfqTzqxB1IGgimuhtfAhGd8lHF

####################################################################################################

LECTURE OVERVIEW: INTRODUCTION TO GRAPH NEURAL NETWORKS (GNNs)

1. THE CORE CONCEPT
The lecture focuses on how to perform deep learning on graphs.
Unlike images (fixed grids) or text (sequences), graphs have an arbitrary structure and no fixed ordering of nodes. 
The goal of a GNN is to learn a numerical vector (an embedding) for every node that captures both its own features 
and its position/role within the graph structure.

2. THE COMPUTATION GRAPH
For any specific node (let's call it "Node A"), the GNN creates a computation graph based on its local neighborhood. 
- Layer 0: The initial input features of Node A and its neighbors.
- Layer 1: Node A aggregates information from its direct neighbors.
- Layer 2: Node A's neighbors aggregate information from their own neighbors (which are 2 steps away from Node A).
By stacking K layers, a node can "see" information from K steps away.

3. THE BASIC MATH OF MESSAGE PASSING
The process of updating a node's representation involves two main steps: Aggregation and Transformation.

The mathematical formula for a node "v" at layer "k" is:

h_v^(k) = SIGMA( W_k * AGGREGATE({ h_u^(k-1) for all u in Neighbors(v) }) + B_k * h_v^(k-1) )

Definitions:
- h_v^(k): The state (embedding) of node v at layer k.
- h_u^(k-1): The state of a neighbor node u from the previous layer.
- Neighbors(v): The set of nodes directly connected to v.
- W_k: A trainable weight matrix applied to neighbor information.
- B_k: A trainable weight matrix applied to the node's own previous state.
- SIGMA: A non-linear activation function (usually ReLU).
- AGGREGATE: A function like Sum, Mean, or Max that handles a variable number of neighbors.

4. GRAPH CONVOLUTIONAL NETWORKS (GCN) SPECIFIC MATH
In a standard GCN, the aggregation is often a simple normalized mean:

h_v^(k) = SIGMA( W_k * SUM[ h_u^(k-1) / SQRT(Degree(v) * Degree(u)) ] )

Here, Degree(v) is the number of edges connected to node v. 
This normalization ensures that nodes with many neighbors don't produce massive values that destabilize the neural network.

5. MATRIX FORM (FOR EFFICIENCY)
To compute this for all nodes at once, we use matrices:

H^(k) = SIGMA( INV(D) * A * H^(k-1) * W_k^T )

Definitions:
- H^(k): A matrix where each row is a node's embedding at layer k.
- A: The Adjacency Matrix (1 if nodes are connected, 0 otherwise).
- INV(D): The inverse of the Degree Matrix (used for averaging).
- W_k^T: The transposed weight matrix for the layer.

6. KEY TAKEAWAYS
- Local Neighborhoods: Nodes are defined by their neighbors.
- Weight Sharing: The same weights (W and B) are used for every node in the graph, making the model scale to very large graphs.
- Permutation Invariance: Because the AGGREGATE function (like Sum) doesn't care about the order of neighbors, 
  the GNN produces the same result regardless of how nodes are indexed in the data.
  
####################################################################################################

Then watch these two videos too:
+ Understanding GNN (1): https://www.youtube.com/watch?v=fOctJB4kVlM&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z
+ Understanding GNN (2): https://www.youtube.com/watch?v=ABCGCf8cJOE&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=2

####################################################################################################

MANUAL GNN CALCULATION EXAMPLE
================================================================================

GRAPH SETUP:
------------
We have a simple graph with 4 nodes: A, B, C, D

Edges (connections):
- A connects to B and C
- B connects to A and D
- C connects to A
- D connects to B

Visual representation:
    C
    |
    A --- B --- D

ADJACENCY MATRIX (A):
     A  B  C  D
A [  0  1  1  0 ]
B [  1  0  0  1 ]
C [  1  0  0  0 ]
D [  0  1  0  0 ]

DEGREE MATRIX (D):
(Degree = number of neighbors)
     A  B  C  D
A [  2  0  0  0 ]
B [  0  2  0  0 ]
C [  0  0  1  0 ]
D [  0  0  0  1 ]

================================================================================
INITIAL NODE FEATURES (Layer 0):
--------------------------------------------------------------------------------
Let's say each node starts with a 2-dimensional feature vector:

h_A^(0) = [1.0, 0.5]
h_B^(0) = [0.8, 1.2]
h_C^(0) = [1.5, 0.3]
h_D^(0) = [0.4, 0.9]

As a matrix H^(0):
     Feature1  Feature2
A [    1.0      0.5    ]
B [    0.8      1.2    ]
C [    1.5      0.3    ]
D [    0.4      0.9    ]

================================================================================
LAYER 1 COMPUTATION:
--------------------------------------------------------------------------------
Formula: h_v^(1) = SIGMA( W_1 * AGGREGATE({ h_u^(0) / sqrt(deg(v)*deg(u)) }) )

Let's use a simple weight matrix W_1 (2x2):
W_1 = [ 0.5  0.3 ]
      [ 0.2  0.4 ]

STEP-BY-STEP FOR NODE A:
-------------------------
Neighbors of A: {B, C}
Degree of A: 2

1. Normalized neighbor features:
   
   From B: h_B^(0) / sqrt(deg(A) * deg(B)) 
         = [0.8, 1.2] / sqrt(2 * 2)
         = [0.8, 1.2] / 2
         = [0.4, 0.6]
   
   From C: h_C^(0) / sqrt(deg(A) * deg(C))
         = [1.5, 0.3] / sqrt(2 * 1)
         = [1.5, 0.3] / 1.414
         = [1.061, 0.212]

2. Aggregate (Sum):
   aggregated = [0.4, 0.6] + [1.061, 0.212]
              = [1.461, 0.812]

3. Apply weight matrix W_1:
   W_1 * aggregated = [ 0.5  0.3 ] * [ 1.461 ]
                      [ 0.2  0.4 ]   [ 0.812 ]
   
   First element:  0.5 * 1.461 + 0.3 * 0.812 = 0.731 + 0.244 = 0.975
   Second element: 0.2 * 1.461 + 0.4 * 0.812 = 0.292 + 0.325 = 0.617
   
   Result = [0.975, 0.617]

4. Apply activation (ReLU - keeps positive values):
   h_A^(1) = ReLU([0.975, 0.617]) = [0.975, 0.617]

STEP-BY-STEP FOR NODE B:
-------------------------
Neighbors of B: {A, D}
Degree of B: 2

1. Normalized neighbor features:
   
   From A: [1.0, 0.5] / sqrt(2 * 2) = [0.5, 0.25]
   From D: [0.4, 0.9] / sqrt(2 * 1) = [0.283, 0.636]

2. Aggregate: [0.5, 0.25] + [0.283, 0.636] = [0.783, 0.886]

3. Apply W_1:
   First:  0.5 * 0.783 + 0.3 * 0.886 = 0.392 + 0.266 = 0.658
   Second: 0.2 * 0.783 + 0.4 * 0.886 = 0.157 + 0.354 = 0.511
   
   Result = [0.658, 0.511]

4. h_B^(1) = [0.658, 0.511]

STEP-BY-STEP FOR NODE C:
-------------------------
Neighbors of C: {A}
Degree of C: 1

1. From A: [1.0, 0.5] / sqrt(1 * 2) = [0.707, 0.354]

2. Aggregate: [0.707, 0.354]

3. Apply W_1:
   First:  0.5 * 0.707 + 0.3 * 0.354 = 0.354 + 0.106 = 0.460
   Second: 0.2 * 0.707 + 0.4 * 0.354 = 0.141 + 0.142 = 0.283

4. h_C^(1) = [0.460, 0.283]

STEP-BY-STEP FOR NODE D:
-------------------------
Neighbors of D: {B}

1. From B: [0.8, 1.2] / sqrt(1 * 2) = [0.566, 0.849]

2. Apply W_1:
   First:  0.5 * 0.566 + 0.3 * 0.849 = 0.283 + 0.255 = 0.538
   Second: 0.2 * 0.566 + 0.4 * 0.849 = 0.113 + 0.340 = 0.453

3. h_D^(1) = [0.538, 0.453]

================================================================================
FINAL EMBEDDINGS AFTER LAYER 1:
--------------------------------------------------------------------------------
H^(1):
     Feature1  Feature2
A [   0.975     0.617   ]
B [   0.658     0.511   ]
C [   0.460     0.283   ]
D [   0.538     0.453   ]

================================================================================
KEY OBSERVATIONS:
--------------------------------------------------------------------------------
1. Node A has the highest values because it aggregated from 2 neighbors (B, C)
2. Each node's new embedding incorporates information from its neighbors
3. The normalization prevents nodes with many neighbors from dominating
4. If we added Layer 2, Node A would now aggregate from B and C, which have
   already seen information from D and A respectively - this is how information
   propagates through the graph!
5. The same weight matrix W_1 was used for all nodes (weight sharing)

LAYER 2 COMPUTATION (BRIEF VERSION)
================================================================================

STARTING FROM LAYER 1 EMBEDDINGS:
--------------------------------------------------------------------------------
H^(1):
     Feature1  Feature2
A [   0.975     0.617   ]
B [   0.658     0.511   ]
C [   0.460     0.283   ]
D [   0.538     0.453   ]

NEW WEIGHT MATRIX FOR LAYER 2:
W_2 = [ 0.6  0.4 ]
      [ 0.3  0.5 ]

================================================================================
LAYER 2 CALCULATIONS:
--------------------------------------------------------------------------------

NODE A (neighbors: B, C):
-------------------------
Normalized aggregation from B: [0.658, 0.511] / 2 = [0.329, 0.256]
Normalized aggregation from C: [0.460, 0.283] / 1.414 = [0.325, 0.200]
Sum: [0.654, 0.456]

Apply W_2: [0.6*0.654 + 0.4*0.456, 0.3*0.654 + 0.5*0.456]
         = [0.575, 0.424]

h_A^(2) = [0.575, 0.424]

NODE B (neighbors: A, D):
-------------------------
From A: [0.975, 0.617] / 2 = [0.488, 0.309]
From D: [0.538, 0.453] / 1.414 = [0.380, 0.320]
Sum: [0.868, 0.629]

Apply W_2: [0.6*0.868 + 0.4*0.629, 0.3*0.868 + 0.5*0.629]
         = [0.772, 0.575]

h_B^(2) = [0.772, 0.575]

NODE C (neighbors: A):
-------------------------
From A: [0.975, 0.617] / 1.414 = [0.689, 0.436]

Apply W_2: [0.6*0.689 + 0.4*0.436, 0.3*0.689 + 0.5*0.436]
         = [0.588, 0.425]

h_C^(2) = [0.588, 0.425]

NODE D (neighbors: B):
-------------------------
From B: [0.658, 0.511] / 1.414 = [0.465, 0.361]

Apply W_2: [0.6*0.465 + 0.4*0.361, 0.3*0.465 + 0.5*0.361]
         = [0.423, 0.320]

h_D^(2) = [0.423, 0.320]

================================================================================
FINAL EMBEDDINGS AFTER LAYER 2:
--------------------------------------------------------------------------------
H^(2):
     Feature1  Feature2
A [   0.575     0.424   ]
B [   0.772     0.575   ]
C [   0.588     0.425   ]
D [   0.423     0.320   ]

================================================================================
IMPORTANT INSIGHTS:
--------------------------------------------------------------------------------
1. Node D now indirectly contains information from Node A!
   - Layer 1: D only saw B
   - Layer 2: D sees B, but B had already aggregated from A
   - So information from A reached D in 2 hops

2. Node A now knows about Node D:
   - Layer 1: A saw B and C
   - Layer 2: A sees B (which had seen D) and C
   
3. With 2 layers, nodes can "communicate" up to 2 steps away in the graph

4. Receptive field:
   - 1 layer = 1-hop neighbors
   - 2 layers = 2-hop neighbors
   - K layers = K-hop neighbors
'''