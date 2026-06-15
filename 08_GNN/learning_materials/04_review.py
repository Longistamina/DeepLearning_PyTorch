'''
Officical website: https://antoniolonga.github.io/Pytorch_geometric_tutorials/index.html

github: https://github.com/AntonioLonga/PytorchGeometricTutorial/tree/main?tab=readme-ov-file

Watch these videos:
+ https://www.youtube.com/watch?v=A-yKQamf2Fc&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=6
+ https://www.youtube.com/watch?v=CwsPoa7z2c8

#######################################

GRAPH ATTENTION NETWORKS (GAT) - LECTURE AND MATHEMATICS SUMMARY

This explanation covers the concepts presented in the video regarding Graph Attention Networks (GAT), 
based on the landmark paper by Velickovic et al. (2017).

1. CORE CONCEPT
---------------
The fundamental idea of GAT is to allow nodes in a graph to attend to their neighbors' features with different "importance" or weights. 

Unlike Graph Convolutional Networks (GCNs), which use fixed weights based on the graph's structure (like node degrees), 
GAT learns these weights dynamically during training based on the node features themselves.

2. THE MATHEMATICAL WORKFLOW
----------------------------
The GAT layer transforms a set of input node features into a new set of output features through the following steps:

# STEP A: LINEAR TRANSFORMATION
Every node feature vector h_i (of dimension F) is multiplied by a shared weight matrix W (of dimension F' x F).
   Equation: h_transformed_i = W * h_i
                               (F'xF) * F -> F'

# STEP B: CALCULATION OF ATTENTION COEFFICIENTS
To find out how important neighbor "j" is to node "i", the model computes an alignment score (e_ij). 
This is done using a shared attention mechanism "a", which is typically a single-layer feed-forward neural network.
   Equation: e_ij = a(W * h_i, W * h_j)

In practice, this is implemented by concatenating the transformed features of node i and j, 
followed by a dot product with a learnable vector "vector_a" and a LeakyReLU non-linearity.
   Equation: e_ij = LeakyReLU( vector_a^T * [ W*h_i || W*h_j ] )
   (where || denotes concatenation)

# STEP C: SOFTMAX NORMALIZATION
To make the coefficients comparable across different neighborhoods and ensure they sum to 1, 
a softmax function is applied over all neighbors "j" of node "i".
   Equation: alpha_ij = exp(e_ij) / Sum_{k in N_i}( exp(e_ik) )
   (where N_i is the set of neighbors of node i, including itself)

# STEP D: AGGREGATION
The final output feature for node i (h'_i) is a weighted sum of its neighbors' transformed features, 
followed by a non-linear activation function (like ELU or ReLU).
   Equation: h'_i = Sigma( alpha_ij * W * h_j )

3. MULTI-HEAD ATTENTION
To stabilize the learning process and capture different aspects of the neighborhood 
(e.g., one head focused on local structure, another on feature similarity), GAT uses "multi-head attention." 
- Multiple independent attention mechanisms (heads) are calculated.
- For hidden layers: The outputs of the K heads are concatenated.
- For the final (output) layer: The outputs are averaged before the final activation.

4. KEY ADVANTAGES
- Computational Efficiency: The attention mechanism is parallelizable across all edges.
- Inductive Capability: Because the weights are based on features rather than fixed node indices, the model can be applied to completely unseen graphs.
- Flexibility: It does not require the graph to be undirected or the structure to be known upfront.

5. IMPLEMENTATION NOTE (PYTORCH GEOMETRIC)
In the lecture, the GATConv layer from PyTorch Geometric is used. It abstracts the math into a simple class where you define:
- in_channels (input feature size)
- out_channels (output feature size)
- heads (number of attention heads)
- concat (boolean, whether to concatenate or average head outputs)
'''