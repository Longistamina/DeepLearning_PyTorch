'''
MULTI-HEAD SELF-ATTENTION MECHANISM

The multi-head self-attention mechanism is a core component of the Transformer architecture. Here's how it works:

BASIC SELF-ATTENTION (Single Head)

Self-attention allows each position in a sequence to attend to all positions and compute a weighted representation. For each element in the input:

1. Create three vectors from each input embedding:
   - Query (Q): "What am I looking for?"
   - Key (K): "What do I contain?"
   - Value (V): "What information do I carry?"

2. Compute attention scores by comparing queries with all keys:
   - Score = Q · K^T / sqrt(d_k)
   - This tells us how much each position should focus on every other position

3. Apply softmax to get attention weights (probabilities that sum to 1)

4. Multiply attention weights by values to get the output:
   - Output = softmax(Q·K^T / sqrt(d_k)) · V

###################################################################################################

WHY MULTI-HEAD?

Instead of performing attention once, we do it multiple times in parallel with different learned projections. This allows the model to:

- Attend to different aspects of the input simultaneously
- Capture different types of relationships (syntactic, semantic, positional, etc.)
- Learn richer representations by combining multiple attention patterns


HOW MULTI-HEAD WORKS

1. Split the embedding dimension into h "heads"
   - If embedding dim = 512 and h = 8, each head works with 64 dimensions

2. For each head, create separate Q, K, V projections:
   - Q_i = X · W^Q_i
   - K_i = X · W^K_i  
   - V_i = X · W^V_i
   
3. Compute attention independently for each head:
   - head_i = Attention(Q_i, K_i, V_i)

4. Concatenate all heads:
   - MultiHead = Concat(head_1, head_2, ..., head_h)

5. Apply final linear projection:
   - Output = MultiHead · W^O

KEY BENEFITS

- Parallel computation: All heads computed simultaneously
- Different perspectives: Each head learns different patterns
- Richer representations: Combined heads capture complex relationships
- No sequential bottleneck: Unlike RNNs, can process entire sequence at once

#######################################################################################

MULTI-HEAD SELF-ATTENTION WITH SPECIFIC NUMBERS

Setup:
- Input sequence: 3 tokens
- Embedding dimension: 4
- Number of heads: 2
- Dimension per head: 4/2 = 2

INPUT EMBEDDINGS (3 tokens x 4 dimensions)

X = [[1.0, 0.5, 0.2, 0.8],   # Token 1
     [0.3, 0.9, 0.1, 0.4],   # Token 2
     [0.7, 0.2, 0.6, 0.5]]   # Token 3


WEIGHT MATRICES FOR HEAD 1 (4 x 2 each)

W^Q_1 = [[0.1, 0.2],
         [0.3, 0.1],
         [0.2, 0.4],
         [0.1, 0.3]]

W^K_1 = [[0.2, 0.1],
         [0.1, 0.3],
         [0.4, 0.2],
         [0.2, 0.1]]

W^V_1 = [[0.3, 0.2],
         [0.2, 0.4],
         [0.1, 0.2],
         [0.3, 0.1]]


STEP 1: COMPUTE Q, K, V FOR HEAD 1

Q_1 = X · W^Q_1 (3 x 2)
    = [[1.0x0.1 + 0.5x0.3 + 0.2x0.2 + 0.8x0.1,  1.0x0.2 + 0.5x0.1 + 0.2x0.4 + 0.8x0.3],
       [0.3x0.1 + 0.9x0.3 + 0.1x0.2 + 0.4x0.1,  0.3x0.2 + 0.9x0.1 + 0.1x0.4 + 0.4x0.3],
       [0.7x0.1 + 0.2x0.3 + 0.6x0.2 + 0.5x0.1,  0.7x0.2 + 0.2x0.1 + 0.6x0.4 + 0.5x0.3]]
    
    = [[0.37, 0.57],
       [0.34, 0.25],
       [0.24, 0.55]]

K_1 = X · W^K_1 (3 x 2)
    = [[0.43, 0.42],
       [0.29, 0.41],
       [0.52, 0.33]]

V_1 = X · W^V_1 (3 x 2)
    = [[0.67, 0.62],
       [0.40, 0.48],
       [0.49, 0.41]]


STEP 2: COMPUTE ATTENTION SCORES (Q · K^T)

Scores_1 = Q_1 · K_1^T / sqrt(2)   # sqrt(d_k) = sqrt(2) ≈ 1.41

Q_1 · K_1^T = [[0.37x0.43 + 0.57x0.42,  0.37x0.29 + 0.57x0.41,  0.37x0.52 + 0.57x0.33],
               [0.34x0.43 + 0.25x0.42,  0.34x0.29 + 0.25x0.41,  0.34x0.52 + 0.25x0.33],
               [0.24x0.43 + 0.55x0.42,  0.24x0.29 + 0.55x0.41,  0.24x0.52 + 0.55x0.33]]

            = [[0.40,  0.34,  0.38],
               [0.25,  0.20,  0.26],
               [0.33,  0.29,  0.31]]

Scores_1 / 1.41 = [[0.28,  0.24,  0.27],
                   [0.18,  0.14,  0.18],
                   [0.23,  0.21,  0.22]]


STEP 3: APPLY SOFTMAX (row-wise)

Attention_weights_1 = softmax(Scores_1)

For row 1: exp(0.28)=1.32, exp(0.24)=1.27, exp(0.27)=1.31
Sum = 3.90

Attention_weights_1 = [[0.34,  0.33,  0.34],   # Token 1 attends to all tokens
                       [0.33,  0.33,  0.33],   # Token 2 (nearly uniform)
                       [0.33,  0.33,  0.33]]   # Token 3 (nearly uniform)


STEP 4: COMPUTE OUTPUT FOR HEAD 1

head_1 = Attention_weights_1 · V_1

       = [[0.34x0.67 + 0.33x0.40 + 0.34x0.49,  0.34x0.62 + 0.33x0.48 + 0.34x0.41],
          [0.33x0.67 + 0.33x0.40 + 0.33x0.49,  0.33x0.62 + 0.33x0.48 + 0.33x0.41],
          [0.33x0.67 + 0.33x0.40 + 0.33x0.49,  0.33x0.62 + 0.33x0.48 + 0.33x0.41]]

       = [[0.52, 0.50],
          [0.52, 0.50],
          [0.52, 0.50]]


SIMILARLY FOR HEAD 2 (with different weights):

head_2 = [[0.45, 0.38],
          [0.41, 0.42],
          [0.48, 0.35]]


STEP 5: CONCATENATE HEADS

MultiHead_output = Concat(head_1, head_2)  (3 x 4)

                 = [[0.52, 0.50, 0.45, 0.38],   # Token 1
                    [0.52, 0.50, 0.41, 0.42],   # Token 2
                    [0.52, 0.50, 0.48, 0.35]]   # Token 3


STEP 6: FINAL LINEAR PROJECTION

W^O = [[0.5, 0.3, 0.2, 0.4],
       [0.2, 0.4, 0.1, 0.3],
       [0.3, 0.2, 0.5, 0.1],
       [0.4, 0.1, 0.3, 0.2]]

Final_output = MultiHead_output · W^O  (3 x 4)

             = [[0.86, 0.51, 0.53, 0.56],
                [0.84, 0.54, 0.52, 0.58],
                [0.85, 0.49, 0.56, 0.53]]

This is the final output of the multi-head attention layer!
'''