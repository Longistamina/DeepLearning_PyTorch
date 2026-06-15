'''
THE TRANSFORMER ARCHITECTURE

The Transformer is a neural network architecture introduced in "Attention Is All You Need" (2017) that revolutionized NLP 
by replacing recurrence with attention mechanisms.

================================================================================
CORE IDEA
================================================================================

Traditional RNNs/LSTMs process sequences step-by-step sequentially. 
The Transformer processes the ENTIRE sequence in parallel using self-attention to capture relationships between all positions simultaneously.

Key Innovation: "Attention is all you need" - no recurrence, no convolution, just attention mechanisms.

================================================================================
HIGH-LEVEL ARCHITECTURE
================================================================================

ENCODER-DECODER STRUCTURE:

Input Sequence → ENCODER → Context Representation → DECODER → Output Sequence

Example (Translation):
"I love cats" → [Encoder] → Rich Representation → [Decoder] → "J'aime les chats"


ENCODER STACK (Left Side):
- Takes input sequence
- Produces rich contextual representations
- N identical layers (typically N=6)
- Each position can "see" all other positions

DECODER STACK (Right Side):
- Takes encoder output + previous predictions
- Generates output sequence one token at a time
- N identical layers (typically N=6)
- Uses masked attention to prevent looking ahead


================================================================================
DETAILED ENCODER ARCHITECTURE
================================================================================

Each encoder layer contains TWO sub-layers:

1. MULTI-HEAD SELF-ATTENTION
   - Lets each position attend to all positions in the input
   - Uses multiple attention heads to capture different relationships
   - Output: Contextually aware representation

2. POSITION-WISE FEED-FORWARD NETWORK
   - Applies same dense network to each position independently
   - Two linear transformations with ReLU activation
   - FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

Each sub-layer has:
- Residual connection: output = SubLayer(x) + x
- Layer normalization: LayerNorm(x + SubLayer(x))


FULL ENCODER LAYER:

Input Embeddings (with positional encoding)
    ↓
Multi-Head Self-Attention
    ↓
Add & Normalize (residual connection)
    ↓
Feed-Forward Network
    ↓
Add & Normalize (residual connection)
    ↓
Output to next layer

This repeats N times through the stack.

================================================================================
DETAILED DECODER ARCHITECTURE
================================================================================

Each decoder layer contains THREE sub-layers:

1. MASKED MULTI-HEAD SELF-ATTENTION
   - Attends to previously generated tokens only
   - Masking prevents positions from attending to future positions
   - Ensures predictions depend only on known outputs

2. ENCODER-DECODER ATTENTION (Cross-Attention)
   - Queries come from decoder
   - Keys and Values come from encoder output
   - Allows decoder to focus on relevant parts of input

3. POSITION-WISE FEED-FORWARD NETWORK
   - Same as encoder


FULL DECODER LAYER:

Output Embeddings (with positional encoding)
    ↓
Masked Multi-Head Self-Attention
    ↓
Add & Normalize
    ↓
Encoder-Decoder Attention (Cross-Attention)
    ↓
Add & Normalize
    ↓
Feed-Forward Network
    ↓
Add & Normalize
    ↓
Output to next layer

This repeats N times through the stack.


================================================================================
KEY COMPONENTS IN DETAIL
================================================================================

1. POSITIONAL ENCODING

Problem: Self-attention has no inherent notion of position/order
Solution: Add positional information to embeddings

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Why this formula?
- Allows model to learn relative positions
- Can extrapolate to longer sequences than seen in training
- Unique encoding for each position


2. MULTI-HEAD ATTENTION (detailed in previous explanation)

Allows model to jointly attend to information from different representation subspaces:
- Head 1 might focus on syntactic relationships
- Head 2 might focus on semantic relationships
- Head 3 might focus on long-range dependencies
- etc.


3. RESIDUAL CONNECTIONS

output = LayerNorm(x + SubLayer(x))

Benefits:
- Helps gradients flow during backpropagation
- Allows model to learn identity function if needed
- Stabilizes training of deep networks


4. LAYER NORMALIZATION

Normalizes activations across features for each example:
- Speeds up training
- Reduces sensitivity to parameter initialization
- Applied AFTER each sub-layer


5. FEED-FORWARD NETWORKS

FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

- Applied to each position separately and identically
- Same across all positions but different across layers
- Typically: d_ff = 4 x d_model (e.g., 2048 vs 512)
- Adds non-linearity and transformation capacity


================================================================================
COMPLETE INFORMATION FLOW
================================================================================

TRAINING EXAMPLE: English → French

INPUT: "I love cats"
TARGET: "J'aime les chats"


STEP 1: ENCODING

"I love cats"
    ↓
Token Embedding: [e₁, e₂, e₃]
    ↓
+ Positional Encoding: [e₁+PE₁, e₂+PE₂, e₃+PE₃]
    ↓
Encoder Layer 1:
    - Self-Attention: Each word attends to all words
      ("love" can see "I" and "cats")
    - Feed-Forward transformation
    ↓
Encoder Layer 2:
    - More refined self-attention
    - More transformation
    ↓
... (repeat for N layers)
    ↓
Final Encoder Output: Rich contextual representation [c₁, c₂, c₃]


STEP 2: DECODING (Auto-regressive)

Start with: <SOS> (start of sequence token)

Iteration 1:
    <SOS> → Decoder → Predicts "J'aime"
    
Iteration 2:
    <SOS> J'aime → Decoder → Predicts "les"
    
Iteration 3:
    <SOS> J'aime les → Decoder → Predicts "chats"
    
Iteration 4:
    <SOS> J'aime les chats → Decoder → Predicts <EOS>

In each iteration:
- Decoder uses masked self-attention on its previous outputs
- Decoder uses cross-attention to encoder outputs
- Generates next token probability distribution


================================================================================
ATTENTION MECHANISMS IN TRANSFORMER
================================================================================

Three types of attention:

1. ENCODER SELF-ATTENTION
   - Q, K, V all from encoder input
   - Each position attends to all positions
   - No masking (can see full sequence)

2. DECODER MASKED SELF-ATTENTION
   - Q, K, V all from decoder input
   - Each position attends only to earlier positions
   - Future positions masked with -infinity before softmax

3. ENCODER-DECODER ATTENTION (Cross-Attention)
   - Q from decoder
   - K, V from encoder output
   - Decoder positions attend to all encoder positions
   - This is where translation actually happens!


================================================================================
WHY TRANSFORMER WORKS SO WELL
================================================================================

1. PARALLELIZATION
   - Processes entire sequence at once
   - Much faster training than RNNs
   - Can leverage GPU parallelism effectively

2. LONG-RANGE DEPENDENCIES
   - Direct connections between all positions
   - No gradient vanishing over long distances
   - Can capture dependencies regardless of distance

3. FLEXIBLE ATTENTION
   - Model learns what to attend to
   - Different heads capture different relationships
   - Adaptive to different tasks

4. SCALABILITY
   - Architecture scales well with more data
   - More layers = more capacity
   - Larger models generally perform better


================================================================================
COMPUTATIONAL COMPLEXITY
================================================================================

Self-Attention: O(n² x d)
- n = sequence length
- d = dimension
- Each position attends to all n positions

Feed-Forward: O(n x d²)
- Applied to each of n positions
- Two matrix multiplications

For short sequences: Self-attention is efficient
For long sequences: Quadratic complexity can be problematic
(Led to variants like Transformer-XL, Longformer, etc.)


================================================================================
HYPERPARAMETERS (Original Paper)
================================================================================

Base Model:
- Layers: N = 6
- Model dimension: d_model = 512
- Feed-forward dimension: d_ff = 2048
- Attention heads: h = 8
- Head dimension: d_k = d_v = 64 (512/8)
- Dropout: 0.1

Big Model:
- Layers: N = 6
- Model dimension: d_model = 1024
- Feed-forward dimension: d_ff = 4096
- Attention heads: h = 16


================================================================================
TRAINING TRICKS
================================================================================

1. WARM-UP LEARNING RATE
   - Start with small learning rate
   - Linearly increase for first steps
   - Then decay proportional to inverse square root of step number

2. LABEL SMOOTHING
   - Instead of hard targets (0 or 1)
   - Use soft targets (0.1 or 0.9)
   - Prevents overconfidence

3. RESIDUAL DROPOUT
   - Applied to output of each sub-layer
   - Applied to embeddings before they're summed

4. ATTENTION DROPOUT
   - Applied to attention weights after softmax


================================================================================
ADVANTAGES OVER PREVIOUS ARCHITECTURES
================================================================================

VS. RNNs/LSTMs:
✓ Parallel processing (RNNs are sequential)
✓ No vanishing gradients over long distances
✓ Direct access to any position in sequence
✓ Much faster training

VS. CNNs:
✓ Global receptive field from layer 1
✓ Better at capturing long-range dependencies
✓ More interpretable (can visualize attention)


================================================================================
REAL-WORLD IMPACT
================================================================================

The Transformer architecture enabled:

- BERT (2018): Pre-training breakthrough
- GPT series (2018-present): Large language models
- T5 (2019): Text-to-text framework
- Vision Transformer (2020): Applied to computer vision
- DALL-E (2021): Text-to-image generation
- And countless other models...

It's the foundation of modern NLP and increasingly dominant in other domains too.
'''