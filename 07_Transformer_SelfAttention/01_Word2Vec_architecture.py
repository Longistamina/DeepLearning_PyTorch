'''
WORD2VEC ARCHITECTURE EXPLAINED

Based on the StatQuest breakdown, 
Word2Vec is a shallow neural network. 
Its primary goal is not to perform a classification task, but rather to learn the weights of the hidden layer, 
which become the "Word Embeddings."

Here is the step-by-step architecture:

1. THE INPUT (One-Hot Encoding)
   - The process begins with a corpus of text (the training data).
   - A vocabulary is created from every unique word in that text.
   - The input to the network is a "One-Hot Encoded" vector.
   - If your vocabulary has 10,000 words, the input vector has 10,000 components.
   - To represent a specific word, the component corresponding to that word is set to "1," and all other components are set to "0."

2. THE HIDDEN LAYER (The Embedding Layer)
   - The input vector is fed into a single hidden layer.
   - Crucially, this is a Linear Layer—it does not use an activation function (like ReLU or Sigmoid) to curve the data. 
     It just sums the weighted inputs.
   - The number of neurons in this layer determines the "dimensionality" of the resulting word embeddings.
     - In the video example, he might use 2 neurons to visualize it on a 2D graph.
     - In real-world applications (like Google's dataset), this is often 300 neurons.
   - The weights connecting the Input Layer to this Hidden Layer are the "Lookup Table." 
   Since the input is a One-Hot vector, the Hidden Layer effectively just "looks up" the row of weights corresponding to the active input word.

3. THE OUTPUT LAYER (Softmax)
   - The Hidden Layer connects to the Output Layer.
   - The Output Layer has the same number of neurons as the Input Layer (e.g., 10,000 neurons for a 10,000-word vocabulary).
   - This layer uses the "Softmax" activation function.
   - Softmax ensures the output is a probability distribution (all values sum to 1).
   - The output represents the probability of every other word in the vocabulary appearing near the input word (context).

4. TRAINING PROCESS (Skip-Gram Version)
   - The network is trained using pairs of words found in the text (Target Word, Context Word).
   - Example sentence: "The quick brown fox."
   - Training pair: (quick, brown).
   - Input: "quick" (One-Hot).
   - The network predicts probabilities.
   - The error is calculated by comparing the predicted probabilities against the actual target "brown" (One-Hot).
   - Backpropagation adjusts the weights to minimize this error.

5. THE RESULT
   - Once training is complete, the Output Layer is discarded.
   - We only care about the weights matrix in the Hidden Layer.
   - These weights are the coordinates (vectors) for the words.
   - Words that appear in similar contexts (like "cool" and "cold") will have similar vector values, 
   placing them close together in the vector space.

#########################################################################################
#########################################################################################

nn.EMBEDDING WORKFLOW

1. INITIALIZATION (Creation)
----------------------------
When you define `self.embeddings = nn.Embedding(vocab_size, embedding_dim)`:

   - PyTorch creates a trainable Weight Matrix (Tensor).
   - Shape: [vocab_size x embedding_dim]
   - Initial values: Random numbers (usually from a normal distribution).
   - Gradients: Enabled (requires_grad=True), meaning these numbers will change 
     during training.

   Example (vocab_size=6, embedding_dim=3):
   
   Index |  Dimension 0 | Dimension 1 | Dimension 2
   ------------------------------------------------
     0   |     0.54     |    -1.22    |     0.33    (Vector for word 0)
     1   |     0.88     |     0.01    |    -0.56    (Vector for word 1)
     2   |    -0.12     |     1.45    |     0.99    (Vector for word 2)
     3   |     ...      |     ...     |     ...
     4   |     ...      |     ...     |     ...
     5   |     ...      |     ...     |     ...


2. THE INPUT (Forward Pass)
---------------------------
The layer does NOT accept One-Hot vectors. It accepts **Integers** (Indices).

   Input Batch: [2, 0] 
   (Meaning: "I want the vector for Word ID 2 and Word ID 0")


3. THE MECHANISM (The Lookup)
-----------------------------
Instead of performing matrix multiplication (which is computationally expensive), 
PyTorch performs a direct **array slice** or **lookup**.

   - It goes to Row 2 and copies the values.
   - It goes to Row 0 and copies the values.

   *Mathematical Equivalence:*
   Multiplying a [1 x 6] One-Hot vector (with a 1 at index 2) by the 
   [6 x 3] matrix results in exactly the values of Row 2. 
   `nn.Embedding` just skips the multiplication math and grabs the row directly.


4. THE OUTPUT
-------------
The result is a tensor containing the dense vectors for the requested indices.

   Output Shape: [Batch_Size x Embedding_Dim] -> [2 x 3]

   Result:
   [
     [-0.12,  1.45,  0.99],  <-- Vector for Index 2
     [ 0.54, -1.22,  0.33]   <-- Vector for Index 0
   ]


5. BACKPROPAGATION (Updating Weights)
-------------------------------------
During `loss.backward()`:
   - PyTorch calculates gradients.
   - Crucially, it creates a "Sparse Gradient."
   - Since we only looked up Row 2 and Row 0, **ONLY Row 2 and Row 0 are updated.**
   - Rows 1, 3, 4, and 5 remain untouched for this specific batch. 
     (This makes training very efficient for massive vocabularies).

#########################################################################################
#########################################################################################

WORD2VEC WITH SPECIFIC NUMBERS

EXAMPLE WORD VECTORS (simplified to 4 dimensions)
------------------------------------------------
king   = [0.5,  0.8,  0.1,  0.2]
queen  = [0.4,  0.7,  0.9,  0.3]
man    = [0.3,  0.6,  0.1,  0.1]
woman  = [0.2,  0.5,  0.9,  0.2]
apple  = [0.8,  0.1,  0.3,  0.7]
orange = [0.7,  0.2,  0.4,  0.6]


CALCULATING SIMILARITY (Cosine Similarity)
-----------------------------------------
Formula: similarity = (A · B) / (||A|| x ||B||)

Example: How similar are "king" and "queen"?

king · queen = (0.5x0.4) + (0.8x0.7) + (0.1x0.9) + (0.2x0.3)
             = 0.20 + 0.56 + 0.09 + 0.06
             = 0.91

||king||  = √(0.5² + 0.8² + 0.1² + 0.2²) = √0.94 = 0.97
||queen|| = √(0.4² + 0.7² + 0.9² + 0.3²) = √1.15 = 1.07

similarity(king, queen) = 0.91 / (0.97 x 1.07) = 0.88


COMPARING SIMILARITIES
---------------------
similarity(king, queen)  = 0.88  ← High! Similar context
similarity(king, man)    = 0.94  ← Very high! Related concepts
similarity(king, apple)  = 0.36  ← Low! Different contexts


THE FAMOUS ANALOGY: king - man + woman ≈ queen
---------------------------------------------
Step by step calculation:

1. king - man:
   [0.5, 0.8, 0.1, 0.2] - [0.3, 0.6, 0.1, 0.1] = [0.2, 0.2, 0.0, 0.1]

2. (king - man) + woman:
   [0.2, 0.2, 0.0, 0.1] + [0.2, 0.5, 0.9, 0.2] = [0.4, 0.7, 0.9, 0.3]

3. Result: [0.4, 0.7, 0.9, 0.3]
   Compare to queen: [0.4, 0.7, 0.9, 0.3]
   
   Perfect match! This captures: "king is to man as queen is to woman"


TRAINING EXAMPLE (Skip-gram)
---------------------------
Sentence: "The cat sat on the mat"
Window size: 2

Training pair: target="sat", context="cat"

Input:  sat  → [0, 0, 1, 0, 0, 0] (one-hot, 6-word vocabulary)
                     ↓
            Hidden layer (embedding)
                [0.3, 0.7, 0.2, 0.5] ← This becomes "sat" vector
                     ↓
            Output prediction
        [0.05, 0.82, 0.03, 0.06, 0.02, 0.02]
                 ↑
         Predicts "cat" (position 1) with 0.82 probability

The network adjusts weights so that words appearing together 
get similar embeddings.
'''

