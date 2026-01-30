'''
WORD2VEC EMBEDDING EXPLANATION

Word2vec is a technique for learning word embeddings - dense vector 
representations of words that capture semantic relationships.

CORE IDEA
--------
Words that appear in similar contexts should have similar vector 
representations. For example, "cat" and "dog" should be closer in 
vector space than "cat" and "airplane".

TWO MAIN ARCHITECTURES
---------------------

1. CBOW (Continuous Bag of Words)
   - Predicts a target word from surrounding context words
   - Example: Given ["the", "sat", "on", "mat"] → predict "cat"
   
2. Skip-gram
   - Predicts context words from a target word
   - Example: Given "cat" → predict ["the", "sat", "on", "mat"]
   - Better for smaller datasets and rare words

HOW IT WORKS
-----------
1. Start with random vectors for each word
2. Slide a window through text corpus
3. Train neural network to predict word relationships
4. The learned hidden layer weights become word embeddings

RESULTING PROPERTIES
-------------------
- Each word → fixed-size dense vector (e.g., 300 dimensions)
- Semantic similarity: similar(king, queen) > similar(king, apple)
- Analogies: king - man + woman ≈ queen
- Mathematical operations become meaningful

ADVANTAGES
----------
- Captures semantic meaning
- Reduces dimensionality vs one-hot encoding
- Enables transfer learning
- Fast training with negative sampling

TYPICAL USE
-----------
Pre-trained embeddings (Google News, Wikipedia) used as input 
features for downstream NLP tasks like sentiment analysis, 
translation, or text classification.

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
Formula: similarity = (A · B) / (||A|| × ||B||)

Example: How similar are "king" and "queen"?

king · queen = (0.5×0.4) + (0.8×0.7) + (0.1×0.9) + (0.2×0.3)
             = 0.20 + 0.56 + 0.09 + 0.06
             = 0.91

||king||  = √(0.5² + 0.8² + 0.1² + 0.2²) = √0.94 = 0.97
||queen|| = √(0.4² + 0.7² + 0.9² + 0.3²) = √1.15 = 1.07

similarity(king, queen) = 0.91 / (0.97 × 1.07) = 0.88


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

