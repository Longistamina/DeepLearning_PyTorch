import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# WORD2VEC SKIP-GRAM MODEL IN PYTORCH
# ============================================================

class SkipGramModel(nn.Module):
    """
    Skip-gram model: predict context words from target word
    """
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        
        # Input embedding (target word)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Output embedding (context word)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        self.target_embeddings.weight.data.uniform_(-0.5/embedding_dim, 
                                                      0.5/embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5/embedding_dim, 
                                                       0.5/embedding_dim)
    
    def forward(self, target_word, context_word):
        # Get embeddings
        target_embed = self.target_embeddings(target_word)    # [batch, embed_dim]
        context_embed = self.context_embeddings(context_word)  # [batch, embed_dim]
        
        # Compute dot product (similarity)
        score = torch.sum(target_embed * context_embed, dim=1)
        
        return score

# ============================================================
# DATA PREPARATION
# ============================================================

# Sample corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are animals",
    "the cat and dog are friends"
]

# Tokenize
tokens = []
for sentence in corpus:
    tokens.extend(sentence.lower().split())

print(f"Total tokens: {len(tokens)}")
print(f"Sample tokens: {tokens[:10]}")
'''
Total tokens: 23
Sample tokens: ['the', 'cat', 'sat', 'on', 'the', 'mat', 'the', 'dog', 'sat', 'on']
'''

# Build vocabulary
from collections import Counter
word_counts = Counter(tokens)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
idx_to_word = {idx: word for word, idx in vocab.items()}

vocab_size = len(vocab)
print(f"\nVocabulary size: {vocab_size}")
print(f"Vocabulary: {list(vocab.keys())}")
'''
Vocabulary size: 13
Vocabulary: ['the', 'cat', 'sat', 'on', 'dog', 'and', 'are', 'mat', 'log', 'cats', 'dogs', 'animals', 'friends']
'''

# ============================================================
# CREATE TRAINING PAIRS (Skip-gram)
# ============================================================

def create_skipgram_dataset(tokens, vocab, window_size=2):
    """
    Create (target, context) pairs
    """
    pairs = []
    
    for i, target_word in enumerate(tokens):
        target_idx = vocab[target_word]
        
        # Get context words within window
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:  # Don't pair word with itself
                context_word = tokens[j]
                context_idx = vocab[context_word]
                pairs.append((target_idx, context_idx))
    
    return pairs

window_size = 2
training_pairs = create_skipgram_dataset(tokens, vocab, window_size)

print(f"\nTraining pairs: {len(training_pairs)}")
print("Sample pairs (target -> context):")
for i in range(5):
    target_idx, context_idx = training_pairs[i]
    print(f"  {idx_to_word[target_idx]} -> {idx_to_word[context_idx]}")

'''
Training pairs: 86
Sample pairs (target -> context):
  the -> cat
  the -> sat
  cat -> the
  cat -> sat
  cat -> on
'''

# ============================================================
# TRAINING WITH NEGATIVE SAMPLING
# ============================================================

embedding_dim = 50
learning_rate = 0.025
epochs = 100
num_negative_samples = 5

model = SkipGramModel(vocab_size, embedding_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Word frequency for negative sampling
word_freqs = np.array([word_counts[idx_to_word[i]] for i in range(vocab_size)])
word_freqs = word_freqs ** 0.75  # Smooth distribution
word_freqs = word_freqs / word_freqs.sum()


from tldm import tldm

for epoch in tldm(range(epochs), desc='Training'):
    total_loss = 0
    
    for target_idx, context_idx in training_pairs:
        # Positive pair
        target = torch.LongTensor([target_idx]).to(device)
        context = torch.LongTensor([context_idx]).to(device)
        
        # Positive score (should be high)
        pos_score = model(target, context)
        pos_loss = -torch.log(torch.sigmoid(pos_score))
        
        # Negative sampling
        neg_indices = np.random.choice(vocab_size, 
                                       size=num_negative_samples, 
                                       p=word_freqs)
        neg_indices = torch.LongTensor(neg_indices).to(device)
        
        # Negative scores (should be low)
        neg_scores = model(target.repeat(num_negative_samples), neg_indices)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_scores)))
        # Example:
        # target_idx = 3  (word "cat")
        # num_negative_samples = 5
        # neg_indices = [7, 2, 9, 1, 4]  (random "wrong" words)
        # target.repeat(num_negative_samples)         ->        [3, 3, 3, 3, 3]
        # compare "cat" against each negative sample            [7, 2, 9, 1, 4]
        # model([3, 3, 3, 3, 3], [7, 2, 9, 1, 4]) 
        #  -> neg_scores = [0.8, -0.3, 1.2, 0.1, -0.5]
        #  - High score (0.8, 1.2) = BAD! "cat" is similar to negative sample
        #  - Low score (-0.3, -0.5) = GOOD! "cat" is dissimilar to negative sample
        #
        # neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_scores)))
        # -> Make neg_scores LOW (we want dissimilarity)
        # -neg_scores: [-0.8, 0.3, -1.2, -0.1, 0.5] => Why negate? Because sigmoid(-x) is high when x is low
        # sigmoid(x): [0.31, 0.57, 0.23, 0.48, 0.62]
        # - If neg_score was high (0.8) → sigmoid(-0.8) = 0.31 (low)
        # - If neg_score was low (-0.5) → sigmoid(0.5) = 0.62 (high)
        
        # Total loss
        loss = pos_loss + neg_loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

'''
Training:  23%|███████████████████████████▊                                                                                             | 23/100 [00:01<00:03, 22.64it/s]
Epoch 20/100, Loss: 236.3641
Training:  44%|█████████████████████████████████████████████████████▏                                                                   | 44/100 [00:01<00:02, 25.47it/s]
Epoch 40/100, Loss: 230.5507
Training:  65%|██████████████████████████████████████████████████████████████████████████████▋                                          | 65/100 [00:02<00:01, 25.58it/s]
Epoch 60/100, Loss: 219.7868
Training:  83%|████████████████████████████████████████████████████████████████████████████████████████████████████▍                    | 83/100 [00:03<00:00, 25.63it/s]
Epoch 80/100, Loss: 210.4355
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:04<00:00,  3.00it/s]
Epoch 100/100, Loss: 200.8782
'''

# ============================================================
# EXTRACT TRAINED EMBEDDINGS
# ============================================================

embeddings = model.target_embeddings.weight.data.cpu().numpy()

print(f"\n{'='*60}")
print("TRAINED EMBEDDINGS")
print(f"{'='*60}")
print(f"Shape: {embeddings.shape}")
print(f"\nWord 'cat' embedding (first 10 dims):")
print(embeddings[vocab['cat']][:10])

'''
============================================================
TRAINED EMBEDDINGS
============================================================
Shape: (13, 50)

Word 'cat' embedding (first 10 dims):
[ 0.01814973 -0.5085046   0.14865027  0.1861113  -0.13569748  0.04037447
 -0.07867606  0.00525767 -0.3990992   0.18380988]
'''

# ============================================================
# COMPUTE SIMILARITIES
# ============================================================

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def find_most_similar(word, vocab, embeddings, top_k=5):
    """Find most similar words"""
    if word not in vocab:
        return []
    
    word_idx = vocab[word]
    word_vec = embeddings[word_idx]
    
    similarities = []
    for idx in range(len(embeddings)):
        if idx != word_idx:
            sim = cosine_similarity(word_vec, embeddings[idx])
            similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return [(idx_to_word[idx], sim) for idx, sim in similarities[:top_k]]


print(f"\n{'='*60}")
print("WORD SIMILARITIES")
print(f"{'='*60}")

test_words = ['cat', 'dog', 'sat']
for word in test_words:
    if word in vocab:
        print(f"\nMost similar to '{word}':")
        similar = find_most_similar(word, vocab, embeddings, top_k=3)
        for similar_word, sim in similar:
            print(f"  {similar_word}: {sim:.4f}")
            
'''
============================================================
WORD SIMILARITIES
============================================================

Most similar to 'cat':
  mat: 0.7801
  log: 0.7175
  sat: 0.7101

Most similar to 'dog':
  the: 0.8487
  dogs: 0.6186
  log: 0.5724

Most similar to 'sat':
  mat: 0.9713
  cat: 0.7101
  on: 0.6464
'''

# ============================================================
# WORD ANALOGY: king - man + woman ≈ queen
# ============================================================

def word_analogy(word_a, word_b, word_c, vocab, embeddings, idx_to_word):
    """
    Solve analogy: word_a is to word_b as word_c is to ?
    Example: king - man + woman ≈ queen
    """
    if word_a not in vocab or word_b not in vocab or word_c not in vocab:
        return None
    
    vec_a = embeddings[vocab[word_a]]
    vec_b = embeddings[vocab[word_b]]
    vec_c = embeddings[vocab[word_c]]
    
    # Analogy vector: (a - b) + c
    target_vec = vec_a - vec_b + vec_c
    
    # Find closest word
    best_word = None
    best_sim = -1
    
    for idx in range(len(embeddings)):
        word = idx_to_word[idx]
        if word not in [word_a, word_b, word_c]:
            sim = cosine_similarity(target_vec, embeddings[idx])
            if sim > best_sim:
                best_sim = sim
                best_word = word
    
    return best_word, best_sim


print(f"\n{'='*60}")
print("WORD ANALOGIES")
print(f"{'='*60}")

# Example: cat - mat + log ≈ dog (from our corpus)
if all(w in vocab for w in ['cat', 'mat', 'log']):
    result, sim = word_analogy('cat', 'mat', 'log', vocab, embeddings, idx_to_word)
    print(f"\ncat - mat + log ≈ {result} (similarity: {sim:.4f})")
# cat - mat + log ≈ dogs (similarity: 0.6260)