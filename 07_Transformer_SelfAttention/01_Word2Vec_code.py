import torch
import torch.nn as nn
import torch.optim as optim
import re

# 1. DATA PREPARATION
# -------------------
english = [
    "Better late than never",
    "Actions speak louder than words",
    "When the cat's away, the mice will play",
    "Don't count your chickens before they hatch",
    "The early bird catches the worm"
]

french = [
    "Mieux vaut tard que jamais",
    "Les actes valent mieux que les paroles",
    "Quand le chat n'est pas là, les souris dansent",
    "Il ne faut pas vendre la peau de l'ours avant de l'avoir tué",
    "L'avenir appartient à ceux qui se lèvent tôt"
]

# Combine corpus
corpus = english + french

for _, proverb in enumerate(corpus):
    print(proverb)
'''
Better late than never
Actions speak louder than words
When the cat's away, the mice will play
Don't count your chickens before they hatch
The early bird catches the worm
Mieux vaut tard que jamais
Les actes valent mieux que les paroles
Quand le chat n'est pas là, les souris dansent
Il ne faut pas vendre la peau de l'ours avant de l'avoir tué
L'avenir appartient à ceux qui se lèvent tôt
'''

# Simple preprocessing: lowercase and remove punctuation
def clean_text(text):
    # Keep only alphanumeric characters and normalize case
    return re.findall(r'\b\w+\b', text.lower())

tokenized_corpus = [clean_text(sentence) for sentence in corpus]

for _, sentence in enumerate(tokenized_corpus):
    print(sentence)
# ['better', 'late', 'than', 'never']
# ['actions', 'speak', 'louder', 'than', 'words']
# ['when', 'the', 'cat', 's', 'away', 'the', 'mice', 'will', 'play']
# ['don', 't', 'count', 'your', 'chickens', 'before', 'they', 'hatch']
# ['the', 'early', 'bird', 'catches', 'the', 'worm']
# ['mieux', 'vaut', 'tard', 'que', 'jamais']
# ['les', 'actes', 'valent', 'mieux', 'que', 'les', 'paroles']
# ['quand', 'le', 'chat', 'n', 'est', 'pas', 'là', 'les', 'souris', 'dansent']
# ['il', 'ne', 'faut', 'pas', 'vendre', 'la', 'peau', 'de', 'l', 'ours', 'avant', 'de', 'l', 'avoir', 'tué']
# ['l', 'avenir', 'appartient', 'à', 'ceux', 'qui', 'se', 'lèvent', 'tôt']

# Build Vocabulary
vocabulary = set(word for sentence in tokenized_corpus for word in sentence) # {'à', 'speak', 'l', 'chat', 'tôt', 'your', 's', ...}
word2idx = {word: i for i, word in enumerate(vocabulary)} # {'word0': 0, 'word1': 1, ...}
idx2word = {i: word for word, i in word2idx.items()} # {0: 'word0', 1: 'word1'}

vocab_size = len(vocabulary)
print(vocab_size) # 66

# 2. CREATE TRAINING DATA (Skip-gram)
# -----------------------------------
# Pairs of (Center Word, Context Word)
data = []
window_size = 2  # Look 2 words to the left and 2 to the right

for sentence in tokenized_corpus:
    for idx, word in enumerate(sentence):
        # Define window boundaries
        start = max(idx - window_size, 0)
        end = min(idx + window_size + 1, len(sentence))
        
        # Add pairs
        for context_word in sentence[start:end]:
            if context_word != word: # Don't pair word with itself
                data.append((word2idx[word], word2idx[context_word]))
'''
Illustrate one iteration:

# sentence = ["the", "early", "bird", "catches", "the", "worm"]
# word = "bird"
# idx = 2  (0-based index of "bird")
# window_size = 2

start = max(idx - window_size, 0)
# start = max(2 - 2, 0) -> 0
# The window starts at index 0 ("the")

end = min(idx + window_size + 1, len(sentence))
# end = min(2 + 2 + 1, 6) -> 5
# The window ends at index 5 (Python slices are exclusive, so it stops before index 5)

sentence[start:end] = sentence[0:5] = ["the", "early", "bird", "catches", "the"]

Generate pairs:
# Iteration 1:
  + context_word = "the"
  + Is "the" == "bird"? -> No.
  + Result: Append pair ("bird", "the")
  
# Iteration 2:
  + context_word = "early"
  + Is "early" == "bird"? No.
  + Result: Append pair ("bird", "early")
... (and so on) ...

sentence[start:end] = sentence[0:5] = ["the", "early", "bird", "catches", "the"]
=> This basically means pair the word "bird" with all other words within ± window_size (except itself)
=> [('bird', 'the'), ('bird', 'early'), ('bird', 'catches'), ('bird', 'the')]
'''

#######################
'''
NOTE: here, if we choose window_size=1
=> "late" and "never" will not be matched as training pair
=> their similarity score later on will be very slow
'''
#######################

print(f"Total training pairs: {len(data)}")
print(f"Example pair: {idx2word[data[0][0]]} -> {idx2word[data[0][1]]}")
# Total training pairs: 252
# Example pair: better -> late

# 3. MODEL ARCHITECTURE
# ---------------------
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        # HIDDEN LAYER (The Embeddings)
        # Note: In Pytorch, nn.Embedding is optimized 'One-hot * Linear_Layer'BaseExceptionGroup
        # This replaces the massive matrix multiplication of One-Hot vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # OUTPUT LAYER
        # Projects from hidden dimension back to vocabulary size
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, inputs):
        # Inputs (Index number of the word) -> Hidden Layer (Vector)
        embeds = self.embeddings(inputs)
        
        # Hidden Layer -> Output Layer (Raw Scores)
        out = self.linear(embeds)
        
        # Note: We don't apply Softmax here because CrossEntropyLoss in training step will does this automatically
        return out

#########################
## Initialize Word2Vec ##
#########################

# Hyperparameters
EMBEDDING_DIM = 10 # Small dimension for small dataset
LEARNING_RATE = 0.001
EPOCHS = 10000

model = Word2Vec(vocab_size, EMBEDDING_DIM)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f'{sum(p.numel() for p in model.parameters()):,}')
# 1,386

# 4. TRAINING LOOP
# ----------------
from tldm import tldm

for epoch in tldm(range(EPOCHS), desc='Training'):
    total_loss = 0
    
    # Convert data to tensors
    # In a real scenario, you would use a DataLoader for batches
    inputs = torch.tensor([pair[0] for pair in data]) # Indices of Center words
    targets = torch.tensor([pair[1] for pair in data]) # Indices of Context words
    
    # Forward pass
    log_probs = model(inputs)
    
    # Compute Loss
    loss = loss_fn(log_probs, targets)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass (Update weights)
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    
    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
'''
Training:  16%|█▌        | 1555/10000 [00:00<00:03, 2858.07it/s]
Epoch 1000, Loss: 1.5955
Training:  24%|██▍       | 2435/10000 [00:00<00:02, 2895.02it/s]
Epoch 2000, Loss: 1.4184
Training:  33%|███▎      | 3323/10000 [00:01<00:02, 2931.69it/s]
Epoch 3000, Loss: 1.4003
Training:  45%|████▌     | 4547/10000 [00:01<00:01, 3040.42it/s]
Epoch 4000, Loss: 1.3954
Training:  55%|█████▍    | 5469/10000 [00:01<00:01, 3014.43it/s]
Epoch 5000, Loss: 1.3935
Training:  64%|██████▍   | 6382/10000 [00:02<00:01, 2740.34it/s]
Epoch 6000, Loss: 1.3927
Training:  76%|███████▌  | 7606/10000 [00:02<00:00, 2972.53it/s]
Epoch 7000, Loss: 1.3923
Training:  85%|████████▌ | 8504/10000 [00:02<00:00, 2942.82it/s]
Epoch 8000, Loss: 1.3921
Training:  94%|█████████▍| 9412/10000 [00:03<00:00, 2978.56it/s]
Epoch 9000, Loss: 1.3920
Training: 100%|██████████| 10000/10000 [00:03<00:00, 301.49it/s]
Epoch 10000, Loss: 1.3919
'''

# 5. EXTRACTING EMBEDDINGS
# ------------------------
# We only care about the hidden layer weights
word_embeddings = model.embeddings.weight.data

# Function to calculate Cosine Similarity
def get_similarity(word1, word2):
    if (word1 not in word2idx) or (word2 not in word2idx):
        return "Word not in vocab"
    
    # Get vectors
    vec1 = word_embeddings[word2idx[word1]]
    vec2 = word_embeddings[word2idx[word2]]
    
    # Calculate cosine similarity manually
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    
    similarity = dot_product / (norm1 * norm2)
    return similarity.item()

print("\n--- RESULTS ---")
print("Similarity between 'late' and 'never':")
print(f"{get_similarity('late', 'never'):.4f}")

print("Similarity between 'cat' and 'mice':")
print(f"{get_similarity('cat', 'mice'):.4f}")

print("Similarity between 'late' and 'bird' (Unrelated):")
print(f"{get_similarity('late', 'bird'):.4f}")
'''
Similarity between 'late' and 'never':
0.3748

Similarity between 'cat' and 'mice':
0.4199

Similarity between 'late' and 'bird' (Unrelated):
0.1300
'''