import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# DATA PREPARATION
# ============================================================================

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

# Simple tokenization (split by space and lowercase)
def tokenize(sentences):
    tokens = []
    for sentence in sentences:
        tokens.append(sentence.lower().split())
    return tokens

english_tokens = tokenize(english)
french_tokens = tokenize(french)

print("English tokens:", english_tokens[0]) # ['better', 'late', 'than', 'never']
print("French tokens:", french_tokens[0]) # ['mieux', 'vaut', 'tard', 'que', 'jamais']

# Build vocabularies
def build_vocab(tokenized_sentences):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    idx = 3
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

english_vocab = build_vocab(english_tokens)
french_vocab = build_vocab(french_tokens)

print(f"\nEnglish vocab size: {len(english_vocab)}") # 29
print(f"French vocab size: {len(french_vocab)}") # 39

# Convert tokens to indices
def tokens_to_indices(tokenized_sentences, vocab):
    indices = []
    for sentence in tokenized_sentences:
        sent_indices = [vocab['<SOS>']] + [vocab[token] for token in sentence] + [vocab['<EOS>']]
        indices.append(sent_indices)
    return indices

english_indices = tokens_to_indices(english_tokens, english_vocab)
french_indices = tokens_to_indices(french_tokens, french_vocab)

print("\nExample English indices:", english_indices[0]) # [1, 3, 4, 5, 6, 2]
print("Example French indices:", french_indices[0]) # [1, 3, 4, 5, 6, 7, 2]

# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute division term for sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
    
# ============================================================================
# MULTI-HEAD SELF-ATTENTION
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, d_k)
        # x shape: (batch_size, seq_len, d_model)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # Transpose to: (batch_size, num_heads, seq_len, d_k)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided (for padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # attention_output shape: (batch_size, num_heads, seq_len, d_k)
        
        # Concatenate heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights
    
# ============================================================================
# TRANSFORMER ENCODER LAYER
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attn_weights

# ============================================================================
# COMPLETE MODEL
# ============================================================================

class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(TranslationModel, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask=None):
        # Embedding
        x = self.embedding(src) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        all_attention_weights = []
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, mask)
            all_attention_weights.append(attn_weights)
        
        return x, all_attention_weights

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# Hyperparameters
d_model = 64        # Embedding dimension
num_heads = 4       # Number of attention heads
num_layers = 2      # Number of encoder layers
d_ff = 128          # Feed-forward dimension
max_len = 50        # Maximum sequence length
dropout = 0.1

# Create model
model = TranslationModel(
    src_vocab_size=len(english_vocab),
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_len=max_len,
    dropout=dropout
)

print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
print(model)
print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))

# Prepare batch
batch_sentences = [english_indices[0], english_indices[1]]
max_batch_len = max(len(s) for s in batch_sentences)

# Pad sequences
padded_batch = []
for sent in batch_sentences:
    padded = sent + [english_vocab['<PAD>']] * (max_batch_len - len(sent))
    padded_batch.append(padded)

# Convert to tensor
input_tensor = torch.LongTensor(padded_batch)
print("\n" + "="*70)
print("INPUT EXAMPLE")
print("="*70)
print("Input shape:", input_tensor.shape)
print("Input tensor:\n", input_tensor)

# Forward pass
with torch.no_grad():
    output, attention_weights = model(input_tensor)

print("\n" + "="*70)
print("OUTPUT")
print("="*70)
print("Output shape:", output.shape)
print("Output (first sentence, first 3 positions, first 8 dimensions):")
print(output[0, :3, :8])

# Visualize attention weights from first layer, first head
print("\n" + "="*70)
print("ATTENTION WEIGHTS (Layer 1, Head 1)")
print("="*70)
attn_first_head = attention_weights[0][0, 0].numpy()  # First batch, first head
print("Shape:", attn_first_head.shape)
print("\nAttention matrix (how much each position attends to others):")
print(np.round(attn_first_head, 3))

# Show what each position attends to
print("\n" + "="*70)
print("ATTENTION INTERPRETATION")
print("="*70)
sentence = english[0].lower().split()
sentence_with_special = ['<SOS>'] + sentence + ['<EOS>']

print("Sentence:", ' '.join(sentence_with_special))
print("\nFor each token, showing which tokens it attends to most:\n")

for i, token in enumerate(sentence_with_special[:len(sentence_with_special)]):
    top_3_indices = np.argsort(attn_first_head[i])[-3:][::-1]
    print(f"{token:15} attends to: ", end="")
    for idx in top_3_indices:
        if idx < len(sentence_with_special):
            print(f"{sentence_with_special[idx]:15} ({attn_first_head[i, idx]:.3f})  ", end="")
    print()

print("\n" + "="*70)
print("POSITIONAL ENCODING VISUALIZATION")
print("="*70)
pe = model.pos_encoding.pe[0, :10, :8].numpy()
print("First 10 positions, first 8 dimensions:")
print(np.round(pe, 3))
