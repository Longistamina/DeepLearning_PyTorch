import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

# Tokenization
def tokenize(sentences):
    return [sentence.lower().split() for sentence in sentences]

english_tokens = tokenize(english)
french_tokens = tokenize(french)

# Build vocabulary
def build_vocab(tokenized_sentences):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    idx = 4
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    
    # Create reverse mapping
    idx_to_token = {v: k for k, v in vocab.items()}
    return vocab, idx_to_token

src_vocab, src_idx2token = build_vocab(english_tokens)
tgt_vocab, tgt_idx2token = build_vocab(french_tokens)

print(f"Source vocabulary size: {len(src_vocab)}")
print(f"Target vocabulary size: {len(tgt_vocab)}")

# Convert tokens to indices
def tokens_to_indices(tokenized_sentences, vocab):
    indices = []
    for sentence in tokenized_sentences:
        sent_indices = [vocab['<SOS>']] + [vocab.get(token, vocab['<UNK>']) for token in sentence] + [vocab['<EOS>']]
        indices.append(sent_indices)
    return indices

src_indices = tokens_to_indices(english_tokens, src_vocab)
tgt_indices = tokens_to_indices(french_tokens, tgt_vocab)

# Create dataset
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.src_data[idx]), torch.LongTensor(self.tgt_data[idx])

# Collate function for padding
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    # Pad source sequences
    src_lengths = [len(s) for s in src_batch]
    max_src_len = max(src_lengths)
    padded_src = torch.zeros(len(src_batch), max_src_len, dtype=torch.long)
    for i, s in enumerate(src_batch):
        padded_src[i, :len(s)] = s
    
    # Pad target sequences
    tgt_lengths = [len(t) for t in tgt_batch]
    max_tgt_len = max(tgt_lengths)
    padded_tgt = torch.zeros(len(tgt_batch), max_tgt_len, dtype=torch.long)
    for i, t in enumerate(tgt_batch):
        padded_tgt[i, :len(t)] = t
    
    return padded_src, padded_tgt


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key), batch_size)
        V = self.split_heads(self.W_v(value), batch_size)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        return self.W_o(output)


# ============================================================================
# FEED-FORWARD NETWORK
# ============================================================================

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ============================================================================
# ENCODER LAYER
# ============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


# ============================================================================
# DECODER LAYER
# ============================================================================

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross-attention with encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4, 
                 num_encoder_layers=2, num_decoder_layers=2, d_ff=256, 
                 max_len=100, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_src_mask(self, src):
        # Create mask for padding tokens (batch_size, 1, 1, src_len)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def create_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        
        # Padding mask
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
        
        # Look-ahead mask (prevent attending to future tokens)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
        
        # Combine masks
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return tgt_mask
    
    def encode(self, src, src_mask):
        # Embedding + positional encoding
        x = self.src_embedding(src) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        # Embedding + positional encoding
        x = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        # Create masks
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        
        # Encode
        enc_output = self.encode(src, src_mask)
        
        # Decode
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.fc_out(dec_output)
        
        return output


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, num_epochs, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Teacher forcing: use actual target as input
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token (<SOS>)
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Calculate loss
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return model


# ============================================================================
# TRANSLATION (INFERENCE)
# ============================================================================

def translate(model, src_sentence, src_vocab, tgt_vocab, tgt_idx2token, max_len=50, device='cpu'):
    model.eval()
    model = model.to(device)
    
    # Tokenize and convert to indices
    src_tokens = src_sentence.lower().split()
    src_indices = [src_vocab['<SOS>']] + [src_vocab.get(token, src_vocab['<UNK>']) for token in src_tokens] + [src_vocab['<EOS>']]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    # Encode source
    src_mask = model.create_src_mask(src_tensor)
    enc_output = model.encode(src_tensor, src_mask)
    
    # Initialize decoder input with <SOS>
    tgt_indices = [tgt_vocab['<SOS>']]
    
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = model.create_tgt_mask(tgt_tensor)
        
        # Decode
        dec_output = model.decode(tgt_tensor, enc_output, src_mask, tgt_mask)
        
        # Get prediction for last position
        output = model.fc_out(dec_output[:, -1, :])
        
        # Get most likely next token
        next_token = output.argmax(dim=-1).item()
        
        tgt_indices.append(next_token)
        
        # Stop if <EOS> is predicted
        if next_token == tgt_vocab['<EOS>']:
            break
    
    # Convert indices to tokens
    translated_tokens = [tgt_idx2token[idx] for idx in tgt_indices[1:-1]]  # Skip <SOS> and <EOS>
    
    return ' '.join(translated_tokens)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Create dataset and dataloader
dataset = TranslationDataset(src_indices, tgt_indices)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=128,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=256,
    max_len=100,
    dropout=0.1
)

print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train the model
print("\n" + "="*70)
print("TRAINING")
print("="*70)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = train_model(model, train_loader, num_epochs=200, learning_rate=0.0003)

# Test translation
print("\n" + "="*70)
print("TRANSLATION RESULTS")
print("="*70)

for i, eng_sent in enumerate(english):
    translation = translate(model, eng_sent, src_vocab, tgt_vocab, tgt_idx2token, device=device)
    print(f"\nInput (English):  {eng_sent}")
    print(f"Predicted:        {translation}")
    print(f"Expected (French): {french[i]}")
    print("-" * 70)

# Try some variations
print("\n" + "="*70)
print("TESTING WITH VARIATIONS")
print("="*70)

test_sentences = [
    "Better late than never",
    "The early bird catches the worm",
    "Actions speak louder than words"
]

for sent in test_sentences:
    translation = translate(model, sent, src_vocab, tgt_vocab, tgt_idx2token, device=device)
    print(f"\nInput:  {sent}")
    print(f"Output: {translation}")
    
'''
======================================================================
MODEL ARCHITECTURE
======================================================================
Total parameters: 1,007,912

======================================================================
TRAINING
======================================================================
Using device: cuda
Epoch [20/200], Loss: 1.1993
Epoch [40/200], Loss: 0.2348
Epoch [60/200], Loss: 0.0291
Epoch [80/200], Loss: 0.0102
Epoch [100/200], Loss: 0.0100
Epoch [120/200], Loss: 0.0025
Epoch [140/200], Loss: 0.0014
Epoch [160/200], Loss: 0.0170
Epoch [180/200], Loss: 0.0013
Epoch [200/200], Loss: 0.0008

======================================================================
TRANSLATION RESULTS
======================================================================

Input (English):  Better late than never
Predicted:        mieux vaut tard que jamais
Expected (French): Mieux vaut tard que jamais
----------------------------------------------------------------------

Input (English):  Actions speak louder than words
Predicted:        les actes valent mieux que les paroles
Expected (French): Les actes valent mieux que les paroles
----------------------------------------------------------------------

Input (English):  When the cat's away, the mice will play
Predicted:        quand le chat n'est pas là, les souris dansent
Expected (French): Quand le chat n'est pas là, les souris dansent
----------------------------------------------------------------------

Input (English):  Don't count your chickens before they hatch
Predicted:        il ne faut pas vendre la peau de l'ours avant de l'avoir tué
Expected (French): Il ne faut pas vendre la peau de l'ours avant de l'avoir tué
----------------------------------------------------------------------

Input (English):  The early bird catches the worm
Predicted:        l'avenir appartient à ceux qui se lèvent tôt
Expected (French): L'avenir appartient à ceux qui se lèvent tôt
----------------------------------------------------------------------

======================================================================
TESTING WITH VARIATIONS
======================================================================

Input:  Better late than never
Output: mieux vaut tard que jamais

Input:  The early bird catches the worm
Output: l'avenir appartient à ceux qui se lèvent tôt

Input:  Actions speak louder than words
Output: les actes valent mieux que les paroles
'''