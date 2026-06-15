import torch
import torch.nn as nn
import random

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# --- 1. The Corpus ---
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

# --- 2. Simple Vocabulary Builder ---
# We need to map "cat" -> 45, "chat" -> 32, etc.
class Vocab:
    def __init__(self):
        self.word2index = {"<sos>": 0, "<eos>": 1, "<unk>": 2}
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<unk>"}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split():
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

# Create Vocabs
input_vocab = Vocab()  # For English
output_vocab = Vocab() # For French

for sent in english: input_vocab.add_sentence(sent)
for sent in french: output_vocab.add_sentence(sent)

# --- 3. Convert Sentences to Tensors ---
def sentence_to_tensor(vocab, sentence, device):
    # Convert words to integers
    indexes = [vocab.word2index[word] for word in sentence.split()]
    # Add <sos> and <eos> tokens
    indexes = [0] + indexes + [1] 
    # Convert to Tensor and reshape to [Seq Len, Batch Size]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

#############
## Encoder ##
#############
'''
This corresponds to the "Reader" from the video. 
Its only job is to take the input sentences and output the hidden and cell states (the Context Vector).
'''

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        # 1. Word Embeddings (Turning words into vectors)
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # 2. The LSTM (The actual processing unit)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [sequence_len, batch_size]
        
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # embedded shape: [sequence_len, batch_size, emb_dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # We discard 'outputs' because the Encoder only cares about the final summary
        # 'hidden' and 'cell' here are the Context Vector (the inputs for decoder)
        return hidden, cell
    
#############
## Decoder ##
#############
'''
This corresponds to the "Writer". 
Note that it takes a single token at a time (e.g., just "Au") alongside the hidden and cell states from the previous step.
'''

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        
        # 1. Embedding
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # 2. LSTM
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        
        # 3. Linear Layer (To predict the next word probability)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, hidden, cell):
        # inputs shape: [batch_size] (We process one word at a time)
        
        # Add a dimension for sequence length (which is 1 here)
        inputs = inputs.unsqueeze(0)
        
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        
        # Pass the inputs word + the Context Vector (hidden, cell)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell
        
#####################
## Seq2Seq wrapper ##
#####################
'''
This class ties everything together. 
It handles the loop where we feed the Decoder's output back into itself as the input for the next step.
'''

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: Input sentence (e.g., "So long")
        # trg: Target sentence (e.g., "Au revoir")
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 1. ENCODER STEP
        # Pass the source into the encoder to get the context vector
        hidden, cell = self.encoder(src)
        
        # 2. DECODER INITIALIZATION
        # The first input to the decoder is the <SOS> token (SOS = start of sequence)
        inputs = trg[0, :]
        
        # 3. DECODER LOOP
        for t in range(1, trg_len):
            
            # Pass input and context vector to decoder
            output, hidden, cell = self.decoder(inputs, hidden, cell)
            
            # Store prediction
            outputs[t] = output
            
            # TEACHER FORCING
            # Decide if we use the actual next word from data (teacher forcing)
            # or the model's predicted word.abs
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # If teacher forcing, next input is target token;
            # else it's predicted token
            inputs = trg[t] if teacher_force else top1
            
        return outputs
    
# --- Setup Model Hyperparameters ---
INPUT_DIM = input_vocab.n_words
OUTPUT_DIM = output_vocab.n_words
ENC_EMB_DIM = 16
DEC_EMB_DIM = 16
HID_DIM = 32
N_LAYERS = 1
ENC_DROPOUT = 0.0 # No dropout for such small data
DEC_DROPOUT = 0.0

# Initialize
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# Optimizer and Loss
# Ignore index 1 (<pad>) if we had padding, but here strictly speaking we ignore nothing or <unk>
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training Loop ---
print("Starting Training...")
model.train()

for epoch in range(100): # 100 epochs to overfit/memorize the data
    epoch_loss = 0
    
    for i in range(len(english)):
        src = sentence_to_tensor(input_vocab, english[i], device)
        trg = sentence_to_tensor(output_vocab, french[i], device)
        
        # Forward pass
        # output shape: [trg_len, batch_size, output_dim]
        output = model(src, trg)
        
        # Reshape for Loss Calculation
        # output_dim is the size of French vocab
        # We discard the first token of output because it corresponds to <sos> which we don't predict
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(english):.4f}")
'''
Starting Training...
Epoch 20 Loss: 0.4170
Epoch 40 Loss: 0.0532
Epoch 60 Loss: 0.0215
Epoch 80 Loss: 0.0123
Epoch 100 Loss: 0.0082
'''

#----------- Translation (Inference) -----------#

def translate_sentence(sentence, model, input_vocab, output_vocab, device, max_len=50):
    model.eval() # Turn off dropout
    
    with torch.no_grad():
        src_tensor = sentence_to_tensor(input_vocab, sentence, device)
        
        # 1. Encoder
        hidden, cell = model.encoder(src_tensor)
        
        # 2. Decoder Setup
        # Start with <sos> token (index 0)
        trg_indexes = [0] 
        
        # 3. Generate word by word
        for i in range(max_len):
            trg_tensor = torch.tensor([trg_indexes[-1]], device=device)
            
            # Predict next word (we pass hidden/cell from previous step)
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
            # Get highest probability word index
            pred_token = output.argmax(1).item()
            
            trg_indexes.append(pred_token)
            
            # Stop if model predicts <eos> (index 1)
            if pred_token == 1:
                break
    
    # Convert indexes back to words
    trg_tokens = [output_vocab.index2word[i] for i in trg_indexes]
    
    # Remove <sos> and <eos> for display
    return trg_tokens[1:-1]

# --- Test it! ---
test_sentence = "Better late than never"
translation = translate_sentence(test_sentence, model, input_vocab, output_vocab, device)

print(f"\nOriginal: {test_sentence}")
print(f"Translated: {' '.join(translation)}")
'''
Original: Better late than never
Translated: Mieux vaut tard que jamais
'''