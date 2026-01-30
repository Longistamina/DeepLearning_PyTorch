from datasets import load_dataset

energy = load_dataset('nenils/time_series_energy', split='train')

energy = energy.to_polars().select(['timestamp', 'KW Demand']).rename(lambda c: c.replace(' ', '_'))

print(energy)
# shape: (1_439, 2)
# ┌────────────────┬───────────┐
# │ timestamp      ┆ KW_Demand │
# │ ---            ┆ ---       │
# │ str            ┆ f64       │
# ╞════════════════╪═══════════╡
# │ 01.06.23 00:00 ┆ 34.2      │
# │ 01.06.23 00:01 ┆ 34.06     │
# │ 01.06.23 00:02 ┆ 34.43     │
# │ 01.06.23 00:03 ┆ 34.09     │
# │ 01.06.23 00:04 ┆ 34.35     │
# │ …              ┆ …         │
# │ 01.06.23 23:55 ┆ 31.74     │
# │ 01.06.23 23:56 ┆ 31.82     │
# │ 01.06.23 23:57 ┆ 31.97     │
# │ 01.06.23 23:58 ┆ 31.95     │
# │ 01.06.23 23:59 ┆ 31.76     │
# └────────────────┴───────────┘

# ============================================
# STEP 1: PREPARE DATA
# ============================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = energy['KW_Demand'].to_numpy()

# Normalize data to [0, 1] range (RNNs work better with normalized data)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(x.reshape(-1, 1)).flatten()

print(data_scaled[:5])
# [0.44639509 0.44400886 0.45031532 0.4445202  0.44895176]

# ============================================
# STEP 2: CREATE SEQUENCES
# ============================================
# Use past N timesteps to predict next 1 timestep

def create_sequences(data, seq_length):
    """
    Convert time series into sequences
    Input: [34.2, 34.06, 34.43, 34.09, 34.35, ...]
    Output: 
        X: [[34.2, 34.06, 34.43], [34.06, 34.43, 34.09], ...]
        y: [34.09, 34.35, ...]
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

        # With SEQ_LENGTH = 10
        # X[0] = data[0:9], y[0] = data[9]
        # X[1] = data[1:10], y[1] = data[10]
        # ...
    return np.array(X), np.array(y)

# Use 10 minutes of data to predict next minute
SEQ_LENGTH = 10

X, y = create_sequences(data_scaled, SEQ_LENGTH)

print(f"\nSequence shape:")
print(f"X shape: {X.shape}  # (samples, sequence_length)")
print(f"y shape: {y.shape}  # (samples,)")

# ============================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================

import torch

train_size = int(0.8 * len(X))

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
# RNN expects input shape: (batch_size, seq_length, input_size)
X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # Add feature dimension
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

'''
Sequence shape:
X shape: (1429, 10)  # (samples, sequence_length)
y shape: (1429,)  # (samples,)

Training data shape: torch.Size([1143, 10, 1])
Testing data shape: torch.Size([286, 10, 1])
'''

# ============================================
# STEP 4: BUILD RNN MODEL
# ============================================

import torch
from torch import nn

class StackedRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=3, output_size=1):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size, # 1
            hidden_size=hidden_size, # 32
            num_layers=3, # 3 stacked RNN layers
            batch_first=True  # Input: (batch, seq, feature)
        )

        # Fully connected layer to produce output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, h_n = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

model = StackedRNN(num_layers=3)

def count_parameters(model):
    return f'{sum(p.numel() for p in model.parameters()):,}'

count_parameters(model)
# '5,377'

# ============================================
# STEP 5: TRAIN THE MODEL
# ============================================

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
batch_size = 32

print("\nTraining...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass
        predictions = model(batch_X).squeeze()
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train):.6f}")

'''
Training...
Epoch [10/50], Loss: 0.000151
Epoch [20/50], Loss: 0.000128
Epoch [30/50], Loss: 0.000117
Epoch [40/50], Loss: 0.000114
Epoch [50/50], Loss: 0.000114
'''

# ============================================
# STEP 6: MAKE PREDICTIONS
# ============================================

model.eval()
with torch.no_grad():
    # Predict on test set
    test_predictions = model(X_test).squeeze().numpy()
    
    # Inverse transform to get actual KW values
    test_predictions_actual = scaler.inverse_transform(
        test_predictions.reshape(-1, 1)
    ).flatten()
    
    y_test_actual = scaler.inverse_transform(
        y_test.numpy().reshape(-1, 1)
    ).flatten()

print("\n" + "="*60)
print("PREDICTIONS vs ACTUAL")
print("="*60)
print(f"{'Time':>6} | {'Actual':>10} | {'Predicted':>10} | {'Error':>10}")
print("-" * 60)

for i in range(10):  # Show first 10 predictions
    actual = y_test_actual[i]
    pred = test_predictions_actual[i]
    error = abs(actual - pred)
    print(f"{i:6} | {actual:10.2f} | {pred:10.2f} | {error:10.2f}")

# Calculate overall error
mae = np.mean(np.abs(y_test_actual - test_predictions_actual))
print(f"\nMean Absolute Error: {mae:.2f} KW")

'''
============================================================
PREDICTIONS vs ACTUAL
============================================================
  Time |     Actual |  Predicted |      Error
------------------------------------------------------------
     0 |      14.27 |      10.85 |       3.42
     1 |      19.24 |      12.38 |       6.86
     2 |      29.02 |      15.23 |      13.79
     3 |      21.26 |      22.23 |       0.97
     4 |      27.91 |      19.48 |       8.43
     5 |      29.01 |      26.53 |       2.48
     6 |      19.54 |      26.89 |       7.35
     7 |      28.45 |      20.95 |       7.50
     8 |      31.72 |      27.53 |       4.19
     9 |      32.05 |      27.67 |       4.38

Mean Absolute Error: 1.93 KW
'''