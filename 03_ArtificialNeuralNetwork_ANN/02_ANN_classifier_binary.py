'''
This code shows how to write an ANN to predict whether the income of a person is <=50k or vice versa (>50k) (binary classes),
other remaining features are used for training
'''

import torch

print(torch.__version__)
# 2.11.0.dev20251216+cu130

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# cuda

######################
## Data preparation ##
######################

import polars as pl
from polars import col as c

pl_adult = (
    pl.read_csv("https://raw.githubusercontent.com/saravrajavelu/Adult-Income-Analysis/refs/heads/master/adult.csv")
    .to_dummies(columns=pl.selectors.string(), drop_first=True)
    .cast(pl.Float32)
)

print(pl_adult)
# shape: (48_842, 101)
# ┌──────┬────────────┬────────────┬────────────┬───┬────────────┬───────────┬───────────┬───────────┐
# │ age  ┆ workclass_ ┆ workclass_ ┆ workclass_ ┆ … ┆ native-cou ┆ native-co ┆ native-co ┆ income_>5 │
# │ ---  ┆ ?          ┆ Federal-go ┆ Local-gov  ┆   ┆ ntry_Trina ┆ untry_Vie ┆ untry_Yug ┆ 0K        │
# │ f32  ┆ ---        ┆ v          ┆ ---        ┆   ┆ dad&Tobago ┆ tnam      ┆ oslavia   ┆ ---       │
# │      ┆ f32        ┆ ---        ┆ f32        ┆   ┆ ---        ┆ ---       ┆ ---       ┆ f32       │
# │      ┆            ┆ f32        ┆            ┆   ┆ f32        ┆ f32       ┆ f32       ┆           │
# ╞══════╪════════════╪════════════╪════════════╪═══╪════════════╪═══════════╪═══════════╪═══════════╡
# │ 25.0 ┆ 0.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 0.0       │
# │ 38.0 ┆ 0.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 0.0       │
# │ 28.0 ┆ 0.0        ┆ 0.0        ┆ 1.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 1.0       │
# │ 44.0 ┆ 0.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 1.0       │
# │ 18.0 ┆ 1.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 0.0       │
# │ …    ┆ …          ┆ …          ┆ …          ┆ … ┆ …          ┆ …         ┆ …         ┆ …         │
# │ 27.0 ┆ 0.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 0.0       │
# │ 40.0 ┆ 0.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 1.0       │
# │ 58.0 ┆ 0.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 0.0       │
# │ 22.0 ┆ 0.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 0.0       │
# │ 52.0 ┆ 0.0        ┆ 0.0        ┆ 0.0        ┆ … ┆ 0.0        ┆ 0.0       ┆ 0.0       ┆ 1.0       │
# └──────┴────────────┴────────────┴────────────┴───┴────────────┴───────────┴───────────┴───────────┘

'''----- X ------'''
from sklearn.preprocessing import StandardScaler

X_raw = pl_adult.select(c("*").exclude("income_>50K")).to_numpy()

X_scaler = StandardScaler().fit(X_raw)

X_scaled = torch.tensor(
    data=X_scaler.transform(X_raw),
    device=device
)

print(X_scaled)
# tensor([[-0.9951, -0.2466, -0.1738,  ..., -0.0235, -0.0420, -0.0217],
#         [-0.0469, -0.2466, -0.1738,  ..., -0.0235, -0.0420, -0.0217],
#         [-0.7763, -0.2466, -0.1738,  ..., -0.0235, -0.0420, -0.0217],
#         ...,
#         [ 1.4118, -0.2466, -0.1738,  ..., -0.0235, -0.0420, -0.0217],
#         [-1.2139, -0.2466, -0.1738,  ..., -0.0235, -0.0420, -0.0217],
#         [ 0.9742, -0.2466, -0.1738,  ..., -0.0235, -0.0420, -0.0217]],
#        device='cuda:0')

print(X_scaled.shape)
# torch.Size([48842, 100])
# Here, 100 will be the in_features

'''----- y ------'''
y = torch.tensor(
    data=pl_adult.select("income_>50K").to_numpy(),
    device=device
)

print(y)
# tensor([[-0.5608],
#         [-0.5608],
#         [ 1.7830],
#         ...,
#         [-0.5608],
#         [-0.5608],
#         [ 1.7830]], device='cuda:0')

print(y.shape)
# torch.Size([48842, 1])
# Here, 1 will be the final out_features

#######################
## Dataset splitting ##
#######################

BATCH_SIZE = 2**11

train_len = int(0.8 * len(X_scaled)) # MUST be INTEGER
val_len = int(0.1 * len(X_scaled))
test_len = len(X_scaled) - (train_len + val_len)

from torch.utils.data import DataLoader, TensorDataset, random_split

full_dataset = TensorDataset(X_scaled, y)
train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])

train_set = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
val_set = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False)
test_set = DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=False)

####################
## Model building ##
####################

from torch import nn

class ANNbinary(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            # Layer 1: 100 inputs -> 128 neurons
            nn.Linear(in_features=100, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Prevent overfitting
            
            # Layer 2: 128 -> 64 neurons
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3: 64 -> 32 neurons
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer: 32 -> 1 output with Sigmoid for binary classification
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, X):
        return self.mlp(X)
    
'''Call out a model as an instance of a class'''
torch.manual_seed(42)
model = ANNbinary()
model.to(device)

# print(f'\nParameters before training:\n{model.state_dict()}')
# .....

##################################
## Loss - Optimizer - Scheduler ##
##################################

loss_fn = nn.BCELoss() # Binary Cross-Entropy loss

optimizer = torch.optim.Adam(
    params=model.parameters(), 
    lr=1
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

################################
## Training - Validating loop ##
################################

epochs = 100

train_loss_list, val_loss_list = [], []

for epoch in range(1, epochs+1, 1):
    # --- TRAINING ---
    _ = model.train()
    for X_batch, y_batch in train_set:
        y_preds = model(X_batch)
        loss = loss_fn(y_preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # --- VALIDATION (Every epoch) ---
    _ = model.eval() 
    val_loss = 0
    with torch.inference_mode(): 
        for X_val, y_val in val_set: 
            val_preds = model(X_val)
            val_loss += loss_fn(val_preds, y_val).item()
    
    avg_val_loss = val_loss / len(val_set)
    
    train_loss_list.append(loss.item())
    val_loss_list.append(avg_val_loss)
    
    scheduler.step()
    
    if epoch % 10 == 0:
        print("+"*50)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {loss:.3e}")
        print(f"Validation loss: {avg_val_loss:.3e}")
'''
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 10
Train loss: 5.122e-01
Validation loss: 3.773e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 20
Train loss: 4.507e-01
Validation loss: 3.374e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 30
Train loss: 3.235e-01
Validation loss: 3.325e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 40
Train loss: 2.940e-01
Validation loss: 3.311e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 50
Train loss: 3.273e-01
Validation loss: 3.603e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 60
Train loss: 5.061e-01
Validation loss: 3.284e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 70
Train loss: 5.282e-01
Validation loss: 3.283e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 80
Train loss: 3.572e-01
Validation loss: 3.310e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 90
Train loss: 3.037e-01
Validation loss: 3.292e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 100
Train loss: 3.575e-01
Validation loss: 3.400e-01
'''

#######################################
## Drawing Train and Val loss curves ##
#######################################

def plot_train_val_loss_curves(epochs, train_loss_list, val_loss_list):
    import plotly.graph_objects as pgo
    import numpy as np
    
    # 1. Define the X-axis (epochs)
    epoch_axis = np.arange(1, epochs + 1, 1)

    fig = pgo.Figure()

    # 2. Add Training Loss
    fig.add_trace(pgo.Scatter(
        x=epoch_axis,
        y=train_loss_list,
        mode='lines+markers',
        name='Train Loss',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    # 3. Add Validation Loss
    fig.add_trace(pgo.Scatter(
        x=epoch_axis,
        y=val_loss_list,
        mode='lines+markers',
        name='Val Loss',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8, symbol='square')
    ))

    # 4. Layout & Styling
    fig.update_layout(
        title='<b>Model Training Progress</b>',
        xaxis_title='Epoch',
        yaxis_title='Loss Value',
        template='plotly_dark', # Clean dark background
        hovermode='x unified',   # Shows both values on hover
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    fig.show()
    
plot_train_val_loss_curves(epochs, train_loss_list, val_loss_list)

#############
## Testing ##
#############

test_loss = 0
test_preds_list = []
test_true_list = []

_ = model.eval()

with torch.inference_mode():
    for X_test, y_test in test_set:
        test_preds = model(X_test)  # Probabilities from sigmoid
        
        # Accumulate loss
        test_loss += loss_fn(test_preds, y_test).item()
        
        # Collect predictions and true
        test_preds_list.append(test_preds)
        test_true_list.append(y_test)

# Calculate average test loss
avg_test_loss = test_loss / len(test_set)
print(f"Average Test Loss: {avg_test_loss:.4f}\n")
# Average Test Loss: 0.3241

# Concatenate all batches
test_preds_proba = torch.cat(test_preds_list, dim=0).cpu().numpy()  # Probabilities
test_true = torch.cat(test_true_list, dim=0).cpu().numpy()

# Convert probabilities to class predictions (threshold = 0.5)
test_preds_class = (test_preds_proba >= 0.5).astype(int)

# Calculate metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

accuracy = accuracy_score(test_true, test_preds_class)
print(f'Accuracy on test set: {accuracy:.4f}\n')
# Accuracy on test set: 0.8538

# Confusion Matrix
labels = ['<=50K', '>50K']  # Adjust based on your dataset
cm = confusion_matrix(test_true, test_preds_class)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print(f'Confusion matrix:\n{cm_df}\n')
# Confusion matrix:
#        <=50K  >50K
# <=50K   3527   179
# >50K     535   644

# Classification Report
print(f'Classification report:\n{classification_report(test_true, test_preds_class, target_names=labels)}\n')
# Classification report:
#               precision    recall  f1-score   support

#        <=50K       0.87      0.95      0.91      3706
#         >50K       0.78      0.55      0.64      1179

#     accuracy                           0.85      4885
#    macro avg       0.83      0.75      0.78      4885
# weighted avg       0.85      0.85      0.84      4885

###############################
## Visualization with Plotly ##
###############################

# Calculate ROC metrics
fpr, tpr, thresholds = roc_curve(test_true, test_preds_proba)
roc_auc = auc(fpr, tpr)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(f'Confusion Matrix<br>Accuracy: {accuracy:.4f}', 
                    f'ROC Curve<br>AUC: {roc_auc:.4f}'),
    specs=[[{'type': 'heatmap'}, {'type': 'scatter'}]],
    horizontal_spacing=0.12
)

# Plot 1: Confusion Matrix Heatmap
fig.add_trace(
    go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 18, "color": "green"},
        colorscale='matter',
        showscale=True,
        colorbar=dict(x=0.46)  # Position colorbar between subplots
    ),
    row=1, col=1
)

fig.update_xaxes(title_text="Predicted values", row=1, col=1)
fig.update_yaxes(title_text="True values", row=1, col=1)

# Plot 2: ROC Curve
# Reference line (diagonal)
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Reference line',
        line=dict(color='blue', dash='dash', width=3),
        showlegend=True
    ),
    row=1, col=2
)

# ROC Curve
fig.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC={roc_auc:.4f})',
        line=dict(color='#FFA500', width=3),
        showlegend=True
    ),
    row=1, col=2
)

fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

# Update overall layout with darker theme
fig.update_layout(
    showlegend=True,
    template='plotly_dark',  # Dark theme
    legend=dict(
        yanchor="bottom",
        y=0.1,
        xanchor="right",
        x=0.95,
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="gray",
        borderwidth=1
    ),
    paper_bgcolor='gray',
    plot_bgcolor='#e5f5e0'
)

fig.show()

############
## Saving ##
############

from pathlib import Path

MODEL_PATH = Path("03_ArtificialNeuralNetwork_ANN").joinpath("save")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# PyTorch model can be saved in .pth or .pt format
PARAMS_NAME = "ANN_classifier_binary.pth"

# Save the model (use model.state_dict() to save only the parameters)
torch.save(obj=model.state_dict(), f=MODEL_PATH.joinpath(PARAMS_NAME))