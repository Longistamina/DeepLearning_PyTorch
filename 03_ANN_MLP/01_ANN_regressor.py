'''
This code shows how to write an ANN to predict the median_house_value,
other remaining features are used for training
'''

import torch

print(torch.__version__)
# 2.11.0.dev20251216+cu130

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# cuda

######################
## Data preparation ##
######################

import polars as pl
from polars import col as c

pl_housing = (
    pl.read_csv(source="https://raw.githubusercontent.com/dmarks84/Ind_Project_California-Housing-Data--Kaggle/refs/heads/main/housing.csv")
    .to_dummies(columns="ocean_proximity", drop_first=True)
    .with_columns(total_bedrooms = c("total_bedrooms").fill_null(pl.col("total_bedrooms").median()))
    .cast(pl.Float32)
)
print(pl_housing.head())
# ┌─────────────┬───────────┬────────────────────┬─────────────┬───┬─────────────────────┬────────────────────────┬────────────────────────┬────────────────────────────┐
# │ longitude   ┆ latitude  ┆ housing_median_age ┆ total_rooms ┆ … ┆ ocean_proximity_<1H ┆ ocean_proximity_INLAND ┆ ocean_proximity_ISLAND ┆ ocean_proximity_NEAR OCEAN │
# │ ---         ┆ ---       ┆ ---                ┆ ---         ┆   ┆ OCEAN               ┆ ---                    ┆ ---                    ┆ ---                        │
# │ f32         ┆ f32       ┆ f32                ┆ f32         ┆   ┆ ---                 ┆ f32                    ┆ f32                    ┆ f32                        │
# │             ┆           ┆                    ┆             ┆   ┆ f32                 ┆                        ┆                        ┆                            │
# ╞═════════════╪═══════════╪════════════════════╪═════════════╪═══╪═════════════════════╪════════════════════════╪════════════════════════╪════════════════════════════╡
# │ -122.230003 ┆ 37.880001 ┆ 41.0               ┆ 880.0       ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# │ -122.220001 ┆ 37.860001 ┆ 21.0               ┆ 7099.0      ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# │ -122.239998 ┆ 37.849998 ┆ 52.0               ┆ 1467.0      ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# │ -122.25     ┆ 37.849998 ┆ 52.0               ┆ 1274.0      ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# │ -122.25     ┆ 37.849998 ┆ 52.0               ┆ 1627.0      ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# └─────────────┴───────────┴────────────────────┴─────────────┴───┴─────────────────────┴────────────────────────┴────────────────────────┴────────────────────────────┘

print(pl_housing.shape)
# (20640, 13)

'''----- X ------'''
from sklearn.preprocessing import RobustScaler

X_raw = pl_housing.select(c("*").exclude("median_house_value")).to_numpy()

X_scaler = RobustScaler().fit(X_raw)

X_scaled = torch.tensor(
    data=X_scaler.transform(X_raw),
    device=device
)

print(X_scaled)
# tensor([[-0.9868,  0.9577,  0.6316,  ...,  0.0000,  0.0000,  0.0000],
#         [-0.9842,  0.9524, -0.4211,  ...,  0.0000,  0.0000,  0.0000],
#         [-0.9894,  0.9497,  1.2105,  ...,  0.0000,  0.0000,  0.0000],
#         ...,
#         [-0.7203,  1.3677, -0.6316,  ...,  1.0000,  0.0000,  0.0000],
#         [-0.7467,  1.3677, -0.5789,  ...,  1.0000,  0.0000,  0.0000],
#         [-0.7256,  1.3519, -0.6842,  ...,  1.0000,  0.0000,  0.0000]],
#        device='cuda:0')

print(X_scaled.shape)
# torch.Size([20640, 12])
# Here, 12 will be the in_features

'''----- y ------'''
from sklearn.preprocessing import RobustScaler

y_raw = pl_housing.select("median_house_value").to_numpy()

y_scaler = RobustScaler().fit(y_raw)

y_scaled = torch.tensor(
    data=y_scaler.transform(y_raw),
    device=device
)

print(y_scaled)
# tensor([[ 1.8804],
#         [ 1.2320],
#         [ 1.1879],
#         ...,
#         [-0.6022],
#         [-0.6546],
#         [-0.6222]], device='cuda:0')

print(y_scaled.shape)
# torch.Size([20640, 1])
# Here, 1 will be the final out_features

#######################
## Dataset splitting ##
#######################

BATCH_SIZE = 2*11

train_len = int(0.8 * len(X_scaled)) # MUST be INTEGER
val_len = int(0.1 * len(X_scaled))
test_len = len(X_scaled) - (train_len + val_len)

from torch.utils.data import DataLoader, TensorDataset, random_split

full_dataset = TensorDataset(X_scaled, y_scaled)
train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])

train_set = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
val_set = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False)
test_set = DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=False)

####################
## Model building ##
####################

from torch import nn

class ANNregressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        '''Use nn.Sequential() to put our layers together'''
        self.mlp = nn.Sequential( # MLP = multilayer perceptron
            
            # Layer 1: 12 inputs -> 64 neurons (upscale from 12 features to 64 features)
            # use nn.ReLU() as activation since median_house_value is non-negative
            nn.Linear(in_features=12, out_features=64),
            nn.BatchNorm1d(64), # Normalize
            nn.ReLU(),
            
            # Layer 2: 64 neurons -> 32 neurons (downscale from 64 features to 32 features)
            nn.Linear(64, 32),
            nn.BatchNorm1d(32), # Normalize
            nn.ReLU(),
            
            # Layer 3: 32 neurons -> 1 output (No activation function here because this is regression!)
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.mlp(X)
    
'''Call out a model as an instance of a class'''
torch.manual_seed(42)
model = ANNregressor()
model.to(device)

# print(f'\nParameters before training:\n{model.state_dict()}')
# .....

##################################
## Loss - Optimizer - Scheduler ##
##################################

loss_fn = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(
    params=model.parameters(), 
    lr=0.01
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
Train loss: 7.284e-02
Validation loss: 4.741e-02
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 20
Train loss: 6.587e-02
Validation loss: 6.457e-02
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 30
Train loss: 5.257e-02
Validation loss: 7.061e-02
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 40
Train loss: 6.239e-02
Validation loss: 7.588e-02
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 50
Train loss: 7.191e-02
Validation loss: 1.074e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 60
Train loss: 8.633e-02
Validation loss: 6.086e-02
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 70
Train loss: 6.222e-02
Validation loss: 8.610e-02
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 80
Train loss: 5.920e-02
Validation loss: 7.495e-02
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 90
Train loss: 5.147e-02
Validation loss: 5.349e-02
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 100
Train loss: 6.496e-02
Validation loss: 6.314e-02
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

test_loss_scaled = 0
test_preds_list = []
test_true_list = []

_ = model.eval()

with torch.inference_mode():
    for X_test, y_test in test_set:
        test_preds = model(X_test)
        
        # Accumulate scaled loss
        test_loss_scaled += loss_fn(test_preds, y_test).item()
        
        # Collect predictions and true for inverse transform
        test_preds_list.append(test_preds)
        test_true_list.append(y_test)

# Calculate average scaled loss
avg_test_loss_scaled = test_loss_scaled / len(test_set)

# Concatenate all batches
test_preds_scaled = torch.cat(test_preds_list, dim=0)
test_true_scaled = torch.cat(test_true_list, dim=0)

# Inverse transform predictions and true (not the loss!)
test_preds_original = y_scaler.inverse_transform(test_preds_scaled.cpu().numpy())
test_true_original = y_scaler.inverse_transform(test_true_scaled.cpu().numpy())

# Now calculate metrics in original scale (dollars)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_original = mean_absolute_error(test_true_original, test_preds_original)
rmse_original = np.sqrt(mean_squared_error(test_true_original, test_preds_original))
r2 = r2_score(test_true_original, test_preds_original)

print("="*50)
print("Test Set Performance:")
print(f"Scaled Loss (SmoothL1): {avg_test_loss_scaled:.3e}")
print(f"MAE (Original $): ${mae_original:,.2f}")
print(f"RMSE (Original $): ${rmse_original:,.2f}")
print(f"R² Score: {r2:.4f}")
print("="*50)

# ==================================================
# Test Set Performance:
# Scaled Loss (SmoothL1): 4.237e-02
# MAE (Original $): $35,606.31
# RMSE (Original $): $50,719.14
# R² Score: 0.7986
# ==================================================

print(
    pl.DataFrame(
        {
            "House_preds": test_preds_original.squeeze(),
            "House_truth": test_true_original.squeeze()
        }
    )
)
# shape: (2_064, 2)
# ┌───────────────┬─────────────┐
# │ House_preds   ┆ House_truth │
# │ ---           ┆ ---         │
# │ f32           ┆ f32         │
# ╞═══════════════╪═════════════╡
# │ 86679.34375   ┆ 55000.0     │
# │ 168245.703125 ┆ 154200.0    │
# │ 171268.1875   ┆ 168300.0    │
# │ 193547.796875 ┆ 184400.0    │
# │ 186676.765625 ┆ 168800.0    │
# │ …             ┆ …           │
# │ 128984.015625 ┆ 159800.0    │
# │ 88847.523438  ┆ 112500.0    │
# │ 123695.71875  ┆ 111500.0    │
# │ 112775.296875 ┆ 102600.0    │
# │ 131399.171875 ┆ 119800.0    │
# └───────────────┴─────────────┘

import matplotlib.pyplot as plt
import seaborn as sbn

y_test_true = test_true_original.squeeze()
y_test_preds = test_preds_original.squeeze()

sbn.set_theme(style='darkgrid')
sbn.lineplot(x = [y_test_true.min(), y_test_true.max()], y = [y_test_true.min(), y_test_true.max()], label = 'Reference line', color = 'green')
sbn.scatterplot(x = y_test_true, y = y_test_preds)
plt.xlabel("Y_test_true", size=15)
plt.ylabel("Y_test_predict", size=15)
plt.show()

############
## Saving ##
############

from pathlib import Path

MODEL_PATH = Path("03_ArtificialNeuralNetwork_ANN").joinpath("save")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# PyTorch model can be saved in .pth or .pt format
PARAMS_NAME = "ANN_regressor.pth"

# Save the model (use model.state_dict() to save only the parameters)
torch.save(obj=model.state_dict(), f=MODEL_PATH.joinpath(PARAMS_NAME))