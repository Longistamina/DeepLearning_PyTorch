'''
This code shows how to write an ANN to predict the median_house_value,
other remaining features are used for training
'''

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

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

train_len = int(0.8 * len(X_scaled)) # MUST be INTEGER
val_len = int(0.1 * len(X_scaled))
test_len = len(X_scaled) - (train_len + val_len)

from torch.utils.data import DataLoader, TensorDataset, random_split

full_dataset = TensorDataset(X_scaled, y_scaled)
train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])

train_set = DataLoader(train_split, batch_size=2**11, shuffle=True)
val_set = DataLoader(val_split, batch_size=2**11, shuffle=True)
test_set = DataLoader(test_split, batch_size=2**11, shuffle=True)

####################
## Model building ##
####################

from torch import nn

class ANNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        
        '''Use nn.Sequential() to put our layers together'''
        self.main = nn.Sequential(
            
            # Layer 1: 12 inputs -> 64 neurons (use nn.ReLU() as activation since median_house_value is non-negative)
            nn.Linear(in_features=12, out_features=64),
            nn.BatchNorm1d(64), # Normalize
            nn.ReLU(),
            
            # Layer 2: 64 neurons -> 32 neurons
            nn.Linear(64, 32),
            nn.BatchNorm1d(32), # Normalize
            nn.ReLU(),
            
            # Layer 3: 32 neurons -> 1 output (No activation function here because this is regression!)
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.main(X)
    
'''Call out a model as an instance of a class'''
torch.manual_seed(42)
model = ANNmodel()
model.to(device)

# print(f'\nParameters before training:\n{model.state_dict()}')
# .....

########################
## Loss and Optimizer ##
########################

loss_fn = nn.SmoothL1Loss()

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
    _ = model.eval() # 1. Set model to evaluation mode
    val_loss = 0
    with torch.inference_mode(): # 2. Turn off gradient tracking to save memory
        for X_val, y_val in val_set: # 3. Iterate through val_set
            val_preds = model(X_val)
            # Accumulate loss to get an average for the whole set
            val_loss += loss_fn(val_preds, y_val).item()
    
    avg_val_loss = val_loss / len(val_set)
    
    train_loss_list.append(loss.item())
    val_loss_list.append(avg_val_loss)
    
    scheduler.step()
    
    if epoch % 20 == 0:
        print("+"*50)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {loss:.3e}")
        print(f"Validation loss: {avg_val_loss:.3e}")
'''
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 20
Train loss: 1.230e-01
Validation loss: 1.276e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 40
Train loss: 1.634e-01
Validation loss: 1.082e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 60
Train loss: 1.553e-01
Validation loss: 1.064e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 80
Train loss: 1.412e-01
Validation loss: 1.590e-01
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 100
Train loss: 1.817e-01
Validation loss: 1.366e-01
'''