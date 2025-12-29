'''
This file put all the steps into a complete workflow:

0. Data simulation
1. Dataset splitting
2. Model building
3. Loss
4. Optimizer
5. Training - Validating loop
6. Draw Train and Val loss curves
7. Testing
8. Saving model (model.state_dict() only)
9. Loading model (model.state_dict() only)
'''

import torch
import numpy as np

print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

########################
## 0. Data simulation ##
########################

#---- X -----
np.random.seed(24)
X = torch.tensor(
        np.random.uniform(low=1, high=11, size=(200, 1)),
        dtype=torch.float32,
        device=device
    ).sort(dim=0).values

torch.manual_seed(24)
X += torch.normal(mean=2.5, std=1, size=(200, 1), device=device) # Add variation

#---- y -----
np.random.seed(25)
y = torch.tensor(
        np.random.uniform(low=100, high=150, size=(200,)),
        dtype=torch.float32,
        device=device
    ).sort(dim=0).values

torch.manual_seed(25)
y += torch.normal(mean=10, std=1, size=(200,), device=device) # Add variation

##########################
## Train-Val-Test split ##
##########################

train_len = int(0.7 * len(X)) # MUST be INTEGER
val_len = int(0.15 * len(X))
test_len = len(X) - (train_len + val_len)

print(train_len, val_len, test_len)
# 140 30 30

from torch.utils.data import DataLoader, TensorDataset, random_split

full_dataset = TensorDataset(X, y)
train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])

train_set = DataLoader(train_split, batch_size=16, shuffle=True)
val_set = DataLoader(val_split, batch_size=16, shuffle=True)
test_set = DataLoader(test_split, batch_size=16, shuffle=True)

#################
## Build model ##
#################

from torch import nn

class LinearRegressionModel(nn.Module):  
    def __init__(self):
        super().__init__()
        self.coefs = nn.Parameter(torch.randn(size=(1, ), requires_grad=True, dtype=torch.float32)) # initialize self.coefs as a random number
        self.bias = nn.Parameter(torch.randn(size=(1, ), requires_grad=True, dtype=torch.float32)) # initialize self.bias as a random number
        
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.coefs*X + self.bias
    
##################
## Define model ##
##################

torch.manual_seed(42)
model = LinearRegressionModel()
model.to(device)

print(f'\nParameters before training:\n{model.state_dict()}') # Parameters before training
# OrderedDict({'coefs': tensor([0.3367], device='cuda:0'), 'bias': tensor([0.1288], device='cuda:0')})

##########
## Loss ##
##########

loss_fn = nn.MSELoss() # mean squared error

###############
## Optimizer ##
###############

optimizer = torch.optim.Adam(
    params=model.parameters(), 
    lr=9.3,
)

################################
## Training - Validating loop ##
################################

epochs = 10

train_loss_list, val_loss_list = [], []

for epoch in range(epochs):
    print("+"*50)
    # --- TRAINING ---
    _ = model.train() # Turn on training mode, enable gradient tracking
    for X_batch, y_batch in train_set:
        # (Standard training steps: forward, loss, zero_grad, backward, step)
        y_preds = model(X_batch).squeeze()
        loss = loss_fn(y_preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # --- VALIDATION (Every epoch) ---
    _ = model.eval() # 1. Set model to evaluation mode
    val_loss = 0
    with torch.inference_mode(): # 2. Turn off gradient tracking to save memory
        for X_val, y_val in val_set: # 3. Iterate through val_set
            val_preds = model(X_val).squeeze()
            # Accumulate loss to get an average for the whole set
            val_loss += loss_fn(val_preds, y_val).item()
    
    avg_val_loss = val_loss / len(val_set)
    
    print(f"Epoch: {epoch + 1}")
    print(f"Train loss: {loss:.4f}")
    print(f"Validation loss: {avg_val_loss:.4f}")
    
    train_loss_list.append(loss.item())
    val_loss_list.append(avg_val_loss)
    
'''
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 1
Train loss: 2364.0293
Validation loss: 3680.6317
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 2
Train loss: 3459.6724
Validation loss: 2119.7612
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 3
Train loss: 409.0802
Validation loss: 317.3638
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 4
Train loss: 354.7455
Validation loss: 328.5134
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 5
Train loss: 252.1030
Validation loss: 231.5259
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 6
Train loss: 25.2443
Validation loss: 85.0192
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 7
Train loss: 75.0284
Validation loss: 71.7913
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 8
Train loss: 38.7094
Validation loss: 49.6114
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 9
Train loss: 6.6849
Validation loss: 40.4321
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 10
Train loss: 29.3571
Validation loss: 35.8692
'''

print(f'\nParameters after training:\n{model.state_dict()}') # Parameters after training
# OrderedDict({'coefs': tensor([4.9168], device='cuda:0'), 'bias': tensor([93.1045], device='cuda:0')})

#######################################
## 6. Draw Train and Val loss curves ##
#######################################

def plot_train_val_loss_curves():
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
    
plot_train_val_loss_curves()

################
## 7. Testing ##
################

_ = model.eval()

test_loss = 0
with torch.inference_mode():
    for X_test, y_test in test_set: # Use the test_set loader
        test_preds = model(X_test)
        test_loss += loss_fn(test_preds, y_test.unsqueeze(1)).item()

print(f"Final Test Loss: {test_loss / len(test_set):.4f}")
# Final Test Loss: 39.2411

###############
## 8. Saving ##
###############

from pathlib import Path

MODEL_PATH = Path("02_General_Workflow").joinpath("save")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# PyTorch model can be saved in .pth or .pt format
PARAMS_NAME = "Altogether_StateDict.pth"

# Save the model (use model.state_dict() to save only the parameters)
torch.save(obj=model.state_dict(), f=MODEL_PATH.joinpath(PARAMS_NAME))

################
## 9. Loading ##
################

from pathlib import Path

MODEL_PATH = Path("02_General_Workflow").joinpath("save")
PARAMS_NAME = "Altogether_StateDict.pth"

# Create a new instance of the model class
loaded_model = LinearRegressionModel()
loaded_model.to(device)

loaded_model.load_state_dict(torch.load(MODEL_PATH.joinpath(PARAMS_NAME)))
# <All keys matched successfully>

