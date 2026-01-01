'''
This file puts all the steps into a complete workflow:

0. Data simulation
1. Dataset splitting
2. Model building
3. Loss
4. Optimizer
5. Training - Validating loop
6. Drawing Train and Val loss curves
7. Testing
8. Saving model (model.state_dict() only)
9. Loading model (model.state_dict() only)
10. Inference (prediction)
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
## 1. Dataset splitting ##
##########################

train_len = int(0.7 * len(X)) # MUST be INTEGER
val_len = int(0.15 * len(X))
test_len = len(X) - (train_len + val_len)

from torch.utils.data import DataLoader, TensorDataset, random_split

full_dataset = TensorDataset(X, y)
train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])

train_set = DataLoader(train_split, batch_size=16, shuffle=True)
val_set = DataLoader(val_split, batch_size=16, shuffle=False)
test_set = DataLoader(test_split, batch_size=16, shuffle=False)

#######################
## 2. Model building ##
#######################

from torch import nn

class LinearRegressionModel(nn.Module):  
    def __init__(self):
        super().__init__()
        self.activate = nn.ReLU()
        self.layer_1 = nn.Linear(in_features=1, out_features=4, bias=True) # bias=True by default
        self.layer_2 = nn.Linear(4, 8)
        self.layer_3 = nn.Linear(8, 4)
        self.layer_out = nn.Linear(4, 1)
        
        '''
        nn.Linear: applies an affine linear transformation y = xAᵀ + b
        in_features: the number of input features (Our X has only 1 column, i.e 1D vector -> in_features = 1)
        out_features: the number of output features (Our y is also a 1D vector -> out_features = 1)
        
        use nn.ReLU() as activation function to capture non-linear patterns
        '''
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y1 = self.activate(self.layer_1(X))
        y2 = self.activate(self.layer_2(y1))
        y3 = self.activate(self.layer_3(y2))
        y_out = self.layer_out(y3) # The final layer does not need activation function (because this is regression)
        return y_out
        # return self.layer_out(self.activate(self.layer_3(self.activate(self.layer_2(self.activate(self.layer_1(X)))))))
    
'''Call out a model as an instance of a class'''
torch.manual_seed(42)
model = LinearRegressionModel()
model.to(device)

print(f'\nParameters before training:\n{model.state_dict()}') # Parameters before training
'''
Parameters before training:
OrderedDict({'layer_1.weight': tensor([[ 0.7645],
        [ 0.8300],
        [-0.2343],
        [ 0.9186]], device='cuda:0'), 'layer_1.bias': tensor([-0.2191,  0.2018, -0.4869,  0.5873], device='cuda:0'), 'layer_2.weight': tensor([[ 0.4408, -0.3668,  0.4346,  0.0936],
        [ 0.3694,  0.0677,  0.2411, -0.0706],
        [ 0.3854,  0.0739, -0.2334,  0.1274],
        [-0.2304, -0.0586, -0.2031,  0.3317],
        [-0.3947, -0.2305, -0.1412, -0.3006],
        [ 0.0472, -0.4938,  0.4516, -0.4247],
        [ 0.3860,  0.0832, -0.1624,  0.3090],
        [ 0.0779,  0.4040,  0.0547, -0.1577]], device='cuda:0'), 'layer_2.bias': tensor([ 0.1343, -0.1356,  0.2104,  0.4464,  0.2890, -0.2186,  0.2886,  0.0895],
       device='cuda:0'), 'layer_3.weight': tensor([[ 0.1795, -0.2155, -0.3500, -0.1366, -0.2712,  0.2901,  0.1018,  0.1464],
        [ 0.1118, -0.0062,  0.2767, -0.2512,  0.0223, -0.2413,  0.1090, -0.1218],
        [ 0.1083, -0.0737,  0.2932, -0.2096, -0.2109, -0.2109,  0.3180,  0.1178],
        [ 0.3402, -0.2918, -0.3507, -0.2766, -0.2378,  0.1432,  0.1266,  0.2938]],
       device='cuda:0'), 'layer_3.bias': tensor([-0.1826, -0.2410,  0.1876, -0.1429], device='cuda:0'), 'layer_out.weight': tensor([[ 0.3035, -0.1187,  0.2860, -0.3885]], device='cuda:0'), 'layer_out.bias': tensor([-0.2523], device='cuda:0')})
'''

#############
## 3. Loss ##
#############

loss_fn = nn.MSELoss() # mean squared error

##############################
## 4. Optimizer - Scheduler ##
##############################

optimizer = torch.optim.Adam(
    params=model.parameters(), 
    lr=0.1
)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.2)

###################################
## 5. Training - Validating loop ##
###################################

###################################
## 5. Training - Validating loop ##
###################################

epochs = 100

train_loss_list, val_loss_list = [], []

for epoch in range(1, epochs+1, 1):
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
    
    train_loss_list.append(loss.item())
    val_loss_list.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    if epoch % 10 == 0:
        print("+"*50)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")
    
'''
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 10
Train loss: 197.4626
Validation loss: 361.6837
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 20
Train loss: 25.7720
Validation loss: 44.8799
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 30
Train loss: 37.2075
Validation loss: 31.3751
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 40
Train loss: 57.6644
Validation loss: 29.4621
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 50
Train loss: 28.8694
Validation loss: 29.4546
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 60
Train loss: 29.2111
Validation loss: 29.4546
++++++++++++++++++++++++++++++++++++++++++++++++++
...
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 100
Train loss: 7.8877
Validation loss: 29.4546
'''

print(f'\nParameters after training:\n{model.state_dict()}') # Parameters after training
'''
Parameters after training:
OrderedDict({'layer_1.weight': tensor([[ 0.4082],
        [ 0.2574],
        [-0.2343],
        [ 0.3265]], device='cuda:0'), 'layer_1.bias': tensor([ 3.9902,  4.5269, -0.4869,  5.0715], device='cuda:0'), 'layer_2.weight': tensor([[ 0.6912,  0.0335,  0.4346,  0.5438],
        [-0.2311, -0.5328,  0.2411, -0.6711],
        [ 1.0081,  0.8675, -0.2334,  0.9812],
        [-0.8309, -0.6592, -0.2031, -0.2689],
        [-0.3947, -0.2305, -0.1412, -0.3006],
        [ 0.0472, -0.4938,  0.4516, -0.4247],
        [ 1.0080,  0.8785, -0.1624,  1.1658],
        [ 0.4461,  0.9356,  0.0547,  0.4300]], device='cuda:0'), 'layer_2.bias': tensor([ 4.1179, -0.7361,  4.9368, -0.1541,  0.2890, -0.2186,  5.1007,  4.4857],
       device='cuda:0'), 'layer_3.weight': tensor([[ 0.1795, -0.2155, -0.3500, -0.1366, -0.2712,  0.2901,  0.1018,  0.1464],
        [-0.6617, -0.6067, -0.4200, -0.8518,  0.0223, -0.2413, -0.5786, -0.8454],
        [ 0.8783,  0.5269,  1.2460,  0.3910, -0.2109, -0.2109,  1.3061,  0.9976],
        [-0.2157, -0.2918, -0.9066, -0.2766, -0.2378,  0.1432, -0.4293, -0.2621]],
       device='cuda:0'), 'layer_3.bias': tensor([-0.1826, -0.8825,  4.6285, -0.6988], device='cuda:0'), 'layer_out.weight': tensor([[0.3035, 0.4973, 1.2638, 0.1674]], device='cuda:0'), 'layer_out.bias': tensor([3.3210], device='cuda:0')})
'''

##########################################
## 6. Drawing Train and Val loss curves ##
##########################################

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
# Final Test Loss: 25.6673

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

################################
## 10. Inference (prediction) ##
################################

'''Create new input'''
import numpy as np

np.random.seed(43)
X_new = torch.tensor(
    np.random.uniform(low=1, high=11, size=(5, 1)),
    dtype=torch.float32,
    device=device
)
    
print(X_new)
# tensor([[2.1505],
#         [7.0907],
#         [2.3339],
#         [3.4059],
#         [4.2714]], device='cuda:0')

'''Inference'''
_ = loaded_model.eval() # Turn off gradient tracking

with torch.inference_mode():
    y_inference = loaded_model(X_new)
    
print(y_inference)
# tensor([[105.0232],
#         [126.8314],
#         [105.8327],
#         [110.5650],
#         [114.3857]], device='cuda:0')