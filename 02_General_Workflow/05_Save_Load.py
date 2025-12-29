'''
0. Data Prepare, Model building and Training

1. Save Model: torch.save()

2. Load Model: torch.load() and torch.nn.Module.load_state_dict()

3. Save and Load entire model (less recommended)
'''

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


#--------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------- 0. Data Prepare, Model building and Training --------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------#

#####################
## Data simulation ##
#####################

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

print(model.state_dict()) # Parameters before training
# OrderedDict({'coefs': tensor([0.3367], device='cuda:0'), 'bias': tensor([0.1288], device='cuda:0')})

##########
## Loss ##
##########

loss_fn = nn.MSELoss() # mean squared error

###############
## Optimizer ##
###############

optimizer = torch.optim.SGD(
    params=model.parameters(), 
    lr=0.01,                   
)

#################
## Train model ##
#################

print(f"Parameters before training:\n{model.state_dict()}")
# OrderedDict({'coefs': tensor([0.3367], device='cuda:0'), 'bias': tensor([0.1288], device='cuda:0')})

epochs = 10

for epoch in range(epochs):

    _ = model.train() # Turn on training mode, enable gradient tracking
    for X_batch, y_batch in train_set:
        # (Standard training steps: forward, loss, zero_grad, backward, step)
        y_preds = model(X_batch).squeeze()
        loss = loss_fn(y_preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
_ = model.eval() # Turn off training mode, disable gradient tracking

print(f"Parameters after training:\n{model.state_dict()}")
# OrderedDict({'coefs': tensor([9.3961], device='cuda:0'), 'bias': tensor([18.0144], device='cuda:0')})


#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. Save Model: torch.save()  -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

from pathlib import Path

MODEL_PATH = Path("02_General_Workflow").joinpath("save")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# PyTorch model can be saved in .pth or .pt format
PARAMS_NAME = "LinearRegression_StateDict.pth"

# Save the model (use model.state_dict() to save only the parameters)
torch.save(obj=model.state_dict(), f=MODEL_PATH.joinpath(PARAMS_NAME))


#----------------------------------------------------------------------------------------------------------------------#
#-------------------------- 2. Load Model: torch.load() and torch.nn.Module.load_state_dict() -------------------------#
#----------------------------------------------------------------------------------------------------------------------#
'''
Since we saved only the model's state_dict, we need to create a new instance of the model class first,
then load the saved state_dict into this new instance.
'''

from pathlib import Path

MODEL_PATH = Path("02_General_Workflow").joinpath("save")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# PyTorch model can be saved in .pth or .pt format
PARAMS_NAME = "LinearRegression_StateDict.pth"

# Create a new instance of the model class
loaded_model = LinearRegressionModel()
loaded_model.to(device)

'''Params before loading state_dict'''
print(f"Parameters before loading state_dict:\n{loaded_model.state_dict()}")
# OrderedDict({'coefs': tensor([-0.3278], device='cuda:0'), 'bias': tensor([0.7950], device='cuda:0')})

'''
Load the saved state_dict into the new model instance

Since our model is a subclass of nn.Module, we can use the load_state_dict() method.
(torch.nn.Module.load_state_dict() is the method of nn.Module class)
'''
loaded_model.load_state_dict(torch.load(MODEL_PATH.joinpath(PARAMS_NAME)))
# <All keys matched successfully>

'''Params after loading state_dict'''
print(f"Parameters after loading state_dict:\n{loaded_model.state_dict()}")
# OrderedDict({'coefs': tensor([9.3961], device='cuda:0'), 'bias': tensor([18.0144], device='cuda:0')})


#----------------------------------------------------------------------------------------------------------------------#
#----------------------------- 3. Save and Load entire model (less recommended) ---------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

#######################
## Save entire model ##
#######################

from pathlib import Path

MODEL_PATH = Path("02_General_Workflow").joinpath("save")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# PyTorch model can be saved in .pth or .pt format
MODEL_NAME = "LinearRegression_Model.pth"

# Save the entire model
torch.save(obj=model, f=MODEL_PATH.joinpath(MODEL_NAME))

#######################
## Load entire model ##
#######################

loaded_entire_model = torch.load(f=MODEL_PATH.joinpath(MODEL_NAME), weights_only=False) # MUST set weights_only=False

print(loaded_entire_model)
# LinearRegressionModel()

print(loaded_entire_model.state_dict())
# OrderedDict({'coefs': tensor([9.3961], device='cuda:0'), 'bias': tensor([18.0144], device='cuda:0')})