'''
1. Prepare Data and Model

2. Loss function

3. Optimizer

4. Training Loop
'''

import torch
import numpy as np


#--------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. Prepare Data and Model  -----------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#

#################################
## Create X in ascending order ##
#################################

np.random.seed(24)
X = torch.tensor(
        np.random.uniform(low=1, high=11, size=(200, 1)),
        dtype=torch.float32,
        device='cpu'
    ).sort(dim=0).values

torch.manual_seed(24)
X += torch.normal(mean=2.5, std=1, size=(200, 1)) # Add variation

print(X[:10])
# tensor([[1.9797],
#         [3.2454],
#         [3.3088],
#         [2.5079],
#         [5.9215],
#         [3.3124],
#         [4.8586],
#         [2.3132],
#         [4.7322],
#         [4.9559]])

#################################
## Create y in ascending order ##
#################################

np.random.seed(25)
y = torch.tensor(
        np.random.uniform(low=100, high=150, size=(200,)),
        dtype=torch.float32,
        device='cpu'
    ).sort(dim=0).values

torch.manual_seed(25)
y += torch.normal(mean=10, std=1, size=(200,)) # Add variation

print(y[:10])
# tensor([110.4176, 110.1430, 111.1111, 109.7773, 110.7190, 112.1797, 113.0042,
#         112.2051, 113.8155, 111.9879])

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


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 2. Loss function -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
'''
Loss function is a function measures how poor the model performs.
The more different between truth and prediction, the higher the loss is
=> The lower the better.

There are many different Loss functions. Which one to use is problem specific.

###########

Loss functions live in torch.nn
https://docs.pytorch.org/docs/stable/nn.html#loss-functions
'''

'''Setup a loss function: use L1Loss (mean absolute error - MAE)'''
loss_fn = nn.L1Loss()

print(loss_fn)
# L1Loss()


#------------------------------------------------------------------------------------------------------#
#----------------------------------------- 3. Optimizer -----------------------------------------------#
#------------------------------------------------------------------------------------------------------#
'''
Optimizer will take into account the loss function and try to optimize it.
Meaning, it attempts to modify the parameters of the model based on the Loss function.
=> the goal is to achieve the parameter's values where the Loss function is smallest
                                                           (predictions get closest to truth)
                                                           
There are many different Optimizers. Which one to use is problem specific.

##################

Optimizers live in torch.optim
https://docs.pytorch.org/docs/stable/optim.html
'''

'''Set up an optimizer: use SGD (stochastic gradient descent)'''
optimizer = torch.optim.SGD(
    params=model.parameters(), # Parameters of the model that need to be optimized
    lr=0.001,                  # The higher the learning rate, the more the parameters will be adjusted after every training step
)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 4. Training Loop -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
'''
A couple of things we need in a training loop:
    0. Loop through the data
    1. Forward pass: this involes data moving through the 'forward()' method
    2. Calculate the loss: compare the predictions made by 'forward()' to ground truth values/labels
    3. Optimizer zero grad
    4. Loss backward (backpropagation): move backwards through the network 
                                        to calculate the gradients of each of the model's parameters with respect to the loss
    5. Optimizer step: use the optimizer to adjust our model's parameters to try and improve the loss
'''