'''
1. Data preparation

2. Build Model!!!!!
'''

import torch
import numpy as np


#-------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. Data preparation -----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------#

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


#-----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 2. Build Model!!! -----------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

from torch import nn
'''torch.nn has all the basic building blocks for a neural netowrk (computational graph)'''

class LinearRegressionModel(nn.Module): # nn.Module is the base class for all neural network modules. (Should always subclass this class)
    '''
    What will our model do?
    + First, it start with random values (coefficients and bias)
    + Then, it looks at the training data and adjust these random values
    + The goal is to find the best coefs and bias that best represent the training data (smallest errors)
    
    How it does so? Through two main algorithms:
    + Gradient descent
    + Backpropagation
    '''
    
    def __init__(self):
        super().__init__()
        self.coefs = nn.Parameter(torch.randn(size=1, requires_grad=True, dtype=torch.float32)) # initialize self.coefs as a random number
        self.bias = nn.Parameter(torch.randn(size=1, requires_grad=True, dtype=torch.float32)) # initialize self.bias as a random number
        
        
    # Forward method define the computation in the model
    # If you subclass nn.Module above, then should always overwrite forward()
    def forward(self, x: torch.Tensor) -> torch.Tensor: # Takes x as input (expected torch.Tensor), and also returns torch.Tensor as output
        return self.weights*x + self.bias