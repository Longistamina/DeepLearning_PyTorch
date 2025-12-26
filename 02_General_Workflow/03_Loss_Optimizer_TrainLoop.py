'''
0. Prepare Data and Model

1. Loss function

2. Optimizer

3. Training Loop
'''

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


#--------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 0. Prepare Data and Model  -----------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#

#################################
## Create X in ascending order ##
#################################

np.random.seed(24)
X = torch.tensor(
        np.random.uniform(low=1, high=11, size=(200, 1)),
        dtype=torch.float32,
        device=device
    ).sort(dim=0).values

torch.manual_seed(24)
X += torch.normal(mean=2.5, std=1, size=(200, 1), device=device) # Add variation

#################################
## Create y in ascending order ##
#################################

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

print(model.state_dict()) # parameters before training
# OrderedDict({'coefs': tensor([0.3367], device='cuda:0'), 'bias': tensor([0.1288], device='cuda:0')})


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. Loss function -----------------------------------------------#
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
#----------------------------------------- 2. Optimizer -----------------------------------------------#
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
    lr=0.25,                   # The higher the learning rate, the more the parameters will be adjusted after every training step
)


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 3. Training Loop -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
'''
A couple of things we need in a training loop:
    0. Loop through the data
    1. Forward pass: this involes data moving through the 'forward()' method
    2. Calculate the loss: compare the predictions made by 'forward()' to ground truth values/labels
    3. Optimizer zero grad: clear the old gradient step of the previous batch
    4. Loss backward (backpropagation): move backwards through the network 
                                        to calculate the gradients of each of the model's parameters with respect to the loss
    5. Optimizer step (gradient descent): use the optimizer to adjust our model's parameters to try and improve the loss
'''

# Set the number of epocsh (A total "one pass through the data")
epochs = 10

# 0. Loop through the data
for epoch in range(epochs):
    print("+"*50)
    
    # Set model to training mode (set all params that require gradients to require gradients)
    _ = model.train() # Assign to '_' to hide the class name printout
    
    # Create an inner loop to iterate over the train_set (a DataLoader object)
    for X_batch, y_batch in train_set:
        
        # 1. Forward pass (using the batch of features)
        y_preds = model(X_batch).squeeze() # Squeeze to make y_preds dimension the same as y_batch
        
        # 2. Calculate the loss (comparing batch predictions to batch truth values)
        loss = loss_fn(y_preds, y_batch) 
        
        # 3. Optimizer zero grad (clear the old gradient step of the previous loop)
        optimizer.zero_grad()
        
        # 4. Loss backward (perform backpropagation, calculate fresh gradient step for this batch)
        loss.backward()
        
        # 5. Optimizer step (perform gradient descent with new calculated step, to adjust the parameters)
        optimizer.step()
    
    # Print out the loss of each epoch (to see how the loss descends)
    print(f"Epoch: {epoch + 1}")
    print(f"Loss: {loss:.2f}")

'''
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 1
Loss: 21.40
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 2
Loss: 20.59
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 3
Loss: 20.63
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 4
Loss: 21.29
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 5
Loss: 18.55
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 6
Loss: 18.27
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 7
Loss: 15.72
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 8
Loss: 23.74
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 9
Loss: 14.79
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 10
Loss: 17.13
'''