'''
1. Data preparation

2. Build Model!!!!!

3. Call out and inspect Model
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
        self.coefs = nn.Parameter(torch.randn(size=(1, ), requires_grad=True, dtype=torch.float32)) # initialize self.coefs as a random number
        self.bias = nn.Parameter(torch.randn(size=(1, ), requires_grad=True, dtype=torch.float32)) # initialize self.bias as a random number
        
        
    # Forward method define the computation in the model
    # If you subclass nn.Module above, then should always overwrite forward()
    def forward(self, X: torch.Tensor) -> torch.Tensor: # Takes X as input (expected torch.Tensor), and also returns torch.Tensor as output
        return self.coefs*X + self.bias
    

#---------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 3. Call out and inspect the Model -----------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#

'''Create a random seed to make parameters reproducible'''
torch.manual_seed(42)

'''Create an instance of the model (this is a subclass of nn.Module)'''
model = LinearRegressionModel()

'''Check out the parameters (must wrap model.parameters() inside a list() to display)'''
print(list(model.parameters()))
# [Parameter containing:
# tensor([0.3367], requires_grad=True), Parameter containing:
# tensor([0.1288], requires_grad=True)]

'''Use model.state_dict() to display also the names of parameters'''
print(model.state_dict())
# OrderedDict({'coefs': tensor([0.3367]), 'bias': tensor([0.1288])})

'''Use torch.inference_mode() to make pre-training predictions with initial random parameters'''
X_test, y_test = next(iter(test_set)) # Get X and y from the first batch in test_set

with torch.inference_mode():
    y_preds = model(X_test) # The model will feed X_test into the forward() method to do computing
                              # y_preds = self.coefs*X + self.bias
print(y_preds)
# tensor([[2.1659],
#         [1.7647],
#         [3.8870],
#         [4.3999],
#         [1.9160],
#         [4.3268],
#         [3.6682],
#         [2.8118],
#         [1.4676],
#         [2.4319],
#         [2.4145],
#         [2.0905],
#         [4.4410],
#         [2.5673],
#         [3.2276],
#         [2.1048]])

'''Put y_preds and y_test inside a dataframe for comparison'''
import polars as pl

df_preds_test = pl.DataFrame(
    {
        "y_preds": y_preds.squeeze().cpu().numpy(),
        "y_test": y_test.cpu().numpy()
    }
)

print(df_preds_test)
# shape: (16, 2)
# ┌──────────┬────────────┐
# │ y_preds  ┆ y_test     │
# │ ---      ┆ ---        │
# │ f32      ┆ f32        │
# ╞══════════╪════════════╡
# │ 2.165875 ┆ 121.818878 │
# │ 1.764662 ┆ 113.004158 │
# │ 3.887038 ┆ 146.144577 │
# │ 4.399948 ┆ 152.599579 │
# │ 1.915996 ┆ 116.432655 │
# │ …        ┆ …          │
# │ 2.09052  ┆ 132.405777 │
# │ 4.441008 ┆ 150.005478 │
# │ 2.56734  ┆ 128.960205 │
# │ 3.227596 ┆ 133.389664 │
# │ 2.104839 ┆ 123.967117 │
# └──────────┴────────────┘

'''Plot the y_test and y_preds'''
import matplotlib.pyplot as plt
import seaborn as sbn

sbn.scatterplot(x=X_test.squeeze().cpu().numpy(), y=df_preds_test['y_test'], label='y_test')
sbn.scatterplot(x=X_test.squeeze().cpu().numpy(), y=df_preds_test['y_preds'], label='y_preds')
plt.xlabel("X")
plt.ylabel('y')
plt.show()

'''
As we can see, before training, the predictions are very poor.
=> Must train the model to update its coefs and bias (parameters) for better predictions
'''