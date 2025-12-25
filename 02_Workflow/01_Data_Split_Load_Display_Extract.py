'''
1. Data creation: simulate a positive correlated data

2. Splitting and Loading: Train, Validation, Test
   + Use iter(set) and for loop to display the split dataset
   + Extract X_train, y_train, X_val, y_val, X_test, y_test

3. (OPTIONAL) data visualization
'''

import torch
import numpy as np


#----------------------------------------------------------------------------------------------------------#
#----------------------------------------- 1. Data creation -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

#################################
## Create X in ascending order ##
#################################
'''X is written in uppercase to represent a MATRIX (must always be)'''

np.random.seed(24)
X = torch.tensor(
        np.random.uniform(low=1, high=11, size=(200, 1)),
        dtype=torch.float32,
        device='cpu'
    ).sort(dim=0).values # use .values to get the values only, not the indices

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
'''y is written in lowercase to represent a vector (must always be)'''

np.random.seed(25)
y = torch.tensor(
        np.random.uniform(low=100, high=150, size=(200,)),
        dtype=torch.float32,
        device='cpu'
    ).sort(dim=0).values # use .values to get the values only, not the indices

torch.manual_seed(25)
y += torch.normal(mean=10, std=1, size=(200,)) # Add variation

print(y[:10])
# tensor([110.4176, 110.1430, 111.1111, 109.7773, 110.7190, 112.1797, 113.0042,
#         112.2051, 113.8155, 111.9879])


#------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 3. Splitting and Loading -----------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
'''
Train set (learning):
+ for fitting the model to find optimized parameters
+ Always need
+ 60-80% original set

Validation set (practicing): 
+ to validate the trained model, and to finetune it
+ Optional
+ 10-20% original set

Test set (final exam): 
+ to see if the model is ready for "the wild"
+ Always need
+ 10-20% original set

##################

So first, we use the train set to train the model.
Then use the validation set to validate it, and make modifications or tuning to improve the model.
This could take a certain amount of time.

Finally, after tuning and validating with the validation set many cycles, we use the test set to test the model one last time,
to see if it is ready for "the wild"
'''

train_len = int(0.7 * len(X)) # MUST be INTEGER
val_len = int(0.15 * len(X))
test_len = len(X) - (train_len + val_len)

print(train_len, val_len, test_len)
# 140 30 30

##################
## Import tools ##
##################

from torch.utils.data import DataLoader, TensorDataset, random_split

###############################
## Use random_split to split ##
###############################

# Create a full dataset object
full_dataset = TensorDataset(X, y)

train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])
print(train_split, val_split, test_split)
# <torch.utils.data.dataset.Subset object at 0x76ab4bcf9160> <torch.utils.data.dataset.Subset object at 0x76ab4bcf8fb0> <torch.utils.data.dataset.Subset object at 0x76ab4bcf90a0>
'''This is not the split dataset yet'''

########################################
## Use DataLoader to load after split ##
########################################

train_set = DataLoader(train_split, batch_size=16, shuffle=True) # shuffle=True to reshuffle the data after every epoch
val_set = DataLoader(val_split, batch_size=16, shuffle=True)
test_set = DataLoader(test_split, batch_size=16, shuffle=True)

'''
Batch size refers to the number of training examples used in one single "iteration"
to update the model's internal parameters (weights).

Can think Batch as a subset of your total dataset. 
If you have 1,000 images and a batch size of 100, you have 10 batches.

Iteration (epoch): One single update of the model's weights. 
In the example above, it would take 10 iterations to finish one pass through the data.

A total "one pass through the data" is called an "epoch".
So, in the example above, one epoch = 10 iterations = 10x100 (batch size) = 1000 images
'''

print(val_set)
# <torch.utils.data.dataloader.DataLoader object at 0x76ab4c6b9970>
'''Still an object'''

############################################
## Use iter(set) with for loop to display ##
############################################

for batch in iter(val_set):
    print(batch)
# [tensor([[ 6.8403],
#         [14.0849],
#         [ 8.7103],
#         [ 8.5969],
#         [ 8.1303],
#         [ 6.7394],
#         [ 9.1344],
#         [ 8.7363],                  THIS IS X (first batch, len=16)
#         [12.6857],
#         [ 3.2847],
#         [12.5291],
#         [12.6861],
#         [ 8.3821],
#         [10.6946],
#         [12.7759],
#         [ 2.3132]]), tensor([118.0952, 150.6765, 130.9869, 137.4054, 134.8092, 131.5888, 136.7604,         THIS IS y (first batch, len=16)
#         135.6703, 152.5996, 112.9057, 146.4777, 152.5966, 130.8830, 136.6032,
#         157.1859, 112.2051])]
# [tensor([[ 8.8350],
#         [10.2011],
#         [11.5110],
#         [10.2900],
#         [ 9.6067],
#         [ 4.3744],                   THIS IS X (second batch, len=14)
#         [ 9.2037],
#         [ 7.7846],
#         [12.3906],
#         [12.8076],
#         [ 5.5858],
#         [ 9.7221],
#         [10.2690],
#         [ 6.7887]]), tensor([129.6186, 142.4344, 138.1428, 134.0313, 136.2205, 127.1605, 133.3897,         THIS IS y (second batch, len=14)
#         132.6374, 149.9803, 150.0055, 113.9603, 138.9602, 147.5087, 117.8481])]    

'''
Validation set has len=30
So the first batch = batch_size = 16
The second batch = 30 -16 = 14
'''

#############

'''Use next(iter(...)) to show the first batch'''
print(next(iter(test_set)))
# [tensor([[ 7.3018],
#         [ 2.6696],
#         [ 4.3445],
#         [10.7169],
#         [ 3.3088],
#         [ 5.7329],
#         [14.8453],
#         [ 6.0503],
#         [ 6.9569],
#         [ 5.7400],
#         [ 8.0558],
#         [11.5717],
#         [10.4460],
#         [11.6567],
#         [ 8.0847],
#         [ 4.3411]]), tensor([127.2862, 115.0915, 117.1131, 155.4955, 111.1111, 119.4410, 156.7803,
#         121.8189, 123.0551, 113.0312, 130.6982, 152.6853, 139.4277, 140.2617,
#         129.2378, 115.5734])]

'''Display X and y separately'''
batch_x, batch_y = next(iter(test_set))

print(batch_y)
# tensor([131.6067, 155.5813, 111.1111, 140.2617, 117.1131, 123.0551, 111.9879,
#         121.8189, 130.6982, 123.1350, 157.2921, 152.7896, 112.1797, 155.4955,
#         156.6623, 133.7207])

print(batch_x.shape) # torch.Size([16, 1])
print(batch_y.shape) # torch.Size([16])

############################################################
## Extract X_train, y_train, X_val, y_val, X_test, y_test ##
############################################################

def extract_values(dataset):
    import numpy as np
    X_list, y_list = [], []
    
    for batch in dataset:
        X_batch, y_batch = batch
        X_list.append(X_batch.reshape(-1).numpy())
        y_list.append(y_batch.cpu().numpy())
        
        X_arr = np.concatenate(X_list) # convert list of X_bactch into ndarray
        y_arr = np.concatenate(y_list) # convert list of y_bactch into ndarray

        return X_arr, y_arr

############

X_test, y_test = extract_values(test_set)

print(X_test)
# [ 6.0502653 12.6057005  7.968792   5.7329483 10.716923   6.956892
#   4.341133  12.05383   13.329298   8.055841   3.3123798  5.7399592
#  11.656659  14.845282  12.556503  10.5121565]

print(y_test)
# [121.81888  153.26802  133.72066  119.44098  155.49545  123.055145
#  115.57343  157.29207  155.58127  130.69817  112.17973  113.031166
#  140.26169  156.78029  150.6218   149.67296 ]


#--------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 2. (OPTIONAL) Data visualization -----------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------#


def data_plot(train, val, test):
    import plotly.graph_objects as pgo
    
    X_train, y_train = extract_values(train)
    X_val, y_val = extract_values(val)
    X_test, y_test = extract_values(test)
    
    # Create and empty figure
    fig = pgo.Figure()

    # Add the 1st scatter plot for Train set
    fig.add_trace(pgo.Scatter(
        x=X_train,
        y=y_train,
        mode="markers",
        marker=dict(size=12),
        name="Train"
    ))

    # Add the 2nd scatter plot for Val set
    fig.add_trace(pgo.Scatter(
        x=X_val,
        y=y_val,
        mode="markers",
        marker=dict(size=12),
        name="Validation"
    ))

    # Add the 3rd scatter plot for Test set
    fig.add_trace(pgo.Scatter(
        x=X_test,
        y=y_test,
        mode="markers",
        marker=dict(size=12),
        name="Test"
    ))

    fig.show()
    
data_plot(train_set, val_set, test_set)