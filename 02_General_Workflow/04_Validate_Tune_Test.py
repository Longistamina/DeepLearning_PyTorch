'''
0. Prepare Data, Model, Loss, Optimizer

1. Integrate Validation and val_set into the Training loop.

2. Tune the model (explanation only)

3. Test the model with test_set (final exam)

4. (Optional) Visualize the y_preds and y_test

5. (Optional) Compare with sklearn LinearRegression
'''

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


#----------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- 0. Prepare Data, Model, Loss, Optimizer  -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

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


#----------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------- 1. Integrate Validation and val_set into the Training loop. -------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

epochs = 10

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
    
    print("+"*50)
    print(f"Epoch: {epoch}")
    print(f"Train loss: {loss:.4f}")
    print(f"Validation loss: {avg_val_loss:.4f}")
    
'''
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 1
Train loss: 1271.3748
Validation loss: 909.5925
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 2
Train loss: 2052.3772
Validation loss: 2003.3079
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 3
Train loss: 1224.3154
Validation loss: 887.6179
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 4
Train loss: 947.8354
Validation loss: 994.4067
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 5
Train loss: 1269.6996
Validation loss: 944.4518
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 6
Train loss: 315.4417
Validation loss: 804.8121
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 7
Train loss: 1195.7412
Validation loss: 861.0732
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 8
Train loss: 869.2205
Validation loss: 1422.4598
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 9
Train loss: 1641.3568
Validation loss: 891.3362
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 10
Train loss: 280.6619
Validation loss: 646.2153
'''

print(model.state_dict()) # Parameters after training
# OrderedDict({'coefs': tensor([11.2700], device='cuda:0'), 'bias': tensor([17.9875], device='cuda:0')})

#####################################
## Draw Train loss Val loss curves ##
#####################################

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


#----------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------- 2. Tune the model (explanation only) -----------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#
'''
Tuning or Fine-tuning typically means looking at your Validation Loss results and making decisions:

If Val Loss is much higher than Train Loss: Your model is "overfitting" (memorizing). 
=> You might need to simplify the model or add more data.

If both losses are high
=> You might need to increase the epochs or adjust the learning rate lr in your optimizer.

Adjustment: You change the parameters in your script (like lr=0.001) and run the whole training process again.

#################

Try the above model with lr=0.015 and lr=0.005 to see what happens.
'''


#----------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------- 3. Test the model with test_set (final exam) --------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

# --- FINAL TESTING (After all epochs are done) ---
_ = model.eval()

test_loss = 0
with torch.inference_mode():
    for X_test, y_test in test_set: # Use the test_set loader
        test_preds = model(X_test)
        test_loss += loss_fn(test_preds, y_test.unsqueeze(1)).item()

print(f"Final Test Loss: {test_loss / len(test_set):.4f}")
# Final Test Loss: 573.3136


#----------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------- 4. (Optional) Visualize the y_preds and y_test -------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

import seaborn as sbn
import matplotlib.pyplot as plt
import pandas as pd
import torch

# 1. Gather all test data and predictions
X_test_all, y_test_all, y_preds_all = [], [], []

_ = model.eval() # Set to evaluation mode
with torch.inference_mode(): # Turn off gradient tracking
    for X_batch, y_batch in test_set:
        # Use squeeze() to ensure y_preds matches the (batch_size,) shape of y_batch
        y_preds = model(X_batch).squeeze() 
        
        X_test_all.append(X_batch)
        y_test_all.append(y_batch)
        y_preds_all.append(y_preds)

# 2. Convert to a single Numpy array/DataFrame for Seaborn
X_test = torch.cat(X_test_all).cpu().numpy()
y_test = torch.cat(y_test_all).cpu().numpy()
y_test_preds = torch.cat(y_preds_all).cpu().numpy()

###################

df = pd.DataFrame({
    'Input (X)': X_test.flatten(),
    'Actual (y)': y_test,
    'Predicted (y)': y_test_preds
})

# Set the visual style
sbn.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))

# Plot the Actual Data points
sbn.scatterplot(data=df, x='Input (X)', y='Actual (y)', 
                color='royalblue', label='Actual Test Data', alpha=0.7, s=100)

# Plot the Model's Prediction line
# We sort by X to ensure the line draws smoothly from left to right
sbn.lineplot(data=df.sort_values('Input (X)'), x='Input (X)', y='Predicted (y)', 
             color='crimson', label='Model Prediction (Linear Regression)', linewidth=2.5)

# Customizing the layout
plt.title("Model Evaluation: Predictions vs. Ground Truth (PyTorch)", fontsize=15)
plt.xlabel(xlabel='Input (X)', fontsize=20)
plt.ylabel(ylabel='Output (y)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(3, 16)
plt.ylim(0, 170)
plt.legend()
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------ 5. (Optional) Compare with sklearn LinearRegression  ----------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

X_train_list, y_train_list = [], []
for X_batch, y_batch in train_set:
    X_train_list.append(X_batch)
    y_train_list.append(y_batch)

X_train = torch.cat(X_train_list).cpu().numpy()
y_train = torch.cat(y_train_list).cpu().numpy()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr_sklearn = LinearRegression().fit(X_train, y_train)

y_test_preds = lr_sklearn.predict(X_test)

print(mean_squared_error(y_test, y_test_preds))
# 25.54711151123047
'''Much lower than using torch'''

print(lr_sklearn.coef_, lr_sklearn.intercept_) # Parameters
# [4.411296] 95.52976

#################

df = pd.DataFrame({
    'Input (X)': X_test.flatten(),
    'Actual (y)': y_test,
    'Predicted (y)': y_test_preds
})

# Set the visual style
sbn.set_theme(style="darkgrid")

# Plot the Actual Data points
sbn.scatterplot(data=df, x='Input (X)', y='Actual (y)', 
                color='royalblue', label='Actual Test Data', alpha=0.7, s=100)

# Plot the Model's Prediction line
# We sort by X to ensure the line draws smoothly from left to right
sbn.lineplot(data=df.sort_values('Input (X)'), x='Input (X)', y='Predicted (y)', 
             color='crimson', label='Model Prediction (Linear Regression)', linewidth=2.5)

# Customizing the layout
plt.title("Model Evaluation: Predictions vs. Ground Truth (sklearn)", fontsize=15)
plt.xlabel(xlabel='Input (X)', fontsize=20)
plt.ylabel(ylabel='Output (y)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(3, 16)
plt.ylim(0, 170)
plt.legend()
plt.show()

'''
The MSE and plot show that sklearn performs better than torch
=> Some time, basic Machine Learning is better...
'''