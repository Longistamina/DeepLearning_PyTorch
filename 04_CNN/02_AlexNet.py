'''
AlexNet architecture
+---------+-----------------+-------------------------------+------------------+
| Layer   | Type            | Configuration                 | Output Size      |
+---------+-----------------+-------------------------------+------------------+
| Input   | Image           | 227 x 227 x 3 (RGB)           | 227 x 227 x 3    |
| Conv 1  | Convolution     | 96 filters (11x11), Stride 4  | 55 x 55 x 96     |
| Pool 1  | Max Pooling     | 3x3 window, Stride 2          | 27 x 27 x 96     |
| Conv 2  | Convolution     | 256 filters (5x5), Padding 2  | 27 x 27 x 256    |
| Pool 2  | Max Pooling     | 3x3 window, Stride 2          | 13 x 13 x 256    |
| Conv 3  | Convolution     | 384 filters (3x3), Padding 1  | 13 x 13 x 384    |
| Conv 4  | Convolution     | 384 filters (3x3), Padding 1  | 13 x 13 x 384    |
| Conv 5  | Convolution     | 256 filters (3x3), Padding 1  | 13 x 13 x 256    |
| Pool 3  | Max Pooling     | 3x3 window, Stride 2          | 6 x 6 x 256      |
| FC 6    | Fully Connected | 4096 Neurons + Dropout        | 4096             |
| FC 7    | Fully Connected | 4096 Neurons + Dropout        | 4096             |
| FC 8    | Fully Connected | 1000 Neurons (Softmax)        | 1000             |
+---------+-----------------+-------------------------------+------------------+

After the Conv3 and Conv4, we don't add MaxPool or AvgPool to avoid shrinking too quickly,
losing critical spatial information before the network could extract high-level features.
'''

#########################
## Importing libraries ##
#########################

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# cuda

#########################
## Dataset downloading ##
#########################
'''
Import CIFAR-10 dataset from HuggingFace
Run this first in terminal: pip install datasets
'''

from datasets import load_dataset

train_set = load_dataset(
    'cifar10',
    split='train', # Download the training set
    verification_mode='basic_checks'  # checks if the data files exist and verifies basic metadata
)
print(train_set)
# Dataset({
#     features: ['img', 'label'],
#     num_rows: 50000
# })

val_set =  load_dataset(
    'cifar10',
    split='test', # Download the training set
    verification_mode='basic_checks'  # checks if the data files exist and verifies basic metadata
)
print(val_set)
# Dataset({
#     features: ['img', 'label'],
#     num_rows: 10000
# })

# View an image
train_set[0]['img']

print(train_set[0]['img'])
# <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x7A2EBA2F0410>

#########################
## Image preprocessing ##
#########################

IMG_SIZE = 32
'''
Most CNNs are designed to only accept images of a fixed size
=> Must fix the IMG_SIZE, and reshape the input to adapt this norm.
'''

#----
## Build preprocess transforms
#----

preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize the input image to a given size (IMG_SIZE, IMG_SIZE)
        transforms.ToTensor()                  # Convert to tensor (and also convert to [0, 1] tensors)
    ]
)

#----
## Change from grayscale to RGB, and apply preprocess
#----

inputs_train = []

for record in tqdm(iterable=train_set, desc="Preprocessing Images"):
    image = record['img']
    label = record['label']
    
    # Convert from grayscale to RGB (3-colour channels)
    if image.mode == "L":
        image = image.convert("RGB")
        
    # preprocessing
    input_tensor = preprocess(image)
    label_tensor = torch.tensor(label)
    
    # append to inputs_train
    inputs_train.append([input_tensor, torch.tensor(label)])

#----
## Re-normalize the pixel values for train set
#----
'''
Since transforms.Tensor() normalizes all into [0, 1],
we need to modify this normalization to fit this dataset.

Doing so by calculating the mean and std for all images across separe 3 color channles
then use transforms.Normalize(mean=, std=) with this calculated mean and std.
'''

# First, we need to calculate the mean and std for each of the RGB channels across all images

import numpy as np

# Choosing a random sample to calculate mean and std (this sample containing random 512 images)
np.random.seed(0)
idx = np.random.randint(0, len(inputs_train), 512)

# Concatenate this subset of images into a new tensor )tensor_placeholder)
tensor_placeholder = torch.concat([inputs_train[i][0] for i in idx], axis=1)
print(tensor_placeholder.shape)
# torch.Size([3, 16384, 32])
'''
we concatenate 512 images of size (3x32x32) (Channel*Height*Width) along the Height channel
=> (3x16384x32), 16384=32*512
'''

# Calculate the mean and std across all images, for separate channel
mean_all = torch.mean(tensor_placeholder, dim=(1, 2)) # dim=(1, 2) meanin only uses Heigh*Width for calculation, ignore the channel
std_all = torch.std(tensor_placeholder, dim=(1, 2))

print(mean_all) # tensor([0.4855, 0.4792, 0.4421])
print(std_all) # tensor([0.2464, 0.2418, 0.2599])

#### RE-NORMALIZE ###

preprocess = transforms.Compose([transforms.Normalize(mean=mean_all, std=std_all)])

for idx in tqdm(range(len(inputs_train))):
    input_tensor = preprocess(inputs_train[idx][0])
    inputs_train[idx][0] = input_tensor # replace with re-normalized tensor
    
#----
## Re-normalize the pixel values for val set
#----

preprocess_full = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize the input image to a given size (IMG_SIZE, IMG_SIZE)
        transforms.ToTensor(),                  # Convert to tensor (and also convert to [0, 1] tensors)
        transforms.Normalize(mean=mean_all, std=std_all) # Re-normalize with new mean and std
    ]
)

inputs_val = []

for record in tqdm(iterable=val_set, desc="Preprocessing Images"):
    image = record['img']
    label = record['label']
    
    # Convert from grayscale to RGB (3-colour channels)
    if image.mode == "L":
        image = image.convert("RGB")
        
    # preprocessing
    input_tensor = preprocess_full(image)
    label_tensor = torch.tensor(label)
    
    # append to inputs_train
    inputs_val.append([input_tensor, torch.tensor(label)])
    
################
## Dataloader ##
################

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(inputs_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(inputs_val, batch_size=BATCH_SIZE, shuffle=False)

###################################
## Building AlexNet-inspired CNN ##
###################################
'''
At each block, the images are downsampled by the max-pooling layer. 

Contrary, the number of channels from one layer to another increased from 3 to 64, to 192, ... to 256. 
=> As we learned before, deeper layers have larger receptive fields and generally detect more specific and complex features, 
such as ears, eyes, or even human faces and dogs. The chosen filter (or kernel) size is either or even human faces and dogs.

How will it increase from 3 channels to 64 channels?
    => The layer creates 64 separate filters (kernels), where each filter processes all 3 input channels together
    => Each has shape: 4x4x3 (kernel_size=4, and depth=3 to match input channels)
    => Filter 1 (4x4x3) convolves with RGB input → produces feature map 1
       Filter 2 (4x4x3) convolves with RGB input → produces feature map 2
       ...
       Filter 64 (4x4x3) convolves with RGB input → produces feature map 64
    => Stack all 64 feature maps together = 64 output channels

The kernel_size refers to the height and width of the sliding window (also called filter)
The chosen filter (or kernel) size is either 3 or 4. Example, kernel_size=4 => sliding window is 4x4
This is a common choice - having a smaller filter allows the network to better generalize. 

Padding is the process of adding a "border" of extra pixels (usually zeros) around the edges of your input image before the convolution operation begins.
Padding helps avoid shrinkage and loss of edge information.
=> Here, padding is 1 pixel on each layer.
'''

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1: conv -> relu -> max_pool
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Block 2: conv -> relu -> max_pool
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Block 3: conv -> relu
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Block 4: conv -> relu
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
                       
            # Block 5: conv -> relu -> max_pool
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Convolutional and pooling layers output 4D tensors (Batch, Channels, Height, Width)
            # However, Fully Connected layers expect 2D tensors (Batch, Features)
            # => must FLATTEN the 4D tensor into a 2D tensor
            nn.Flatten(),
                        
            # Block 6: drop_out -> fc_linear -> relu
            nn.Dropout(p=0.5),
            nn.LazyLinear(512), # Automatically figures out input size (1024 -> 512)
            # nn.Linear(1024, 512)
            nn.ReLU(),
            
            # Block 7: drop_out -> fc_linear -> relu
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            
            
            # Block 8: fc_linear -> final logits (output)
            nn.Linear(256, num_classes)
        )
        
    def forward(self, X):
        out = self.cnn(X)
        return out
'''
Block 1:
# Conv2d (kernel=4, padding=1): 32 + 2(1) - 4 + 1 = 31 → (batch, 64, 31, 31)
# MaxPool (kernel=3, stride=2): (31 - 3)/2 + 1 = 15 → (batch, 64, 15, 15)

Block 2:
# Conv2d (kernel=4, padding=1): 15 + 2(1) - 4 + 1 = 14 → (batch, 192, 14, 14)
# MaxPool (kernel=3, stride=2): (14 - 3)/2 + 1 = 6 → (batch, 192, 6, 6)

Block 3:
# Conv2d (kernel=3, padding=1): 6 + 2(1) - 3 + 1 = 6 → (batch, 384, 6, 6)

Block 4:
# Conv2d (kernel=3, padding=1): 6 + 2(1) - 3 + 1 = 6 → (batch, 256, 6, 6)

Block 5:
# Conv2d (kernel=3, padding=1): 6 + 2(1) - 3 + 1 = 6 → (batch, 256, 6, 6)
# MaxPool (kernel=3, stride=2): (6 - 3)/2 + 1 = 2 → (batch, 256, 2, 2) ✓


After Flatten:
256 channels x 2 height x 2 width = 1024 features
Final shape: (batch, 1024) → ready for nn.Linear(1024, 512)

#################

Formula for Conv2d output size:
    output_size = (input_size + 2xpadding - kernel_size) / stride + 1
'''

##########################
## model initialization ##
##########################

num_classes = len(set(train_set['label']))
print(num_classes) # 10

torch.manual_seed(42)
model = AlexNet(num_classes=num_classes).to(device)

# Initialize lazy layers
with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)  # Match your input size
    _ = model(dummy)

##################################
## Loss - Optimizer - Scheduler ##
##################################

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

################################
## Training - Validating loop ##
################################

epochs = 100

train_loss_list, val_loss_list = [], []

for epoch in tqdm(iterable=range(1, epochs+1), desc="Training"):
    # --- TRAINING ---
    _ = model.train() # Turn on training mode, enable gradient tracking
    for _, (images, labels) in enumerate(train_loader):
        # moves values to device
        images = images.to(device)
        labels = labels.to(device)
        
        # (Standard training steps: forward, loss, zero_grad, backward, step)
        preds = model(images).squeeze()
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # --- VALIDATION (Every epoch) ---
    _ = model.eval() # 1. Set model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode(): # 2. Turn off gradient tracking to save memory      
        for _, (images, labels) in enumerate(val_loader): # 3. Iterate through val_set
            # moves values to device
            images = images.to(device)
            labels = labels.to(device)
            
            # calculate predictions
            val_preds = model(images).squeeze()
            
            # Accumulate loss to get an average for the whole set
            val_loss += loss_fn(val_preds, labels).item()
            
            # Calculate accuracy
            total += labels.size(0)
            predicted = torch.argmax(val_preds, dim=1)
            correct += (predicted == labels).sum().item()
        
          
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = 100 * (correct / total)  
    
    train_loss_list.append(loss.item())
    val_loss_list.append(avg_val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    scheduler.step(avg_val_loss)
    
    if epoch % 10 == 0:
        print("+"*50)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {avg_val_acc:.2f}%")
        print(f"Current LR: {current_lr}")
'''
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 10
Train loss: 0.1208
Validation loss: 0.6865
Validation accuracy: 77.58%
Current LR: 0.0001
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 20
Train loss: 0.0749
Validation loss: 1.0052
Validation accuracy: 78.34%
Current LR: 0.0001
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 30
Train loss: 0.0125
Validation loss: 1.3402
Validation accuracy: 80.85%
Current LR: 5e-05
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 40
Train loss: 0.0004
Validation loss: 1.6344
Validation accuracy: 80.31%
Current LR: 2.5e-05
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 50
Train loss: 0.0000
Validation loss: 1.7851
Validation accuracy: 81.52%
Current LR: 1.25e-05
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 60
Train loss: 0.0000
Validation loss: 1.9175
Validation accuracy: 81.39%
Current LR: 6.25e-06
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 70
Train loss: 0.0000
Validation loss: 1.9965
Validation accuracy: 81.70%
Current LR: 3.125e-06
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 80
Train loss: 0.0000
...
Train loss: 0.0000
Validation loss: 2.2699
Validation accuracy: 81.77%
Current LR: 3.90625e-07
'''

#######################################
## Drawing Train and Val loss curves ##
#######################################

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

################################################
## Confusion matrix and Classification report ##
################################################

from sklearn.metrics import confusion_matrix, classification_report

# 1. Put model in eval mode
model.eval()
all_preds = []
all_labels = []

with torch.inference_mode():
    for images, labels in val_loader:
        images = images.to(device)
        
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 2. Generate the raw confusion matrix array
cm = confusion_matrix(all_labels, all_preds)

# 3. Print classification report
print('Classification report\n', classification_report(all_labels, all_preds))
# Classification report
#                precision    recall  f1-score   support

#            0       0.81      0.84      0.83      1000
#            1       0.89      0.91      0.90      1000
#            2       0.80      0.73      0.76      1000
#            3       0.66      0.68      0.67      1000
#            4       0.78      0.81      0.79      1000
#            5       0.74      0.73      0.73      1000
#            6       0.87      0.87      0.87      1000
#            7       0.87      0.86      0.86      1000
#            8       0.89      0.88      0.89      1000
#            9       0.88      0.87      0.87      1000

#     accuracy                           0.82     10000
#    macro avg       0.82      0.82      0.82     10000
# weighted avg       0.82      0.82      0.82     10000
        
import plotly.express as px

# Replace these with your actual class names if you have them 
# e.g., ['airplane', 'automobile', 'bird', ...]
class_names = [str(i) for i in range(len(cm))] 

fig = px.imshow(
    cm,
    text_auto=True,               # Shows the numbers inside the squares
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=class_names,
    y=class_names,
    color_continuous_scale='Blues',
    title='AlexNet Confusion Matrix (Interactive)'
)

fig.update_layout(
    xaxis_title='Predicted Label',
    yaxis_title='True Label',
    width=700,
    height=700
)

fig.show()

########################
## Saving whole model ##
########################

# import os
# os.chdir('../')
# os.getcwd() # '/home/longdpt/Documents/Long_AISDL/DeepLearning_PyTorch'

from pathlib import Path

MODEL_PATH = Path("04_CNN").joinpath("save")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# PyTorch model can be saved in .pth or .pt format
PARAMS_NAME = "AlexNet_model.pth"

# Save the model (use model.state_dict() to save only the parameters)
torch.save(obj=model, f=MODEL_PATH.joinpath(PARAMS_NAME))

###############
## Inference ##
###############

model_loaded = torch.load(MODEL_PATH.joinpath(PARAMS_NAME), weights_only=False)

_ = model_loaded.eval().to(device)

inference_inputs = []
for image in val_set['img'][:10]: # Use 10 images only for inference demonstration
    tensor = preprocess(image)
    inference_inputs.append(tensor.to(device))

print(inference_inputs[0].shape)
# torch.Size([3, 32, 32])

print(len(inference_inputs))
# 10

inference_inputs = torch.stack(inference_inputs)
print(inference_inputs.shape)
# torch.Size([10, 3, 32, 32])

inference_outputs = model_loaded(inference_inputs)
print(inference_outputs)
# tensor([[-1.8792e+01, -1.7055e+01, -1.2150e+01,  2.0642e+01, -1.5233e+01,
#          -1.3342e+00, -7.5604e+00, -1.5334e+01, -7.7339e+00, -1.3140e+01],
#         [ 7.4274e-01,  1.1191e+01, -1.3246e+01, -1.0368e+01, -1.7118e+01,
#          -1.7910e+01, -1.0079e+01, -2.5981e+01,  1.5994e+01, -2.9743e-01],
#         [-3.5441e+00,  5.6458e+00, -1.2926e+01, -7.3448e+00, -2.1433e+01,
#          -1.9943e+01, -4.3886e+00, -2.9696e+01,  1.9463e+01,  9.7131e-01],
#         [ 2.3815e+01, -1.4268e+01, -4.0832e+00, -1.0737e+01, -3.7516e+00,
#          -3.0416e+01, -2.4778e+01, -1.1626e+01, -8.9818e-01, -8.9371e+00],
#         [-2.8288e+01, -9.5688e+00, -5.1894e+00, -9.0462e+00, -2.9350e+00,
#          -1.9445e+01,  3.0575e+01, -3.3225e+01, -1.7907e+01, -9.8912e+00],
#         [-2.1310e+01, -8.5885e+00,  2.9811e+00, -9.0718e-01, -1.4188e+01,
#           9.4866e-01,  1.1958e+01, -1.7442e+01, -1.4355e+01, -9.9505e+00],
#         [-1.0576e+01,  1.6167e+01, -1.6758e+01, -8.1439e+00, -2.6167e+01,
#          -9.1709e+00, -3.1447e+00, -1.4724e+01, -4.7862e+00,  1.2284e+01],
#         [-1.2589e+01, -3.1722e+00,  1.9408e-03, -2.4551e+00, -9.9912e+00,
#          -6.7302e+00,  1.1760e+01, -1.2326e+01, -7.3236e+00, -4.3695e-01],
#         [-1.4554e+01, -2.5642e+01, -1.3260e+01,  2.1597e+01, -8.6114e+00,
#          -1.0002e+01, -9.3784e+00, -9.7159e+00, -1.3780e+01, -1.1472e+01],
#         [-6.2667e+00,  1.8855e+01, -1.2764e+01, -1.1958e+01, -2.0231e+01,
#          -7.4204e+00, -3.9312e+00, -1.8321e+01,  8.1524e-01,  5.5274e+00]],
#        device='cuda:0', grad_fn=<AddmmBackward0>)

predicted = torch.argmax(inference_outputs, dim=1)
print(predicted)
# tensor([3, 8, 8, 0, 6, 6, 1, 6, 3, 1], device='cuda:0')

print(val_set.features['label'].names)
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#----------
## Visualize predicted and image
#----------

import matplotlib.pyplot as plt

for i, image in enumerate(val_set['img'][:10]):
    print("="*50)
    print(val_set.features['label'].names[predicted[i]])
    plt.imshow(image)
    plt.show()