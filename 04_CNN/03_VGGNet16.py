'''
VGG16 Architecture
+---------+-----------------+-------------------------------+------------------+
| Layer   | Type            | Configuration                 | Output Size      |
+---------+-----------------+-------------------------------+------------------+
| Input   | Image           | 224 x 224 x 3 (RGB)           | 224 x 224 x 3    |
|         |                 |                               |                  |
| Conv1-1 | Convolution     | 64 filters (3x3), Stride 1    | 224 x 224 x 64   |
| Conv1-2 | Convolution     | 64 filters (3x3), Stride 1    | 224 x 224 x 64   |
| Pool1   | Max Pooling     | 2x2 window, Stride 2          | 112 x 112 x 64   |
|         |                 |                               |                  |
| Conv2-1 | Convolution     | 128 filters (3x3), Stride 1   | 112 x 112 x 128  |
| Conv2-2 | Convolution     | 128 filters (3x3), Stride 1   | 112 x 112 x 128  |
| Pool2   | Max Pooling     | 2x2 window, Stride 2          | 56 x 56 x 128    |
|         |                 |                               |                  |
| Conv3-1 | Convolution     | 256 filters (3x3), Stride 1   | 56 x 56 x 256    |
| Conv3-2 | Convolution     | 256 filters (3x3), Stride 1   | 56 x 56 x 256    |
| Conv3-3 | Convolution     | 256 filters (3x3), Stride 1   | 56 x 56 x 256    |
| Pool3   | Max Pooling     | 2x2 window, Stride 2          | 28 x 28 x 256    |
|         |                 |                               |                  |
| Conv4-1 | Convolution     | 512 filters (3x3), Stride 1   | 28 x 28 x 512    |
| Conv4-2 | Convolution     | 512 filters (3x3), Stride 1   | 28 x 28 x 512    |
| Conv4-3 | Convolution     | 512 filters (3x3), Stride 1   | 28 x 28 x 512    |
| Pool4   | Max Pooling     | 2x2 window, Stride 2          | 14 x 14 x 512    |
|         |                 |                               |                  |
| Conv5-1 | Convolution     | 512 filters (3x3), Stride 1   | 14 x 14 x 512    |
| Conv5-2 | Convolution     | 512 filters (3x3), Stride 1   | 14 x 14 x 512    |
| Conv5-3 | Convolution     | 512 filters (3x3), Stride 1   | 14 x 14 x 512    |
| Pool5   | Max Pooling     | 2x2 window, Stride 2          | 7 x 7 x 512      |
|         |                 |                               |                  |
| Flatten | Flatten         | -                             | 25088            |
| FC6     | Fully Connected | 4096 Neurons + ReLU + Dropout | 4096             |
| FC7     | Fully Connected | 4096 Neurons + ReLU + Dropout | 4096             |
| FC8     | Fully Connected | 1000 Neurons (Softmax)        | 1000             |
+---------+-----------------+-------------------------------+------------------+

Key Characteristics of VGGNet:

- Uses very small 3x3 convolutional filters throughout the entire network
- All conv layers use stride 1 and padding 1 (to preserve spatial dimensions)
- Max pooling with 2x2 windows and stride 2 is used to downsample
- ReLU activation functions after every convolutional layer
- Dropout (0.5) is applied after the first two fully connected layers
- Total parameters: ~138 million (VGG16)

VGGNet showed that network depth is critical for performance. The uniform use of 
3x3 filters (instead of larger filters) reduces parameters while increasing depth.

Two stacked 3x3 conv layers have an effective receptive field of 5x5, and three 
stacked 3x3 layers have a 7x7 receptive field, but with fewer parameters.

###########################################################################################

VGG19 Architecture
+---------+-----------------+-------------------------------+------------------+
| Layer   | Type            | Configuration                 | Output Size      |
+---------+-----------------+-------------------------------+------------------+
| Input   | Image           | 224 x 224 x 3 (RGB)           | 224 x 224 x 3    |
|         |                 |                               |                  |
| Conv1-1 | Convolution     | 64 filters (3x3), Stride 1    | 224 x 224 x 64   |
| Conv1-2 | Convolution     | 64 filters (3x3), Stride 1    | 224 x 224 x 64   |
| Pool1   | Max Pooling     | 2x2 window, Stride 2          | 112 x 112 x 64   |
|         |                 |                               |                  |
| Conv2-1 | Convolution     | 128 filters (3x3), Stride 1   | 112 x 112 x 128  |
| Conv2-2 | Convolution     | 128 filters (3x3), Stride 1   | 112 x 112 x 128  |
| Pool2   | Max Pooling     | 2x2 window, Stride 2          | 56 x 56 x 128    |
|         |                 |                               |                  |
| Conv3-1 | Convolution     | 256 filters (3x3), Stride 1   | 56 x 56 x 256    |
| Conv3-2 | Convolution     | 256 filters (3x3), Stride 1   | 56 x 56 x 256    |
| Conv3-3 | Convolution     | 256 filters (3x3), Stride 1   | 56 x 56 x 256    |
| Conv3-4 | Convolution     | 256 filters (3x3), Stride 1   | 56 x 56 x 256    |
| Pool3   | Max Pooling     | 2x2 window, Stride 2          | 28 x 28 x 256    |
|         |                 |                               |                  |
| Conv4-1 | Convolution     | 512 filters (3x3), Stride 1   | 28 x 28 x 512    |
| Conv4-2 | Convolution     | 512 filters (3x3), Stride 1   | 28 x 28 x 512    |
| Conv4-3 | Convolution     | 512 filters (3x3), Stride 1   | 28 x 28 x 512    |
| Conv4-4 | Convolution     | 512 filters (3x3), Stride 1   | 28 x 28 x 512    |
| Pool4   | Max Pooling     | 2x2 window, Stride 2          | 14 x 14 x 512    |
|         |                 |                               |                  |
| Conv5-1 | Convolution     | 512 filters (3x3), Stride 1   | 14 x 14 x 512    |
| Conv5-2 | Convolution     | 512 filters (3x3), Stride 1   | 14 x 14 x 512    |
| Conv5-3 | Convolution     | 512 filters (3x3), Stride 1   | 14 x 14 x 512    |
| Conv5-4 | Convolution     | 512 filters (3x3), Stride 1   | 14 x 14 x 512    |
| Pool5   | Max Pooling     | 2x2 window, Stride 2          | 7 x 7 x 512      |
|         |                 |                               |                  |
| Flatten | Flatten         | -                             | 25088            |
| FC6     | Fully Connected | 4096 Neurons + ReLU + Dropout | 4096             |
| FC7     | Fully Connected | 4096 Neurons + ReLU + Dropout | 4096             |
| FC8     | Fully Connected | 1000 Neurons (Softmax)        | 1000             |
+---------+-----------------+-------------------------------+------------------+

Key Differences between VGG16 and VGG19:

VGG16: 13 conv layers + 3 FC layers = 16 weight layers
VGG19: 16 conv layers + 3 FC layers = 19 weight layers

The additional 3 convolutional layers are:
- Conv3-4: One extra layer in the 3rd block
- Conv4-4: One extra layer in the 4th block  
- Conv5-4: One extra layer in the 5th block

Total parameters: ~144 million (VGG19) vs ~138 million (VGG16)

VGG19 is slightly deeper and has more parameters, but in practice, VGG16 and VGG19 
achieve very similar performance on ImageNet. The marginal improvement from the 
extra layers is minimal, making VGG16 often preferred for its efficiency.
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
Import Animals-10 dataset with more than 23k images from HuggingFace
Run this first in terminal: pip install datasets
'''

from datasets import load_dataset

# Load dataset from huggingface
animals_10 = load_dataset(path="Rapidata/Animals-10", split="train")

# Print the structure
print(animals_10)
# Dataset({
#     features: ['image', 'label'],
#     num_rows: 23554
# })

# Show an image
animals_10['image'][0]

###################
## Preprocessing ##
###################

IMG_SIZE = 224

#--------
## Initial preprocessing
#--------

preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ]
)

inputs_temp = []

for record in tqdm(iterable=animals_10, desc="Preprocessing Images"):
    image = record['image']
    label = record['label']
    
    # Convert from grayscale to RGB (3-colour channels)
    if image.mode == "L":
        image = image.convert("RGB")
        
    # preprocessing
    input_tensor = preprocess(image)
    label_tensor = torch.tensor(label)
    
    # append to inputs_temp
    inputs_temp.append([input_tensor, torch.tensor(label)])
    
#----
## Calculate mean and std after inital preprocessing
#----

import numpy as np

np.random.seed(0)
idx = np.random.randint(0, len(inputs_temp), 10000)

tensor_placeholder = torch.concat([inputs_temp[i][0] for i in idx], axis=1)
print(tensor_placeholder.shape)
# torch.Size([3, 2240000, 224])

mean_all = torch.mean(tensor_placeholder, dim=(1, 2))
std_all = torch.std(tensor_placeholder, dim=(1, 2))

print(mean_all) # tensor([0.5210, 0.5053, 0.4184])
print(std_all)  # tensor([0.2645, 0.2610, 0.2787])

#----
## Re-normalize with calculated mean and std
#----

preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_all, std=std_all)
    ]
)

inputs_final = []

for record in tqdm(iterable=animals_10, desc="Preprocessing Images"):
    image = record['image']
    label = record['label']
    
    # Convert from grayscale to RGB (3-colour channels)
    if image.mode == "L":
        image = image.convert("RGB")
        
    # preprocessing
    input_tensor = preprocess(image)
    label_tensor = torch.tensor(label)
    
    # append to inputs_temp
    inputs_final.append([input_tensor, torch.tensor(label)])
    

del inputs_temp

#######################
## Dataset splitting ##
#######################

#-----
## Takes images and labels out
#-----

images, labels = zip(*inputs_final)
'''
Work like this
images = [item[0] for item in data_list]
labels = [item[1] for item in data_list]
'''

images = torch.stack(images, dim=0) # X
labels = torch.stack(labels, dim=0) # y

print(images.shape) # torch.Size([23554, 3, 224, 224])
print(labels.shape) # torch.Size([23554])

#-----
## Train - Val - Test split
#-----

train_len = int(0.7 * len(images)) # MUST be INTEGER
val_len = int(0.15 * len(images))
test_len = len(images) - (train_len + val_len)

print(train_len, val_len, test_len)
# 16487 3533 3534

from torch.utils.data import DataLoader, TensorDataset, random_split

full_dataset = TensorDataset(images, labels)

train_split, val_split, test_split = random_split(dataset=full_dataset, lengths=[train_len, val_len, test_len])

BATCH_SIZE = 32

train_set = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True) # shuffle=True to reshuffle the data after every epoch
val_set = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False)
test_set = DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=False)

#######################
## VGGNet16 building ##
#######################

from torch import nn

class VGGNet16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1: Conv1-1 -> Conv1-2 -> Pool1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Conv2-1 -> Conv2-2 -> Pool2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: Conv3-1 -> Conv3-2 -> Conv3-3 -> Pool3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: Conv4-1 -> Conv4-2 -> Conv4-3 -> Pool4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: Conv4-1 -> Conv4-2 -> Conv4-3 -> Pool4
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            
            # Fully connected layers (7*7*512 = 25088 input features)
            nn.Linear(25088, 4096),  # ✅ Better to be explicit
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, num_classes)                                     
        )
        
    def forward(self, X):
        out = self.cnn(X)
        out = self.fc(out)
        return out
    
##########################
## model initialization ##
##########################

num_classes = len(set(labels.cpu().numpy()))
print(num_classes) # 10

torch.manual_seed(42)
model = VGGNet16(num_classes=num_classes).to(device)

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
    for _, (images, labels) in enumerate(train_set):
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
        for _, (images, labels) in enumerate(val_set): # 3. Iterate through val_set
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
        
          
    
    avg_val_loss = val_loss / len(val_set)
    avg_val_acc = 100 * (correct / total)  
    
    train_loss_list.append(loss.item())
    val_loss_list.append(avg_val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    scheduler.step(avg_val_loss)
    
    if (epoch % 10 == 0) or (epoch == 1):
        print("+"*50)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {avg_val_acc:.2f}%")
        print(f"Current LR: {current_lr}")
'''
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 1
Train loss: 1.9740
Validation loss: 2.1666
Validation accuracy: 23.12%
Current LR: 0.0001
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 10
Train loss: 0.3324
Validation loss: 1.4020
Validation accuracy: 64.65%
Current LR: 0.0001
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 20
Train loss: 0.0008
Validation loss: 2.5002
Validation accuracy: 65.16%
Current LR: 5e-05
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 30
Train loss: 0.0002
Validation loss: 2.7281
Validation accuracy: 65.30%
Current LR: 2.5e-05
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 40
Train loss: 0.0000
Validation loss: 3.4621
Validation accuracy: 64.93%
Current LR: 1.25e-05
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 50
Train loss: 0.0000
Validation loss: 3.6223
Validation accuracy: 64.87%
Current LR: 1.25e-05
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 60
Train loss: 0.0000
Validation loss: 3.9608
Validation accuracy: 65.38%
Current LR: 6.25e-06
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 70
Train loss: 0.0000
Validation loss: 4.0650
Validation accuracy: 65.50%
Current LR: 3.125e-06
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 80
Train loss: 0.0000
Validation loss: 4.1697
Validation accuracy: 65.67%
Current LR: 1.5625e-06
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 90
Train loss: 0.0000
Validation loss: 4.1366
Validation accuracy: 65.86%
Current LR: 7.8125e-07
++++++++++++++++++++++++++++++++++++++++++++++++++
Epoch: 100
Train loss: 0.0000
Validation loss: 4.2042
Validation accuracy: 65.92%
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

#############
## Testing ##
#############

class_names = animals_10.features['label'].names

from sklearn.metrics import confusion_matrix, classification_report

# 1. Put model in eval mode
_ = model.eval()
all_preds = []
all_labels = []

with torch.inference_mode():
    for images, labels in test_set:
        images = images.to(device)
        
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 2. Generate the raw confusion matrix array
cm = confusion_matrix(all_labels, all_preds)

# 3. Print classification report
print('Classification report\n', classification_report(y_true=all_labels, y_pred=all_preds, target_names=class_names))
# Classification report
#                precision    recall  f1-score   support

#    Butterfly       0.69      0.62      0.65       232
#          Cat       0.52      0.30      0.38       181
#      Chicken       0.77      0.75      0.76       443
#          Cow       0.57      0.54      0.55       257
#          Dog       0.65      0.75      0.69       699
#     Elephant       0.65      0.54      0.59       164
#        Horse       0.69      0.70      0.70       404
#        Sheep       0.59      0.50      0.54       191
#       Spider       0.76      0.86      0.81       659
#     Squirrel       0.59      0.56      0.57       304

#     accuracy                           0.68      3534
#    macro avg       0.65      0.61      0.62      3534
# weighted avg       0.67      0.68      0.67      3534

#----------
## Visualize confusion matrix
#----------
        
import plotly.express as px

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

#----------
## Visualize some random predicted images (test_set)
#----------

np.random.seed(42)
idx_random = np.random.randint(0, test_len+1, 10)

import matplotlib.pyplot as plt

for i in idx_random:
    print("="*50)
    idx_original = test_split.indices[i]
    image = animals_10['image'][idx_original]
    predict_class = animals_10.features['label'].names[all_preds[i]]
    plt.title(f"Predicted: {predict_class}")
    plt.imshow(image)
    plt.show()
    
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
PARAMS_NAME = "VGGNet16_model.pth"

# Save the model (use model.state_dict() to save only the parameters)
torch.save(obj=model, f=MODEL_PATH.joinpath(PARAMS_NAME))

'''
Why this VGGNet16 fails?
=> Dataset too small (only ~24k images), while original VGGNet is trained with more than 1 million images
=> Should get the pretrained VGGNet and fine-tune it
'''