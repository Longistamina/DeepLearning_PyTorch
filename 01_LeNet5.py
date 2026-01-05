'''
LeNet-5 architecture
+---------+-----------------+-------------------------------+------------------+
| Layer   | Type            | Configuration                 | Output Size      |
+---------+-----------------+-------------------------------+------------------+
| Input   | Image           | 32 x 32 x 1 (Grayscale)       | 32 x 32 x 1      |
| C1      | Convolution     | 6 filters (5x5), Stride 1     | 28 x 28 x 6      |
| S2      | Avg Pooling     | 2x2 window, Stride 2          | 14 x 14 x 6      |
| C3      | Convolution     | 16 filters (5x5), Stride 1    | 10 x 10 x 16     |
| S4      | Avg Pooling     | 2x2 window, Stride 2          | 5 x 5 x 16       |
| C5      | Convolution     | 120 filters (5x5), Stride 1   | 1 x 1 x 120      |
| F6      | Fully Connected | 84 Neurons                    | 84               |
| Output  | Fully Connected | 10 Neurons (Softmax)          | 10               |
+---------+-----------------+-------------------------------+------------------+

LeNet-5 was designed for handwritten digit recognition (MNIST) and pioneered the use of convolutional layers. 

It originally used tanh activation functions and average pooling instead of ReLU and max pooling found in modern architectures. 

The C5 layer effectively flattens the feature maps to 120 values, 
transitioning from convolutional to fully connected layers.
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
Import MNIST dataset from HuggingFace
Run this first in terminal: pip install datasets
'''

from datasets import load_dataset

train_set = load_dataset(
    'mnist',
    split='train',
    verification_mode='basic_checks'
)
print(train_set)
# Dataset({
#     features: ['image', 'label'],
#     num_rows: 60000
# })

val_set = load_dataset(
    'mnist',
    split='test',
    verification_mode='basic_checks'
)
print(val_set)
# Dataset({
#     features: ['image', 'label'],
#     num_rows: 10000
# })

# View an image
train_set[0]['image']

print(train_set[0]['image'])
# <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28>

#########################
## Image preprocessing ##
#########################

IMG_SIZE = 32
'''
LeNet-5 is designed to accept 32x32 grayscale images
=> Must resize MNIST images from 28x28 to 32x32
'''

#----
## Build preprocess transforms
#----

preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize from 28x28 to 32x32
        transforms.ToTensor()                      # Convert to tensor (and also convert to [0, 1] tensors)
    ]
)

#----
## Apply preprocess to training set
#----

inputs_train = []

for record in tqdm(iterable=train_set, desc="Preprocessing Images"):
    image = record['image']
    label = record['label']
    
    # preprocessing
    input_tensor = preprocess(image)
    label_tensor = torch.tensor(label)
    
    # append to inputs_train
    inputs_train.append([input_tensor, label_tensor])

#----
## Re-normalize the pixel values for train set
#----
'''
Since transforms.ToTensor() normalizes all into [0, 1],
we need to modify this normalization to fit MNIST dataset.

Calculate the mean and std for grayscale images (single channel)
then use transforms.Normalize(mean=, std=) with this calculated mean and std.
'''

import numpy as np

# Choosing a random sample to calculate mean and std (512 images)
np.random.seed(0)
idx = np.random.randint(0, len(inputs_train), 512)

# Concatenate this subset of images into a new tensor
tensor_placeholder = torch.concat([inputs_train[i][0] for i in idx], axis=1)
print(tensor_placeholder.shape)
# torch.Size([1, 16384, 32])
'''
we concatenate 512 images of size (1x32x32) (Channel*Height*Width) along the Height channel
=> (1x16384x32), 16384=32*512
'''

# Calculate the mean and std across all images
mean_all = torch.mean(tensor_placeholder, dim=(1, 2))
std_all = torch.std(tensor_placeholder, dim=(1, 2))

print(mean_all)  # tensor([0.1307]) - typical for MNIST
print(std_all)   # tensor([0.3081]) - typical for MNIST

#### RE-NORMALIZE ###

preprocess = transforms.Compose([transforms.Normalize(mean=mean_all, std=std_all)])

for idx in tqdm(range(len(inputs_train))):
    input_tensor = preprocess(inputs_train[idx][0])
    inputs_train[idx][0] = input_tensor  # replace with re-normalized tensor

#----
## Re-normalize the pixel values for val set
#----

preprocess_full = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_all, std=std_all)
    ]
)

inputs_val = []

for record in tqdm(iterable=val_set, desc="Preprocessing Images"):
    image = record['image']
    label = record['label']
    
    # preprocessing
    input_tensor = preprocess_full(image)
    label_tensor = torch.tensor(label)
    
    # append to inputs_val
    inputs_val.append([input_tensor, label_tensor])

################
## Dataloader ##
################

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(inputs_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(inputs_val, batch_size=BATCH_SIZE, shuffle=False)

############################
## Building LeNet-5 Model ##
############################
'''
LeNet-5 architecture follows a simple pattern:
Conv → AvgPool → Conv → AvgPool → Conv → FC → FC → Output

The architecture uses:
- 5x5 convolutional kernels
- Average pooling (2x2 with stride 2)
- Tanh activation (we use ReLU for modern implementation)
- Gradual increase in channels: 1 → 6 → 16 → 120

How spatial dimensions change:
Input 32x32x1 
→ C1: Conv(6, 5x5) → 28x28x6
→ S2: AvgPool(2x2) → 14x14x6
→ C3: Conv(16, 5x5) → 10x10x16
→ S4: AvgPool(2x2) → 5x5x16
→ C5: Conv(120, 5x5) → 1x1x120 (this effectively flattens to 120)
→ F6: Linear(120, 84)
→ Output: Linear(84, 10)
'''

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            # C1: Convolutional Layer
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),  # Modern implementation uses ReLU instead of tanh
            
            # S2: Average Pooling Layer
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # C3: Convolutional Layer
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            
            # S4: Average Pooling Layer
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # C5: Convolutional Layer (acts as fully connected)
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU(),
            
            # Flatten the output for fully connected layers
            nn.Flatten(),
            
            # F6: Fully Connected Layer
            nn.LazyLinear(84), # LazyLinear() will automatically infer the input_size for us (120 -> 84) 
            nn.ReLU(),
            
            # Output Layer
            nn.Linear(84, num_classes)
        )
        
    def forward(self, X):
        out = self.cnn(X)
        return out

'''
Spatial dimension calculations:

C1: Conv2d (kernel=5, stride=1, padding=0):
    output = (32 - 5)/1 + 1 = 28 → (batch, 6, 28, 28)

S2: AvgPool (kernel=2, stride=2):
    output = (28 - 2)/2 + 1 = 14 → (batch, 6, 14, 14)

C3: Conv2d (kernel=5, stride=1, padding=0):
    output = (14 - 5)/1 + 1 = 10 → (batch, 16, 10, 10)

S4: AvgPool (kernel=2, stride=2):
    output = (10 - 2)/2 + 1 = 5 → (batch, 16, 5, 5)

C5: Conv2d (kernel=5, stride=1, padding=0):
    output = (5 - 5)/1 + 1 = 1 → (batch, 120, 1, 1)

After Flatten:
    120 channels × 1 height × 1 width = 120 features
    Final shape: (batch, 120) → ready for nn.Linear(120, 84)

#################

Formula for Conv2d output size:
    output_size = (input_size - kernel_size + 2×padding) / stride + 1
'''

##########################
## Model initialization ##
##########################

num_classes = 10  # MNIST has 10 digit classes (0-9)
print(num_classes)

torch.manual_seed(42)
model = LeNet5(num_classes=num_classes).to(device)

# Initialize lazy layers
with torch.no_grad():
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)  # Match your input size
    _ = model(dummy)

##################################
## Loss - Optimizer - Scheduler ##
##################################

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

################################
## Training - Validating loop ##
################################

epochs = 20  # LeNet-5 on MNIST converges faster than AlexNet on CIFAR-10

train_loss_list, val_loss_list = [], []

for epoch in tqdm(iterable=range(1, epochs+1), desc="Training"):
    # --- TRAINING ---
    _ = model.train()
    for _, (images, labels) in enumerate(train_loader):
        # moves values to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Standard training steps
        preds = model(images).squeeze()
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # --- VALIDATION (Every epoch) ---
    _ = model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for _, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            val_preds = model(images).squeeze()
            
            # Accumulate loss
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
    
    if epoch % 5 == 0:
        print("+"*50)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {avg_val_acc:.2f}%")
        print(f"Current LR: {current_lr}")

#######################################
## Drawing Train and Val loss curves ##
#######################################

def plot_train_val_loss_curves(epochs, train_loss_list, val_loss_list):
    import plotly.graph_objects as pgo
    import numpy as np
    
    epoch_axis = np.arange(1, epochs + 1, 1)

    fig = pgo.Figure()

    # Add Training Loss
    fig.add_trace(pgo.Scatter(
        x=epoch_axis,
        y=train_loss_list,
        mode='lines+markers',
        name='Train Loss',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    # Add Validation Loss
    fig.add_trace(pgo.Scatter(
        x=epoch_axis,
        y=val_loss_list,
        mode='lines+markers',
        name='Val Loss',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8, symbol='square')
    ))

    fig.update_layout(
        title='<b>LeNet-5 Training Progress on MNIST</b>',
        xaxis_title='Epoch',
        yaxis_title='Loss Value',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    fig.show()

plot_train_val_loss_curves(epochs, train_loss_list, val_loss_list)

################################################
## Confusion matrix and Classification report ##
################################################

from sklearn.metrics import confusion_matrix, classification_report

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

# Generate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Print classification report
print('Classification report\n', classification_report(all_labels, all_preds))

import plotly.express as px

# MNIST digit classes
class_names = [str(i) for i in range(10)]

fig = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=class_names,
    y=class_names,
    color_continuous_scale='Blues',
    title='LeNet-5 Confusion Matrix on MNIST (Interactive)'
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

from pathlib import Path

MODEL_PATH = Path("04_CNN").joinpath("save")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

PARAMS_NAME = "LeNet5_model.pth"

# Save the model
torch.save(obj=model, f=MODEL_PATH.joinpath(PARAMS_NAME))

###############
## Inference ##
###############

model_loaded = torch.load(MODEL_PATH.joinpath(PARAMS_NAME), weights_only=False)

_ = model_loaded.eval().to(device)

inference_inputs = []
for image in val_set['image'][:10]:  # Use 10 images for inference demonstration
    tensor = preprocess_full(image)
    inference_inputs.append(tensor.to(device))

print(inference_inputs[0].shape)
# torch.Size([1, 32, 32])

print(len(inference_inputs))
# 10

inference_inputs = torch.stack(inference_inputs)
print(inference_inputs.shape)
# torch.Size([10, 1, 32, 32])

inference_outputs = model_loaded(inference_inputs)
print(inference_outputs)

predicted = torch.argmax(inference_outputs, dim=1)
print(predicted)

#----------
## Visualize predicted and image
#----------

import matplotlib.pyplot as plt

for i, image in enumerate(val_set['image'][:10]):
    print("="*50)
    print(f"Predicted digit: {predicted[i].item()}")
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted[i].item()}")
    plt.axis('off')
    plt.show()