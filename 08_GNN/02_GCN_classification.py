#########################
## Dataset preparation ##
#########################

from torch_geometric.datasets import  Planetoid
from torch_geometric.transforms import NormalizeFeatures

root_path = '/home/longdpt/Documents/Long_AISDL/DeepLearning_PyTorch/05_GNN/data'

#---------
## Load the dataset
#---------

cora = Planetoid(root=root_path, name="Cora", transform=NormalizeFeatures())

'''
Planetoid is not a single dataset, but rather a collection of three citation network datasets, 
commonly used for benchmarking Graph Neural Networks (GNNs).

########################################

The Cora dataset consists of 2708 scientific publications classified into one of seven classes. 
Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. 
The dictionary consists of 1433 unique words.

Nodes = Publications (Papers, Books ...)
Edges = Citations
Node Features = word vectors
7 Labels = Pubilcation type e.g. Neural_Networks, Rule_Learning, Reinforcement_Learning, Probabilistic_Methods...
We normalize the features using torch geometric's transform functions.
'''

#---------
## Dataset properties
#---------

print(cora) # Cora()
print("number of graphs:\t\t", len(cora))                      # 1 (has only one giant graph)
print("number of classes:\t\t", cora.num_classes)              # 7 (has 7 different features)
print("number of node features:\t", cora.num_node_features)    # 1433 (each node has 1433 features, a bag of 1433 words)
print("number of edge features:\t", cora.num_edge_features)    # 0

#---------
## dataset._data
#---------

print(cora._data)
print("\n")
# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
'''
2708 nodes (papers)
10,556 edges (citations)
1433 features (a bag of 1433 words)
7 classes (research topics)
'''

print("edge_index:\t\t", cora._data.edge_index.shape)
print(cora._data.edge_index)
print("\n")
# edge_index:		 torch.Size([2, 10556])              10556 edges (citation relationships)
# tensor([[ 633, 1862, 2582,  ...,  598, 1473, 2706],    SOURCE NODES
#         [   0,    0,    0,  ..., 2707, 2707, 2707]])   TARGET NODES
# Example: 633 -> 0, 1862 -> 0

print("train_mask:\t\t", cora._data.train_mask.shape)
print(cora._data.train_mask)
print("\n")
# train_mask:		 torch.Size([2708])
# tensor([ True,  True,  True,  ..., False, False, False])
# True: this is from the training set
# False: this is NOT in the training set

print("X:\t\t", cora._data.x.shape)
print(cora._data.x)
print("\n")
# X:		 torch.Size([2708, 1433])
# tensor([[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]])
# Each row represents one node
# Each column represents one feature of a node

print("y:\t\t", cora._data.y.shape)
print(cora._data.y)
print("\n")
# y:		 torch.Size([2708])
# tensor([3, 4, 4,  ..., 3, 3, 3])
# The output label of each node (in this example, we have 7 different classes)

#----
## get the data
#----

data = cora[0] # Since we have only one dataset, use [0] to get it out

################
## Simple GNN ##
################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv # GraphSAGE Convolution layer from PyTorch Geometric

'''
What is SAGEConv?
GraphSAGE (SAmple and aggreGatE) is a type of graph convolution that:
# Samples neighbors of each node
# Aggregates their features
# Combines with the node's own features

What happens inside SAGEConv:
For each node i:
# Gather features from neighbors: {x_j : j ∈ Neighbors(i)}
# Aggregate: h_neighbors = max(x_j for all neighbors j) (element-wise max)
# Combine: h_i = W * concat([x_i, h_neighbors])

############ Example #############

x_i = [1, 0, 1]

neighbor_1 = [1, 2, 1]
neighbor_2 = [3, 4, 2]
neighbor_3 = [8, 1, 0]

aggregated = [max(1, 3, 8),   # position 0
              max(2, 4, 1),   # position 1
              max(1, 2, 0)]   # position 2

aggregated = [8, 4, 2]  ✅

=> concat([x_i, aggregated]) = [1, 0, 1, 8, 4, 2]
```

---

## Visual Representation
```
Position:       0   1   2
              ┌───┬───┬───┐
Neighbor 1:   │ 1 │ 2 │ 1 │
              ├───┼───┼───┤
Neighbor 2:   │ 3 │ 4 │ 2 │
              ├───┼───┼───┤
Neighbor 3:   │ 8 │ 1 │ 0 │
              ├───┼───┼───┤
              │ ↓ │ ↓ │ ↓ │
              ├───┼───┼───┤
Max:          │ 8 │ 4 │ 2 │
              └───┴───┴───┘
              
Node's own features:     [1, 0, 1]
                             +
Aggregated neighbors:    [8, 4, 2]
                              ↓
Concatenated:            [1, 0, 1, 8, 4, 2]
                         └─────┘ └───────┘
                          self   neighbors
'''

#----
## build model
#----

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()

        torch.manual_seed(42)
        
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training) # Only apply dropout while training, to avoid losing information while inferencing
        
        # Second layer
        x = self.conv2(x, edge_index)

        '''SAGEConv already does linear transformation so we don't need to add a linear layer'''
        return x # Return raw logits for CrossEntropyLoss

# Usage
model = GNN(
    in_channels=cora.num_features,  # 1433
    hidden_channels=16,  # typical choice for Cora
    out_channels=cora.num_classes,  # 7
    dropout=0.5
)

print(model)
# GNN(
#   (conv1): SAGEConv(1433, 16, aggr=mean)
#   (conv2): SAGEConv(16, 7, aggr=mean)
# )

################################
## Optimizer, Loss, Scheduler ##
################################

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model, data = model.to(device), data.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
'''
# With weight_decay:
loss = original_loss + (weight_decay / 2) * sum(param^2 for all params)
=> a regularization technique that prevents overfitting by penalizing large weights.
'''

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)

##############
## Training ##
##############

#-----
## Define functions
#-----

def train(model, data, optimizer, loss_fn):
    """Single training epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass (here we use all data as input because all nodes have node features)
    out = model(data.x, data.edge_index)
    
    # Calculate loss only on training nodes
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, data, mask):
    """Evaluate model on a specific mask (train/val/test)"""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].max(1)[1]  # Get predicted class
        labels = data.y[mask]
        
        # Calculate accuracy
        acc = pred.eq(labels).sum().item() / mask.sum().item()
        
        # Calculate loss
        loss = F.cross_entropy(out[mask], labels).item()
    
    return acc, loss, pred.cpu().numpy(), labels.cpu().numpy()


def get_all_predictions(model, data):
    """Get predictions for all nodes (for final evaluation)"""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.max(1)[1]
    
    return pred.cpu().numpy(), data.y.cpu().numpy()

#------
## Loop
#-----

from tqdm.auto import tqdm

best_val_acc = 0
best_test_acc = 0
patience_counter = 0
max_patience = 100

# Store metrics for plotting (optional)
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'test_acc': []
}

for epoch in tqdm(range(1, 201), desc="Training"):
    # Train
    train_loss = train(model, data, optimizer, loss_fn)
    
    # Evaluate on all splits
    train_acc, _, _, _ = evaluate(model, data, data.train_mask)
    val_acc, val_loss, _, _ = evaluate(model, data, data.val_mask)
    test_acc, _, _, _ = evaluate(model, data, data.test_mask)
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Store metrics
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['test_acc'].append(test_acc)
    
    # Track best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        patience_counter = 0
        
        # Save best model (optional)
        #torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
    
    # Print progress
    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")
    
    # Early stopping
    if patience_counter >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break
'''
Epoch 001 | Train Loss: 1.9567 | Train Acc: 0.1429 | Val Loss: 1.9655 | Val Acc: 0.1620 | Test Acc: 0.1490
Epoch 010 | Train Loss: 1.7768 | Train Acc: 0.7857 | Val Loss: 1.8759 | Val Acc: 0.3260 | Test Acc: 0.3240
Epoch 020 | Train Loss: 1.4020 | Train Acc: 0.9500 | Val Loss: 1.6829 | Val Acc: 0.5220 | Test Acc: 0.5060
Epoch 030 | Train Loss: 0.9558 | Train Acc: 0.9929 | Val Loss: 1.3984 | Val Acc: 0.6960 | Test Acc: 0.6830
Epoch 040 | Train Loss: 0.6343 | Train Acc: 0.9929 | Val Loss: 1.1357 | Val Acc: 0.7700 | Test Acc: 0.7640
Epoch 050 | Train Loss: 0.4225 | Train Acc: 1.0000 | Val Loss: 0.9647 | Val Acc: 0.7680 | Test Acc: 0.7810
Epoch 060 | Train Loss: 0.2931 | Train Acc: 1.0000 | Val Loss: 0.8896 | Val Acc: 0.7680 | Test Acc: 0.7830
Epoch 070 | Train Loss: 0.2525 | Train Acc: 1.0000 | Val Loss: 0.8573 | Val Acc: 0.7660 | Test Acc: 0.7840
Epoch 080 | Train Loss: 0.2423 | Train Acc: 1.0000 | Val Loss: 0.8300 | Val Acc: 0.7680 | Test Acc: 0.7850
Epoch 090 | Train Loss: 0.2281 | Train Acc: 1.0000 | Val Loss: 0.8003 | Val Acc: 0.7680 | Test Acc: 0.7920
Epoch 100 | Train Loss: 0.2138 | Train Acc: 1.0000 | Val Loss: 0.7997 | Val Acc: 0.7700 | Test Acc: 0.7860
Epoch 110 | Train Loss: 0.2043 | Train Acc: 1.0000 | Val Loss: 0.7788 | Val Acc: 0.7720 | Test Acc: 0.7940
Epoch 120 | Train Loss: 0.1886 | Train Acc: 1.0000 | Val Loss: 0.7695 | Val Acc: 0.7680 | Test Acc: 0.7990
Epoch 130 | Train Loss: 0.1569 | Train Acc: 1.0000 | Val Loss: 0.7588 | Val Acc: 0.7700 | Test Acc: 0.8000
Epoch 140 | Train Loss: 0.1455 | Train Acc: 1.0000 | Val Loss: 0.7584 | Val Acc: 0.7640 | Test Acc: 0.7940
Epoch 150 | Train Loss: 0.1669 | Train Acc: 1.0000 | Val Loss: 0.7624 | Val Acc: 0.7760 | Test Acc: 0.7920
Epoch 160 | Train Loss: 0.1374 | Train Acc: 1.0000 | Val Loss: 0.7564 | Val Acc: 0.7640 | Test Acc: 0.7940
Epoch 170 | Train Loss: 0.1390 | Train Acc: 1.0000 | Val Loss: 0.7259 | Val Acc: 0.7660 | Test Acc: 0.8030
Epoch 180 | Train Loss: 0.1559 | Train Acc: 1.0000 | Val Loss: 0.7384 | Val Acc: 0.7740 | Test Acc: 0.8020
Epoch 190 | Train Loss: 0.1207 | Train Acc: 1.0000 | Val Loss: 0.7355 | Val Acc: 0.7720 | Test Acc: 0.7990
Epoch 200 | Train Loss: 0.1466 | Train Acc: 1.0000 | Val Loss: 0.7330 | Val Acc: 0.7740 | Test Acc: 0.8000
'''

##############################################
## Classification report - Confusion matrix ##
##############################################

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

_, _, test_preds, test_labels = evaluate(model, data, data.test_mask)

print(classification_report(test_labels, test_preds, 
                          target_names=[f"Class {i}" for i in range(cora.num_classes)]))

cm = confusion_matrix(test_labels, test_preds)

#----
## Cora Dataset Class Names
#----
# Cora has 7 classes representing different research paper topics
class_names = [
    "Case_Based",      # Class 0
    "Genetic_Algorithms", # Class 1
    "Neural_Networks",    # Class 2
    "Probabilistic_Methods", # Class 3
    "Reinforcement_Learning", # Class 4
    "Rule_Learning",      # Class 5
    "Theory"              # Class 6
]

#----
## Plotly Express (Like your example)
#----

import plotly.express as px

fig1 = px.imshow(
    cm,
    text_auto=True,  # Shows the numbers inside the squares
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=class_names,
    y=class_names,
    color_continuous_scale='Blues',
    title='GNN Confusion Matrix on Cora Dataset (Interactive)'
)

fig1.update_layout(
    xaxis_title='Predicted Label',
    yaxis_title='True Label',
    width=800,
    height=800,
    font=dict(size=12)
)

fig1.show()