######################
## Import libraries ##
######################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

#####################
## Data Simulation ##
#####################

# Define the social network graph with edge features

# Node features: [age, gender, num_posts]
node_features = torch.tensor([
    [25, 1, 100],  # Alice (node 0)
    [30, 0, 250],  # Bob (node 1)
    [28, 0, 180],  # Charlie (node 2)
    [22, 1, 320]   # Diana (node 3)
], dtype=torch.float)

# Edge connections: [source_nodes, target_nodes]
edge_index = torch.tensor([
    [0, 0, 1, 2],  # source nodes
    [1, 2, 3, 3]   # target nodes
], dtype=torch.long)

# Edge features: [years_friends, messages_per_week]
edge_features = torch.tensor([
    [5.0, 12],  # Alice -> Bob
    [3.5, 8],   # Alice -> Charlie
    [2.0, 25],  # Bob -> Diana
    [1.5, 15]   # Charlie -> Diana
], dtype=torch.float)

# Create PyTorch Geometric Data object
graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

print("Graph Data:")
print(f"Number of nodes: {graph_data.num_nodes}") # 4
print(f"Number of edges: {graph_data.num_edges}") # 4
print(f"Node feature dimension: {graph_data.num_node_features}") # 3
print(f"Edge feature dimension: {graph_data.edge_attr.shape[1]}") # 2

print(f"\nNode features:\n{graph_data.x}")
# tensor([[ 25.,   1., 100.],
#         [ 30.,   0., 250.],
#         [ 28.,   0., 180.],
#         [ 22.,   1., 320.]])

print(f"\nEdge index:\n{graph_data.edge_index}")
# tensor([[0, 0, 1, 2],   # source node
#         [1, 2, 3, 3]])  # target node

print(f"\nEdge features:\n{graph_data.edge_attr}")
# tensor([[ 5.0000, 12.0000],
#         [ 3.5000,  8.0000],
#         [ 2.0000, 25.0000],
#         [ 1.5000, 15.0000]])

#################################
## GNN that uses edge features ##
#################################

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(EdgeConv, self).__init__(aggr='add')
        # MLP for transforming node features
        self.node_mlp = nn.Linear(in_channels, out_channels)
        # MLP for transforming edge features
        self.edge_mlp = nn.Linear(edge_dim, out_channels)
        # Combine node and edge information
        self.combine_mlp = nn.Linear(out_channels * 2, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # x_j: features of source nodes [num_edges, in_channels]
        # edge_attr: edge features [num_edges, edge_dim]
        
        # Transform node and edge features
        node_transformed = self.node_mlp(x_j)
        edge_transformed = self.edge_mlp(edge_attr)
        
        # Combine them
        combined = torch.cat([node_transformed, edge_transformed], dim=1)
        message = self.combine_mlp(combined)
        
        return F.relu(message)


# Initialize and run the GNN layer
gnn_layer = EdgeConv(in_channels=3, out_channels=8, edge_dim=2)
output = gnn_layer(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

print(f"\n\nGNN Output shape: {output.shape}")
# # torch.Size([4, 8])

print(f"GNN Output (node embeddings):\n{output}")
# tensor([[  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
#            0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,  56.4988,   4.6967,
#            0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,  57.6147,   5.3287,
#            0.0000],
#         [  0.0000,   0.0000,   0.0000,   0.0000,   0.0000, 235.4666,  24.0909,
#            0.0000]], grad_fn=<ScatterAddBackward0>)