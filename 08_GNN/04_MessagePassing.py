'''
MESSAGEPASSING WORKFLOW IN PYTORCH GEOMETRIC (PyG)

The MessagePassing class is the base class for creating custom GNN layers in PyG. 
It implements the message passing scheme which consists of three main steps:

1. MESSAGE PROPAGATION
   - Aggregate neighbor information to each node
   - Formula: x_i' = γ(x_i, aggr_{j∈N(i)} φ(x_i, x_j, e_{ij}))
     where:
     - x_i = node features
     - e_{ij} = edge features
     - φ = message function
     - aggr = aggregation function (sum, mean, max, etc.)
     - γ = update function

2. KEY METHODS TO IMPLEMENT

   a) __init__(self, aggr='add', ...)
      - Initialize the layer
      - Set aggregation scheme: 'add', 'mean', 'max', etc.
      
   b) forward(self, x, edge_index, ...)
      - Main forward pass
      - Calls self.propagate(edge_index, x=x, ...)
      
   c) message(self, x_j, ...)
      - Constructs messages from neighbors to central node
      - x_j: neighbor node features
      - Optional: x_i (central node), edge_attr (edge features)
      - Returns: message tensor
      
   d) aggregate(self, inputs, index, ...)
      - Aggregates messages (usually handled automatically)
      - Default uses the 'aggr' specified in __init__
      
   e) update(self, aggr_out, ...)
      - Updates node embeddings after aggregation
      - aggr_out: aggregated messages
      - Returns: updated node features

3. EXECUTION FLOW

   forward() 
     → propagate()
       → message_and_aggregate() OR (message() → aggregate())
       → update()
     → return updated features

4. IMPORTANT NOTES

   - propagate() automatically creates x_i and x_j from x based on edge_index
   - x_i: central node features [num_edges, features]
   - x_j: neighbor node features [num_edges, features]
   - Edge features can be passed via edge_attr parameter
   - The subscript in method arguments (_i, _j) determines the indexing:
     * _j: source nodes (where edges come from)
     * _i: target nodes (where edges go to)
   - You can pass additional arguments through propagate() to message(), 
     aggregate(), and update() methods

5. TYPICAL USE CASES

   - message(): Apply transformation to neighbor features (e.g., attention weights)
   - aggregate(): Choose how to combine messages (sum, mean, max, etc.)
   - update(): Apply final transformation after aggregation (e.g., activation, dropout)
'''

#########################################
## Example of MessagePassing GNN layer ##
#########################################

from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import MultiAggregation

class GNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        # Use MultiAggregation
        multi_aggr = MultiAggregation(aggrs=['mean', 'max'], mode='cat')
        super().__init__(aggr=multi_aggr) 
        
        self.hidden_dim = hidden_dim
        
        # Edge MLP: learns to transform given edge features into useful messages (edge features will be created later by calculating distance)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim * 2), # hidden_dim*2+5 for [h_i, h_j, distance, direction (3D), dotproduct]
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), # hidden_dim*3 for [h_i, aggregated_messages with 'mean' and 'max']
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, h, pos, edge_index):
        '''
        Inputs:
        + h: [N, hidden_dim] node features
        + pos: [N, 3] node positions (x-y-z coordinates)
        + edge_index: [2, E]
        '''
        # Message passing
        h_updated = self.propagate(edge_index, h=h, pos=pos)
        
        # Residual connection
        return h + h_updated
    
    def message(self, h_i, h_j, pos_i, pos_j):
        '''
        Computs messages from j to i
        
        PyG automatically provides
        + h_i, h_j: node features [E, hidden_dim]
        + pos_i, pos_j: postions [E, 3]
        '''
        # 1. Relative vector (direction)
        rel_pos = pos_j - pos_i # [E, 3]
        
        # 2. Euclidean distance
        dist = rel_pos.norm(dim=-1, keepdim=True) # [E, 1]

        # 3. Normalized direction
        direction = rel_pos / (dist + 1e-8) # [E, 3]

        # 4. Dot prodcut (angular information)
        origin_dir_i = pos_i / (pos_i.norm(dim=-1, keepdim=True) + 1e-8)
        origin_dir_j = pos_j / (pos_j.norm(dim=-1, keepdim=True) + 1e-8)
        dot_product = (origin_dir_i * origin_dir_j).sum(dim=-1, keepdim=True) # [E, 1]
        
        # Concatenate [h_i, h_j, distance, direction, dotproduct]
        edge_features = torch.cat([h_i, h_j, dist, direction, dot_product], dim=-1) # [E, 2*hidden_dim + 5]
        
        messages = self.edge_mlp(edge_features)      
        return messages
    
    def update(self, aggr_out, h):
        '''Update node features'''
        input_combined = torch.cat([h, aggr_out], dim=-1) # [N, 3*hidden_dim]
        node_updated = self.node_mlp(input_combined)
        return node_updated

'''
================================================================================
PART 1: GNNLayer - Custom MessagePassing Layer
================================================================================

INITIALIZATION (__init__)
-------------------------
1. Uses MultiAggregation with ['mean', 'max'] and mode='cat'
   - This means messages will be aggregated in TWO ways:
     * Mean aggregation: average of neighbor messages
     * Max aggregation: max pooling of neighbor messages
   - mode='cat' concatenates both results → output is [N, hidden_dim*2]

2. Edge MLP (edge_mlp)
   - Input: [h_i, h_j, distance, direction(3D), dotproduct] = hidden_dim*2 + 5
   - Purpose: Transform edge features into meaningful messages
   - Output: [E, hidden_dim] messages

3. Node Update MLP (node_mlp)
   - Input: [h_i, mean_aggregated, max_aggregated] = hidden_dim*3
   - Purpose: Update node features based on aggregated neighbor info
   - Output: [N, hidden_dim] updated features


FORWARD PASS
------------
Input:
  - h: [N, hidden_dim] node features
  - pos: [N, 3] node positions
  - edge_index: [2, E] connectivity

Flow:
  h (node features) + pos (positions) + edge_index
    ↓
  self.propagate(edge_index, h=h, pos=pos)
    ↓
  h_updated = [N, hidden_dim]
    ↓
  return h + h_updated (residual connection)


MESSAGE FUNCTION
----------------
Called automatically by propagate() for each edge.

Automatic indexing by PyG:
  - h_i: [E, hidden_dim] - features of TARGET nodes (where edges point to)
  - h_j: [E, hidden_dim] - features of SOURCE nodes (where edges come from)
  - pos_i: [E, 3] - positions of TARGET nodes
  - pos_j: [E, 3] - positions of SOURCE nodes

Computation steps:
  1. rel_pos = pos_j - pos_i
     → Relative vector from i to j: [E, 3]

  2. dist = ||rel_pos||
     → Euclidean distance: [E, 1]

  3. direction = rel_pos / dist
     → Normalized direction vector: [E, 3]

  4. origin_dir_i = pos_i / ||pos_i||
     origin_dir_j = pos_j / ||pos_j||
     dot_product = origin_dir_i · origin_dir_j
     → Angular information (how aligned are nodes from origin): [E, 1]

  5. edge_features = concat[h_i, h_j, dist, direction, dot_product]
     → Combined features: [E, 2*hidden_dim + 5]

  6. messages = edge_mlp(edge_features)
     → Transformed messages: [E, hidden_dim]

Return: [E, hidden_dim] messages for each edge


AGGREGATION (automatic)
------------------------
PyG automatically applies MultiAggregation:
  - Takes messages: [E, hidden_dim]
  - Groups by target node (using edge_index[1])
  - Applies both 'mean' and 'max' aggregation
  - Concatenates results: [N, hidden_dim*2]
    * First hidden_dim: mean-aggregated messages
    * Second hidden_dim: max-aggregated messages


UPDATE FUNCTION
---------------
Input:
  - aggr_out: [N, hidden_dim*2] from MultiAggregation (mean + max concatenated)
  - h: [N, hidden_dim] original node features

Computation:
  1. input_combined = concat[h, aggr_out]
     → [N, 3*hidden_dim]

  2. node_updated = node_mlp(input_combined)
     → [N, hidden_dim]

Return: Updated node features
'''

#####################################
## GNNModel building with GNNLayer ##
#####################################

from torch_geometric.nn import knn_graph

class GNNModel(nn.Module):
    def __init__(self, hidden_dim=16, num_layers=4, k_nn=20, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_nn = k_nn
        self.device = device

        # Project pos [N, 3] matrix into h [N, hidden_dim]
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize gnn_layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GNNLayer(hidden_dim=hidden_dim))

        # Final output head: return new updated x-y-z coordinates
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)   
        )

    def forward(self, data):
        '''
        Inputs:
        + data: Batch object with data.pos [N, 3]
        
        Output:
        + predicted coordinates: [N, 3], predicted noise for each point
        '''
        pos = data.pos # [N, 3]
        batch = data.batch # [N]


        # Build edges dynamically
        edge_index = knn_graph(data.pos, k=self.k_nn, batch=batch)

        # Inital encoding of position
        h = self.pos_encoder(pos) # [N, 3] -> [N, hidden_dim]

        for layer in self.gnn_layers:
            h = layer(h, pos, edge_index)

        predicted_coords = self.output_mlp(h)
        return predicted_coords
'''
================================================================================
PART 2: GNNModel - Complete Model Architecture
================================================================================

INITIALIZATION
--------------
Components:
  1. pos_encoder: Projects 3D coordinates to hidden_dim
  2. gnn_layers: List of num_layers GNNLayer instances
  3. output_mlp: Projects hidden_dim back to 3D coordinates


FORWARD PASS - COMPLETE EXECUTION FLOW
---------------------------------------

INPUT: data.pos [N, 3], data.batch [N]

Step 1: Build k-NN graph
  edge_index = knn_graph(pos, k=20, batch=batch)
  → Connects each node to its 20 nearest neighbors
  → Returns: [2, E] where E ≈ N * k_nn

Step 2: Initial encoding
  h = pos_encoder(pos)
  → [N, 3] → [N, hidden_dim]
  → Embeds 3D positions into learnable feature space

Step 3: Message passing (repeated num_layers times)
  For each GNNLayer:
    
    3.1: Forward call
      h_updated = layer(h, pos, edge_index)
      
    3.2: Inside layer.forward():
      h_updated = propagate(edge_index, h=h, pos=pos)
      
    3.3: PyG automatically calls message() for each edge:
      - Computes geometric features (distance, direction, angle)
      - Creates messages using edge_mlp
      
    3.4: PyG aggregates messages:
      - Mean aggregation: [N, hidden_dim]
      - Max aggregation: [N, hidden_dim]
      - Concatenate: [N, hidden_dim*2]
      
    3.5: PyG calls update():
      - Combines [h, aggr_out] → [N, 3*hidden_dim]
      - Transforms via node_mlp → [N, hidden_dim]
      
    3.6: Residual connection:
      h = h + h_updated
      
  After all layers: h has been refined num_layers times

Step 4: Generate predictions
  predicted_coords = output_mlp(h)
  → [N, hidden_dim] → [N, 3]
  → Maps refined features back to 3D coordinate space

OUTPUT: [N, 3] predicted coordinates
'''
