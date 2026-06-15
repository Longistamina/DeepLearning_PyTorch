###############################
## Model with SELF-ATTENTION ##
###############################

import torch
import torch.nn as nn
import torch.nn.functional as F
from components import GNNLayer, SelfAttentionLayer, random_graph
from utils import SinusoidalPositionEmbedding, get_wasserstein_grad
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.nn import knn_graph

class GenSNUPI(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=6,
        num_heads=8,
        time_embed_dim=32,
        k_nn=40,
        k_random=60,
        device='cuda'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_nn = k_nn
        self.k_random = k_random
        self.device = device
        
        # Timestep embedding (sinusoidal like in Transformers)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initial position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN and Self-Attention
        # Example: GNN -> GNN -> Self-Attention -> GNN -> GNN -> Self-Attention ...
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            # Add GNNlayer
            self.gnn_layers.append(GNNLayer(hidden_dim))

            # Add Self-Attention after every 2 GNN layers (except the last one)
            if ((i + 1) % 2 == 0) and (i < num_layers - 1):
                self.gnn_layers.append(SelfAttentionLayer(hidden_dim, num_heads=num_heads, dropout=0.1))
           # With num_layers=6: GNN -> GNN -> Self-Attention -> GNN -> GNN -> Self-Attention -> GNN -> GNN

        
        # Global Shape Encoder (to get the global shape information)
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        # Final output head: return predicted noise (delta) for each point
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)   
        )
        
    def forward(self, data, target_shape, t):
        '''
        Inputs:
        + data: Batch object with data.pos [N, 3] and data.edge_index[2, E]
        + target_shape: Batch object with target_shape.pos [N, 3]
        + t: Timestep tensor [batch_size]
        
        Output:
        + predicted noise: [N, 3], predicted noise for each point
        '''
        pos = data.pos # [N, 3]
        batch = data.batch # [N]
        grad_pos = get_wasserstein_grad(data, target_shape)*100 # [N, 3]
        pos = torch.cat([pos, grad_pos], dim=1) # [N, 6]
        
        # Build edges dynamically
        edge_knn = knn_graph(data.pos, k=self.k_nn, batch=batch)
        edge_random = random_graph(data.pos, k_random=self.k_random, batch=batch, lambda_g=1.0)
        edge_index = torch.cat([edge_knn, edge_random], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
        
        # Embed timestep: [batch_size] -> [batch_size, hidden_dim]
        t_emb = self.time_mlp(t)
        
        # Expand time embedding to each node in the same graph/structure using data.batch indices
        t_emb_expanded = t_emb[batch] # becomes [N, hidden_dim]
        
        # Inital encoding of position
        h = self.pos_encoder(pos) # [N, 6] -> [N, hidden_dim]
        
        # Add timestep information
        h = h + t_emb_expanded
        
        #------------ INTERLEAVED MESSAGE PASSING + ATTENTION ------------#
        for layer in self.gnn_layers:
            if isinstance(layer, GNNLayer):
                h = layer(h, pos, edge_index)
            elif isinstance(layer, SelfAttentionLayer):
                h = layer(h, batch)

        ###########################
        ## Global shape encoding ##
        ###########################
        
        h_mean = global_mean_pool(h, batch) # mean tells where is the structure's center
        h_max = global_max_pool(h, batch) # max tells where are the outer boundaries
        
        h_std = []
        for b in batch.unique():
            mask = (batch == b)
            std = h[mask].std(dim=0)
            h_std.append(std)
        h_std = torch.stack(h_std) # std tells how spread out the points are
        
        h_global = torch.cat([h_mean, h_max, h_std], dim=1) # [batch_size, hidden_dim*3]
        h_global_expanded = h_global[batch] # [N, hidden_dim*3]
        
        # Combine local + global information
        h = h + self.global_pool(h_global_expanded)
        
        # Predict noise, final output
        predicted_noise = self.output_mlp(h) # [N, 3]inco
        
        return predicted_noise