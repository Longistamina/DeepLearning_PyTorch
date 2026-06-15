import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

##################
## Random graph ##
##################

from torch_geometric.utils import remove_self_loops

def random_graph(x, k_random=40, lambda_g=1.0, seed=None, batch=None, device="cpu"):
    """
    Random Graph Generation
    
    Args:
        x: Tensor of node coordinates (N, 3)
        k_random: Number of stochastic long-range edges
        batch: Tensor indicating which graph each node belongs to (N,)
        lambda_g: Inverse temperature hyperparameter (control how distance affects the outcome, higher lambda_g prioritizes closer distance)
    """
    N = x.size(0)
    device = x.device

    # Stochastic Long-Range Edges
    # Calculate all-to-all Euclidean distances D_ij
    dist_matrix = torch.cdist(x, x)
    
    # Define inverse cubic propensity: log p ∝ -3 * log(D) [1, 4]
    # Add epsilon to avoid log(0) for self-loops
    eps = 1e-6
    c_dist = -3.0 * torch.log(dist_matrix + eps)
    
    # Sample uniform noise U_ij ~ Uniform(0,1)
    if seed is None:
        pass
    else:
        torch.manual_seed(seed)
    
    U = torch.rand_like(dist_matrix)
        
    # Calculate perturbed log probabilities Z_ij with Gumbel noise [5]
    # Z_ij = lambda_g * c(D_ij) - log(-log(U_ij))
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    Z = (lambda_g * c_dist) + gumbel_noise
    
    # Mask self-loops so a node doesn't connect to itself randomly
    Z.fill_diagonal_(float('-inf'))

    # Mask edges between different graphs ***
    if batch is not None:
        # Create mask: same_graph[i,j] = True if node i and j are in same graph
        same_graph = batch.unsqueeze(1) == batch.unsqueeze(0)  # [N, N]
        # Set Z to -inf for edges between different graphs
        Z = Z.masked_fill(~same_graph, float('-inf'))
    
    # Select top k_random neighbors per node [6]
    _, top_indices = torch.topk(Z, k=k_random, dim=1)
    
    # Convert to COO format (edge_index)
    row = torch.arange(N, device=device).view(-1, 1).repeat(1, k_random).view(-1) # Something like [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ... N-1, N-1, N-1, N-1]
    col = top_indices.view(-1)                                                    # Something like [1, 5, 4, 6, 2, 3, 4, 2, 3, 5, 8, 15, ...]
    edge_index = torch.stack([row, col], dim=0).to(x.device)
    
    # Clean up: remove duplicates and self-loops
    edge_index, _ = remove_self_loops(edge_index)
    
    return edge_index

###############
## Diffusion ##
###############

from tqdm import tqdm
from utils import PointCloudDataset
from torch_geometric.data import Batch, Data

class Diffusion(nn.Module):
    '''
    This class contains these functions:
    + noise scheduler
    + noising structures
    + sampling structures (generate)
    '''
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # 1. Calculate schedule
        beta = self.noise_schedule()
        alpha = 1. - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        
        # 2. REGISTER BUFFERS (The Fix for Multi-GPU)
        # PyTorch will now automatically move these to the correct GPU
        self.register_buffer('beta', beta.float().to(device))
        self.register_buffer('alpha', alpha.float().to(device))
        self.register_buffer('alpha_hat', alpha_hat.float().to(device))
        
    def noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, dtype=torch.float32)
    
    def noise_structures(self, data, t):
        """
        Add noise to point cloud coordinates
        
        data: torch_geometric.data.Batch with data.pos and data.batch
        t: [batch_size] timesteps
        """
        
        # Get alpha values for each graph in batch
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])  # [batch_size]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])  # [batch_size]
        
        # Expand to match each point (use data.batch to know which graph each point belongs to)
        sqrt_alpha_hat_expanded = sqrt_alpha_hat[data.batch].unsqueeze(-1)  # [N, 1]
        sqrt_one_minus_alpha_hat_expanded = sqrt_one_minus_alpha_hat[data.batch].unsqueeze(-1)  # [N, 1]
        
        # Add noise to coordinates
        noise = torch.randn_like(data.pos)  # [N, 3]
        noisy_pos = sqrt_alpha_hat_expanded * data.pos + sqrt_one_minus_alpha_hat_expanded * noise
        
        # Create new Data object with noisy positions (edge_index stays the same)
        noisy_data = data.clone()
        noisy_data.pos = noisy_pos
        
        return noisy_data, noise
    
    def sample_timesteps(self, n):
        """Sample random timesteps for n graphs"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    @torch.no_grad()
    def sample(self, model, target_shape, num_points=1024, num_samples=1, save_interval=50, save_steps=False):
        """
        Generate new point clouds with Dynamic Graph Updates.
        """
        model.eval()
        
        device = self.alpha_hat.device
        
        # 1. Start from pure Gaussian noise
        pos = torch.randn(num_samples * num_points, 3).to(device)
        batch = torch.arange(num_samples).repeat_interleave(num_points).to(device)
        data = Data(pos=pos, batch=batch)
        
        # target_shape
        target_shape = PointCloudDataset([target_shape])
        target_shape = Batch.from_data_list([target_shape[0]] * num_samples).to(device)
        
        if save_steps:
            coord_list = [data.pos.cpu().clone()]

        # 2. Denoising Loop
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Sampling"):
            t = torch.ones(num_samples, dtype=torch.long, device=device) * i

            # Predict noise
            predicted_noise = model(data, target_shape, t)
            
            # Get diffusion parameters for step i
            alpha = self.alpha[i]
            alpha_hat = self.alpha_hat[i]
            beta = self.beta[i]
            
            # Add noise (stochastic term), except for the very last step
            if i > 1:
                noise = torch.randn_like(data.pos)
            else:
                noise = torch.zeros_like(data.pos)
            
            # Standard DDPM Update Equation
            data.pos = (1 / torch.sqrt(alpha)) * (
                data.pos - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
            # Save intermediate snapshots for visualization
            if save_steps and ((i % save_interval == 0) or (i == 1)):
                coord_list.append(data.pos.cpu().clone())
        
        model.train()
        
        if save_steps:
            return data.pos, coord_list
        else:
            return data.pos
        
###############
## GNN layer ##
###############

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
            nn.Linear(hidden_dim * 2 + 11, hidden_dim * 2), # hidden_dim*2+5 for [h_i, h_j, grad_loss, distance, direction (3D), dotproduct]
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
        + pos_i, pos_j: postions [E, 6]
        '''
        # get the grad_wassertein_loss
        grad_loss = torch.cat([pos_i[:, 3:], pos_j[:, 3:]], dim=-1) # [N, 6]
        
        # 1. Relative vector (direction)
        rel_pos = pos_j[:, :3] - pos_i[:, :3] # [E, 3]
        
        # 2. Euclidean distance
        dist = rel_pos.norm(dim=-1, keepdim=True) # [E, 1]

        # 3. Normalized direction
        direction = rel_pos / (dist + 1e-8) # [E, 3]

        # 4. Dot prodcut (angular information)
        origin_dir_i = pos_i / (pos_i.norm(dim=-1, keepdim=True) + 1e-8)
        origin_dir_j = pos_j / (pos_j.norm(dim=-1, keepdim=True) + 1e-8)
        dot_product = (origin_dir_i * origin_dir_j).sum(dim=-1, keepdim=True) # [E, 1]
        
        # Concatenate [h_i, h_j, distance, direction, dotproduct]
        edge_features = torch.cat([h_i, h_j, grad_loss, dist, direction, dot_product], dim=-1) # [E, 2*hidden_dim + 8]
        
        messages = self.edge_mlp(edge_features)      
        return messages
    
    def update(self, aggr_out, h):
        '''Update node features'''
        input_combined = torch.cat([h, aggr_out], dim=-1) # [N, 3*hidden_dim]
        node_updated = self.node_mlp(input_combined)
        return node_updated
    
####################
## Self Attention ##
####################

class SelfAttentionLayer(nn.Module):
    '''
    Self-attention layers for batched graph data
    Processes each graph in the batch separately
    '''
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        # Querry, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Normalization and feedforward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, h, batch):
        '''
        Inputs:
        + h: [N, hidden_dim] - node features for all graphs in batch
        + batch: [N] - batch assignment tensor (which graph each node belongs to)

        Outputs:
        + h_out: [N, hidden_dim] - updated node features with self-attention mechanism
        '''

        h_out = []
        
        # Process each graph separately
        for b in batch.unique():
            mask = (batch == b)
            h_graph = h[mask] # [num_nodes_in_graph, hidden_dim]

            # Apply transformer block to this graph
            h_graph_updated = self._transformer_block(h_graph)
            h_out.append(h_graph_updated)

        h_out = torch.cat(h_out, dim=0) # [N, hidden_dim]
        return h_out

    def _transformer_block(self, x):
        '''
        Standard Transformer block with pre-norm architecture

        Input:
        + x: [num_nodes, hidden_dim] - features of a single graph

        Output:
        + x: [num_nodes, hidden_dim] - updated features with transformer
        '''
        # Multi-head self-attention with residual
        x_norm = self.norm1(x)
        attn_out = self._multihead_attention(x_norm)
        x = x + self.dropout(attn_out)

        # Feedforward network with residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        
        x = x + ffn_out
        return x

    def _multihead_attention(self, x):
        '''
        Multi-head self-attention mechanism

        Input:
        + x: [num_nodes, hidden_dim] - features of a single graph

        Output:
        + x: [num_nodes, hidden_dim] - updated features with multi-head self attention
        '''
        num_nodes = x.size(0)

        # Project to Q, K, V
        q = self.q_proj(x).reshape(num_nodes, self.num_heads, self.head_dim) # [N, Heads, Dims]
        k = self.k_proj(x).reshape(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(num_nodes, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        # scores[i, h, j] = attention from node i to node j for head h
        scale = self.head_dim ** 0.5
        scores = torch.einsum('ihd,jhd->hij', q, k) / scale # [H, N, N]
        '''
        torch.einsum('ihd,jhd->hij', q, k) computes attention scores:
        # - q: [N, num_heads, head_dim] labeled as 'ihd'
        # - k: [N, num_heads, head_dim] labeled as 'jhd'
        # - Output: [num_heads, N, N] labeled as 'hij'
        # 
        # For each attention head h:
        #   scores[h, i, j] = dot_product(q[i,h,:], k[j,h,:])
        #                   = sum_over_d(q[i,h,d] * k[j,h,d])
        # 
        # The 'd' dimension disappears (gets summed) because it's not in output 'hij'.
        # This computes how much node i should attend to node j in each head.
        '''

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1) # [H, N, N]
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.einsum('hij,jhd->ihd', attn_weights, v) # [N, H, D]
        out = out.reshape(num_nodes, self.hidden_dim) # [N, hidden_dim]

        # Final projection
        out = self.out_proj(out)
        return out
    
#######################
## Diffusion Wrapper ##
#######################

class DiffusionWrapper(nn.Module):
    # REMOVED 'device' from init arguments. We don't need it!
    def __init__(self, model, diffusion, loss_fn):
        super().__init__()
        self.model = model
        self.diffusion = diffusion
        self.loss_fn = loss_fn

    def forward(self, inputs, target_shapes):
        # 1. Detect the device automatically
        # DataParallel has already put this 'batch' on the correct GPU (e.g., cuda:3)
        # We just ask: "Where are you?"
        current_device = inputs.pos.device
        
        batch_size = inputs.num_graphs
        
        # 2. Sample timesteps on that SAME device
        t = self.diffusion.sample_timesteps(batch_size).to(current_device)
        
        # 3. Add noise to structures
        # (This logic now runs on the correct GPU automatically)
        noisy_batch, noise = self.diffusion.noise_structures(inputs, t)
        
        # 4. Predict noise using your original model
        predicted_noise = self.model(noisy_batch, target_shapes, t)
        
        # 5. Compute and return loss
        loss = self.loss_fn(noise, predicted_noise)
        return loss
