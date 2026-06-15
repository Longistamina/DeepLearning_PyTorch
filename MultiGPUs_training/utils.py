import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

##########################
## Point Clouds Dataset ##
##########################

from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.mixture import GaussianMixture
import numpy as np

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, point_clouds):
        '''
        point_clouds: list of point clouds (structures)
                     - point_clouds[0]: first 3D structure, shape [N1, 3]
                     - point_clouds[1]: second 3D structure, shape [N2, 3]
                     - point_clouds[2]: third 3D structure, shape [N3, 3]
                     ... etc
        num_points: target number of points after downsampling (default: 1000)
        '''
        self.point_clouds = point_clouds

        self.transform = T.Compose([
                T.Center() # Center the structure at (0, 0, 0)
            ])

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        pos = self.point_clouds[idx]
        if not torch.is_tensor(pos): pos = torch.tensor(pos, dtype=torch.float)
        else: pos = pos.float()
        
        data = Data(pos=pos)
        data = self.transform(data)
        
        # RETURN ONLY POS. Do not calculate edges here!
        # We want the model to be blind to the true topology.
        return data

############################
## Target shapes handling ##
############################

import numpy as np
from sklearn.mixture import GaussianMixture

def gmm_downsample(data, n_points=1000, n_components=30, covariance_type='full', random_state=42):
    # 1. Initialize and Fit GMM
    # Note: We create a new GMM instance for each shape
    gmm = GaussianMixture(
        n_components=n_components,      # Adjust based on complexity (e.g., 10-30)
        covariance_type=covariance_type, 
        n_init=1,             # n_init=1 is faster for loops
        max_iter=100,
        random_state=random_state
    )

    # 2. Check type and convert to Numpy
    if isinstance(data, torch.Tensor):
    # Detach removes gradients, cpu moves to host, numpy converts
        data_np = data.detach().cpu().numpy()
    else:
        # Handles lists, tuples, or existing numpy arrays
        data_np = np.array(data)

    # 3. Fit
    gmm.fit(data_np)
    
    weights = gmm.weights_ / gmm.weights_.sum() # Explicitly normalize weights to sum to exactly 1.0
    weights = weights * (1.0 - 1e-6) # This prevents the "sum(pvals) > 1.0" error in numpy
    gmm.weights_ = weights # Assign back

    # 4. Sample fixed number of points
    new_points, _ = gmm.sample(n_samples=n_points)
    
    # 5. Convert back to Tensor
    new_points = torch.tensor(new_points, dtype=torch.float32)
    return new_points

'''
target_shapes = []
for shape in tldm(point_clouds, desc='Preprocessing target shapes', disable=(not is_master)):
    shape_downsampled = gmm_downsample(shape, n_points=1000, n_components=30, covariance_type='full', random_state=42)
    target_shapes.append(shape_downsampled)
    
target_shapes = PointCloudDataset(point_clouds=target_shapes)
'''

##################
## Pair dataset ##
##################

from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB
        
    def __len__(self):
        return len(self.datasetA)
        
    def __getitem__(self, idx):
        # Returns a tuple: (Data object, Data object)
        return self.datasetA[idx], self.datasetB[idx]

#################################
## SinusoidalPositionEmbedding ##
#################################

class SinusoidalPositionEmbedding(nn.Module):
    """Timestep embedding like in Transformers/Diffusion Models"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        '''
        Input:
        + t: [batch_size] integer timesteps
        
        Output:
        + embeddings: [batch_size, dim]
        '''
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

#########################################
## Wassertein Loss (Optimal Transport) ##
#########################################

from torch_geometric.utils import to_dense_batch
from geomloss import SamplesLoss

# Initialize the solver
# potentials=True is NOT needed for gradients, simple loss is sufficient.
loss_solver = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

def get_wasserstein_grad(data, target_shape):
    """
    Calculates Wasserstein loss and the gradient direction to move 'data' towards 'target'.
    
    Returns:
        loss (scalar): The average Wasserstein distance.
        grad_pos (Tensor): Shape [Total_N, 3]. The vector field pushing data.pos 
                           towards target_shape.pos.
    """
    with torch.enable_grad():
    
        # 1. Ensure the input positions require gradients
        # We detach first to avoid messing up previous computation graphs if they exist,
        # then enable gradient tracking for this specific calculation.
        pos_input = data.pos.detach().clone().requires_grad_(True)
        
        # 2. Convert to Dense (Batch, Max_N, 3)
        # We use 'pos_input' here so autograd tracks it.
        batch_pos_A, mask_A = to_dense_batch(pos_input, data.batch)
        batch_pos_B, mask_B = to_dense_batch(target_shape.pos, target_shape.batch)
        
        # 3. Create Weights (normalize to sum to 1 per batch)
        weights_A = mask_A.float()
        weights_B = mask_B.float()
        weights_A = weights_A / (weights_A.sum(dim=1, keepdim=True) + 1e-6)
        weights_B = weights_B / (weights_B.sum(dim=1, keepdim=True) + 1e-6)
        
        # 4. Compute Loss
        # We sum the batch losses to get a single scalar for backward()
        # (Summing allows gradients to flow independently for each batch item)
        dist_matrix = loss_solver(weights_A, batch_pos_A, weights_B, batch_pos_B)
        loss = dist_matrix.sum()
        
        # 5. Calculate Gradient
        # This computes d(Loss)/d(pos_input)
        # The result will be exactly the same shape as pos_input: [Total_N, 3]
        grad_pos = torch.autograd.grad(loss, pos_input)[0]
    
    # NOTE: Gradient points in direction of ASCENT (increasing loss).
    # To move Data -> Target, you typically subtract this gradient.
    return grad_pos

# --- Usage Example ---
# loss, direction = get_wasserstein_grad(data_batch, target_batch)
#
# # Move points closer to target (Gradient Descent step)
# updated_pos = data_batch.pos - 0.1 * direction
    
################
## Train loop ##
################

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path

def train(model, dataloader, diffusion, epochs=5000, lr=1e-4, report_interval=100, save_path=None):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()

    #####################
    ## Scheduler setup ##
    #####################

    # warmup_epochs = int(4e-3*epochs) # 0.4% of total epochs
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=0.1, total_iters=warmup_epochs)

    # main_epochs = epochs - warmup_epochs
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-7)

    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    #####################
    
    best_loss = float('inf')
    device = diffusion.device # Ensure we use the device from the diffusion class

    model.to(device)

    for epoch in tqdm(range(1, epochs+1), desc="Training"):
        
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            inputs, target_shapes = batch
            inputs = inputs.to(device)
            target_shapes = target_shapes.to(device)
            batch_size = inputs.num_graphs 
            
            # 1. Sample timesteps
            t = diffusion.sample_timesteps(batch_size).to(device)
            
            # 2. Add noise to structures
            # noisy_batch has noisy .pos, but still has the old .edge_index (if any)
            noisy_batch, noise = diffusion.noise_structures(inputs, t)
            

            # 3. Predict noise
            # Now the model sees noisy coords AND noisy edges -> Harder task, but robust.
            predicted_noise = model(noisy_batch, target_shapes, t)
            
            # 4. Compute loss
            loss = loss_fn(noise, predicted_noise)
                        
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping is highly recommended for 3D diffusion 
            # to prevent exploding gradients in early training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        # Save if loss decreased
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path is not None:
                save_path_best = save_path.parent / f'{str(save_path.stem)}_best.pth'
                torch.save(model.state_dict(), save_path_best)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch == 1) or (epoch % report_interval == 0):
            save_path_checkpoint = Path(save_path).parent / f'{str(Path(save_path).stem)}_{int(epoch//1000)}k.pth'
            torch.save(model.state_dict(), save_path_checkpoint)
            print("+"*50)
            print(f"Epoch: {epoch} | Loss: {avg_loss:.6f} | Current LR: {current_lr:.6f}")

########################
## Visualize animated ##
########################

import numpy as np

import plotly.graph_objects as go

def create_diffusion_animation(coord_list, batch_idx=0, fps=30, skip_frames=1):
    """
    Create 3D animation of point cloud diffusion sampling process.
    
    Args:
        coord_list: List of coordinate tensors from sampling [num_samples*num_points, 3]
        batch_idx: Which structure to visualize (if multiple were generated)
        fps: Animation speed
        skip_frames: Show every Nth frame
    """
    # Assuming you know num_points per structure
    num_points = len(coord_list[0]) // (batch_idx + 1)  # Adjust based on your setup
    start_idx = batch_idx * num_points
    end_idx = start_idx + num_points
    
    # Extract frames for one structure
    frames = []
    for i in range(0, len(coord_list), skip_frames):
        coords = coord_list[i][start_idx:end_idx].cpu().numpy()  # [num_points, 3]
        frames.append(coords)
    
    # Get axis ranges (for consistent view)
    all_coords = np.concatenate(frames, axis=0)
    x_range = [all_coords[:, 0].min(), all_coords[:, 0].max()]
    y_range = [all_coords[:, 1].min(), all_coords[:, 1].max()]
    z_range = [all_coords[:, 2].min(), all_coords[:, 2].max()]
    
    # Create initial frame
    initial_frame = frames[0]
    
    fig = go.Figure(
        data=[go.Scatter3d(
            x=initial_frame[:, 0],
            y=initial_frame[:, 1],
            z=initial_frame[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue')
        )],
        layout=go.Layout(
            title="Point Cloud Diffusion Denoising Process",
            scene=dict(
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
                aspectmode='cube'
            )
        ),
        frames=[
            go.Frame(data=[go.Scatter3d(
                x=frame[:, 0],
                y=frame[:, 1],
                z=frame[:, 2],
                mode='markers',
                marker=dict(size=2, color='blue')
            )])
            for frame in frames
        ]
    )
    
    # Add play button
    fig.update_layout(
        width=600,
        height=600,
        updatemenus=[{
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 1000/fps}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ],
            "type": "buttons"
        }]
    )
    
    return fig