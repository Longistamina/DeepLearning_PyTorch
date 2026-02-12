import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############
## Diffusion ##
###############

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
        
        # 2. REGISTER BUFFERS (automatically moves to correct device)
        self.register_buffer('beta', beta.float().to(device))
        self.register_buffer('alpha', alpha.float().to(device))
        self.register_buffer('alpha_hat', alpha_hat.float().to(device))
        
    def noise_schedule(self):
        """Linear noise schedule"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, dtype=torch.float32)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to x_start at timestep t
        
        Args:
            x_start: [B, L, 3] - original clean coordinates
            t: [B] - timesteps for each sample in batch
            noise: [B, L, 3] - optional pre-generated noise (if None, generates new)
            
        Returns:
            x_t: [B, L, 3] - noisy coordinates at timestep t
            noise: [B, L, 3] - the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get alpha values for each timestep
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])  # [B]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])  # [B]
        
        # Expand to match coordinate dimensions: [B] -> [B, 1, 1]
        sqrt_alpha_hat = sqrt_alpha_hat.view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.view(-1, 1, 1)
        
        # Apply noise
        x_t = sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise
        
        return x_t, noise
    
    def noise_structures(self, x, t):
        """
        Alias for q_sample for backward compatibility
        
        Args:
            x: [B, L, 3] - coordinates
            t: [B] - timesteps
            
        Returns:
            noisy_x: [B, L, 3] - noisy coordinates
            noise: [B, L, 3] - the noise added
        """
        return self.q_sample(x, t)
    
    def predict_x0_from_noise(self, x_t, t, predicted_noise):
        """
        Predict x_0 (clean data) from x_t and predicted noise
        
        Args:
            x_t: [B, L, 3] - noisy coordinates at timestep t
            t: [B] - timesteps
            predicted_noise: [B, L, 3] - noise predicted by model
            
        Returns:
            x_0_pred: [B, L, 3] - predicted clean coordinates
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t]).view(-1, 1, 1)
        
        # Reverse the forward process formula
        x_0_pred = (x_t - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
        
        return x_0_pred
    
    def sample_timesteps(self, n):
        """Sample random timesteps for n samples"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)
    
    def p_sample(self, x_t, t, predicted_noise):
        """
        Single reverse diffusion step (DDPM sampling)
        
        Args:
            x_t: [B, L, 3] - noisy coordinates at timestep t
            t: [B] - timesteps
            predicted_noise: [B, L, 3] - noise predicted by model
            
        Returns:
            x_t_minus_1: [B, L, 3] - denoised coordinates at timestep t-1
        """
        batch_size = x_t.shape[0]
        
        # Get parameters for current timestep
        # Handle both tensor and scalar timesteps
        if t.dim() == 0:  # scalar
            t_idx = t.item()
            alpha = self.alpha[t_idx]
            alpha_hat = self.alpha_hat[t_idx]
            beta = self.beta[t_idx]
        else:  # tensor [B]
            # For batch, take first element (assumes same t for all)
            t_idx = t[0].item()
            alpha = self.alpha[t_idx]
            alpha_hat = self.alpha_hat[t_idx]
            beta = self.beta[t_idx]
        
        # Add noise (except for final step)
        if t_idx > 1:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        # DDPM update equation
        x_t_minus_1 = (1 / torch.sqrt(alpha)) * (
            x_t - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
        ) + torch.sqrt(beta) * noise
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(
        self,
        model,
        target_coords,
        target_valid_mask,
        num_points,
        num_samples=1,
        save_interval=50,
        save_steps=False
    ):
        """
        Generate new structures conditioned on target shape.
        
        Args:
            model: Your UNet model
            target_coords: [N, 3] - single target structure coordinates
            target_valid_mask: [N] - valid positions in target
            num_points: Number of points to generate in output structure
            num_samples: Number of samples to generate
            save_interval: How often to save intermediate steps
            save_steps: Whether to save intermediate denoising steps
            
        Returns:
            generated_coords: [num_samples, num_points, 3] - generated coordinates
            coord_list: List of intermediate coordinates (if save_steps=True)
        """
        model.eval()
        device = self.alpha_hat.device
        
        # Ensure target is 2D [N, 3]
        if target_coords.dim() == 3:
            # If [1, N, 3], squeeze to [N, 3]
            target_coords = target_coords.squeeze(0)
            target_valid_mask = target_valid_mask.squeeze(0)
        
        # Get target dimensions
        target_len = target_coords.shape[0]
        
        # Expand target to batch: [N, 3] -> [num_samples, N, 3]
        target_coords_batch = target_coords.unsqueeze(0).repeat(num_samples, 1, 1).to(device)
        target_valid_mask_batch = target_valid_mask.unsqueeze(0).repeat(num_samples, 1).to(device)
        
        # 1. Start from pure Gaussian noise
        x = torch.randn(num_samples, num_points, 3, device=device)
        
        # Initialize source valid mask (will be updated based on model predictions)
        source_valid_mask = torch.ones(num_samples, num_points, dtype=torch.bool, device=device)
        
        if save_steps:
            coord_list = [x.cpu().clone()]
        
        # 2. Denoising Loop
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Sampling"):
            t = torch.full((num_samples,), i, dtype=torch.long, device=device)
            
            # Predict noise and labels
            predicted_noise, label_logits = model(
                x,
                t,
                target_coords_batch,
                source_valid_mask,
                target_valid_mask_batch
            )
            
            # Update source_valid_mask based on predictions
            source_valid_mask = (label_logits.sigmoid().squeeze(-1) > 0.5)
            
            # Perform denoising step
            x = self.p_sample(x, t, predicted_noise)
            
            # Save intermediate snapshots for visualization
            if save_steps and ((i % save_interval == 0) or (i == 1)):
                coord_list.append(x.cpu().clone())
        
        model.train()
        
        if save_steps:
            return x, coord_list
        else:
            return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        target_coords,
        target_valid_mask,
        num_points,
        num_samples=1,
        ddim_steps=50,
        eta=0.0,
        save_steps=False
    ):
        """
        DDIM sampling (faster than DDPM, deterministic when eta=0)
        
        Args:
            model: Your UNet model
            target_coords: [N, 3] - single target structure
            target_valid_mask: [N] - valid positions
            num_points: Number of points to generate in output structure
            num_samples: Number of samples
            ddim_steps: Number of denoising steps (can be < noise_steps for speed)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
            save_steps: Whether to save intermediate steps
            
        Returns:
            generated_coords: [num_samples, num_points, 3]
            coord_list: List of intermediate coords (if save_steps=True)
        """
        model.eval()
        device = self.alpha_hat.device
        
        # Ensure target is 2D [N, 3]
        if target_coords.dim() == 3:
            target_coords = target_coords.squeeze(0)
            target_valid_mask = target_valid_mask.squeeze(0)
        
        # Get target dimensions
        target_len = target_coords.shape[0]
        
        # Expand target to batch: [N, 3] -> [num_samples, N, 3]
        target_coords_batch = target_coords.unsqueeze(0).repeat(num_samples, 1, 1).to(device)
        target_valid_mask_batch = target_valid_mask.unsqueeze(0).repeat(num_samples, 1).to(device)
        
        # Create subset of timesteps for DDIM
        step_size = self.noise_steps // ddim_steps
        timesteps = torch.arange(0, self.noise_steps, step_size, device=device)
        timesteps = torch.flip(timesteps, [0])
        
        # Start from noise
        x = torch.randn(num_samples, num_points, 3, device=device)
        source_valid_mask = torch.ones(num_samples, num_points, dtype=torch.bool, device=device)
        
        if save_steps:
            coord_list = [x.cpu().clone()]
        
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            t_batch = torch.full((num_samples,), t.item(), dtype=torch.long, device=device)
            
            # Predict noise
            predicted_noise, label_logits = model(
                x,
                t_batch,
                target_coords_batch,
                source_valid_mask,
                target_valid_mask_batch
            )
            
            # Update mask
            source_valid_mask = (label_logits.sigmoid().squeeze(-1) > 0.5)
            
            # DDIM update
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
            else:
                t_next = torch.tensor(0, device=device)
            
            alpha_t = self.alpha_hat[t]
            alpha_t_next = self.alpha_hat[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
            
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_next - eta**2 * (1 - alpha_t) / (1 - alpha_t_next) * self.beta[t]) * predicted_noise
            
            # Random noise
            noise = eta * torch.sqrt((1 - alpha_t) / (1 - alpha_t_next)) * torch.sqrt(self.beta[t]) * torch.randn_like(x)
            
            # Update
            x = torch.sqrt(alpha_t_next) * x_0_pred + dir_xt + noise
            
            if save_steps:
                coord_list.append(x.cpu().clone())
        
        model.train()
        
        if save_steps:
            return x, coord_list
        else:
            return x

#####################
## UNet components ##
#####################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=31, padding=15, bias=False), # padding = (kernel_size - 1)/2
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=31, padding=15, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            # Residual connection (Input + Output)
            # Note: This requires in_channels == out_channels
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None]
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        
        # Your custom Sequential block with Residuals
        self.conv = nn.Sequential(
            # 1. Refine features (Residual Block)
            DoubleConv(in_channels, in_channels, residual=True),
            # 2. Reduce channels
            DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        
        # --- CRITICAL FIX: Handle Odd Sequence Lengths ---
        # Without this, you get "Expected size 1305 but got 1304"
        diff = skip_x.shape[-1] - x.shape[-1]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        # -------------------------------------------------
        
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        
        emb = self.emb_layer(t)[:, :, None]
        return x + emb
    
####################
## Self Attention ##
####################

class SelfAttention(nn.Module):
    """
    1D Self Attention
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([hidden_dim])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # x is [Batch, Channels, Length]
        # MultiheadAttention expects [Batch, Length, Channels]
        x = x.transpose(1, 2)
        
        # Attention
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        x = attention_value + x
        
        # Feed Forward
        x = self.ff_self(x) + x
        
        # Return to [Batch, Channels, Length]
        return x.transpose(1, 2)

####################
## Trainning loop ##
####################

from pathlib import Path
from tldm import tldm

def train(
    model, 
    diffusion,
    dataloader,
    epochs=100000, 
    lr=2e-4, 
    device='cuda',
    save_path=Path("model.pt"),
    report_interval=10000
):
    # 1. Setup Data & Model
    print(f"Setting up training on {device}...")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Loss functions
    mse = nn.MSELoss()              # For coordinate noise
    bce = nn.BCEWithLogitsLoss()    # For pairing probability
    
    best_loss = float('inf')

    # 2. Training Loop
    pbar = tldm(range(1, epochs + 1), desc='Training')
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            # --- A. Unpack Batch ---
            # batch is a dict with keys: 'source', 'target', 'batch_size', 'max_len'
            # Source data (input to be noised)
            src_coords = batch['source']['coords'].to(device)       # [B, L, 3]
            src_mask = batch['source']['valid_mask'].to(device)     # [B, L]
            
            # Target data (conditioning/guidance)
            tgt_coords = batch['target']['coords'].to(device)       # [B, L, 3]
            tgt_mask = batch['target']['valid_mask'].to(device)     # [B, L]
            
            batch_size = src_coords.shape[0]

            # --- B. Diffusion Process ---
            # 1. Sample random timesteps
            t = diffusion.sample_timesteps(batch_size).to(device)
            
            # 2. Add noise to source coordinates
            # Note: We only noise the coordinates, not the masks
            x_t, noise = diffusion.q_sample(src_coords, t)
            
            # --- C. Model Prediction ---
            # Model takes noisy source + clean target
            predicted_noise, label_logits = model(
                x=x_t, 
                t=t, 
                target_coords=tgt_coords, 
                source_valid_mask=src_mask, 
                target_valid_mask=tgt_mask
            )
            
            # --- D. Calculate Loss ---
            # 1. Coordinate Loss (MSE on valid positions only)
            # We apply the mask to ignore padding/invalid points in the loss
            mask_expanded = src_mask.unsqueeze(-1).expand_as(predicted_noise)
            loss_noise = mse(
                predicted_noise * mask_expanded, 
                noise * mask_expanded
            )
            
            # 2. Classification Loss (Did this node exist?)
            # label_logits: [B, L, 1] -> squeeze to [B, L]
            # src_mask is boolean, convert to float for BCE target
            loss_label = bce(label_logits.squeeze(-1), src_mask.float())
            
            # Combined loss (you can weigh these if needed)
            loss = loss_noise + 0.1*loss_label
            
            # --- E. Optimization ---
            optimizer.zero_grad()
            loss.backward()
            
            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "MSE": f"{loss_noise.item():.4f}",
                "BCE": f"{loss_label.item():.4f}"
            })
        # --- F. End of Epoch ---
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
                
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path_best = save_path.parent / f'{str(save_path.stem)}_best.pth'
            torch.save(model.state_dict(), save_path_best)
            # print(f"Saved best model to {save_path_best}")

        # Save check point
        if (epoch == 1) or (epoch % report_interval == 0):
            save_path_checkpoint = Path(save_path).parent / f'{str(Path(save_path).stem)}_{int(epoch//1000)}k.pth'
            torch.save(model.state_dict(), save_path_checkpoint)
            print("+"*50)
            print(f"Epoch: {epoch} | Loss: {avg_loss:.6f} | Current LR: {current_lr:.6f}")
#-------
## Train
#-------

# train(model=model, data=pokemon, epochs=4000, img_size=IMG_SIZE, batch_size=BATCH_SIZE, report_interval=1000, visualize=True)