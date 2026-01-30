import os
import torch
import torch.distributed as dist
import scipy.io as scp
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from components import PointCloudDataset, Diffusion, DiffusionWrapper
from models import GenSNUPI
from tldm import tldm

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP DDP
# ==========================================
def setup():
    # Torchrun automatically sets these environment variables
    dist.init_process_group(backend="nccl")
    
    # Get local rank (0 or 1) to know which GPU this specific process uses
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()
    
def reduce_loss(tensor, rank, world_size):
    """
    Reduces the loss from all GPUs to Rank 0 for logging.
    """
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            return tensor / world_size
        else:
            return tensor

# ==========================================
# 2. MAIN TRAINING FUNCTION
# ==========================================
def main():
    local_rank = setup()
    world_size = dist.get_world_size()
    
    # Only print logs from the main process (Rank 0) to avoid clutter
    is_master = (local_rank == 0)
    
    # ----------------------
    # A. DATASET
    # ----------------------
    data_folder = Path('block_Lattice_designs')
    point_clouds = []
    
    # Load data (Every process does this, but it's fast enough)
    # Ideally, load once and broadcast, but for simple lists this is fine.
    for entry in data_folder.iterdir():
        try:
            dna = scp.loadmat(entry)
            point_clouds.append(dna.get('finl_coord')[:, :3].astype('float32'))
        except:
            continue
            
    dataset = PointCloudDataset(point_clouds=point_clouds[2:-2])

    # ----------------------
    # B. SAMPLER & LOADER (The "ListLoader" Replacement)
    # ----------------------
    # DistributedSampler splits the data:
    # Rank 0 gets indices [0, 2, 4...]
    # Rank 1 gets indices [1, 3, 5...]
    sampler = DistributedSampler(dataset, shuffle=True)
    
    train_loader = DataLoader(
        dataset,
        batch_size=1,       # Batch size PER GPU
        sampler=sampler,    # Crucial!
        shuffle=False,      # Sampler handles shuffle, so set this to False
        num_workers=4,
        pin_memory=True
    )

    # ----------------------
    # C. MODEL SETUP
    # ----------------------
    # Initialize components
    diffusion = Diffusion(device=f'cuda:{local_rank}')
    base_model = GenSNUPI(
        hidden_dim=64,
        num_layers=8,
        num_heads=8,
        time_embed_dim=64,
        k_nn=20,
        k_random=40,
        device='cuda'
    )
    
    if is_master:
        print(f'Total parameters: {sum(p.numel() for p in base_model.parameters()):,}')
    
    # Load weights (Ensure map_location is set to the local GPU)
    if os.path.exists('./save/fourth_model.pth'):
        map_location = {'cuda:0': f'cuda:{local_rank}'}
        state_dict = torch.load('./save/fourth_model.pth', map_location=map_location)
        base_model.load_state_dict(state_dict)
        if is_master: print("Loaded previous weights.")

    loss_fn = torch.nn.MSELoss()
    
    # Wrap components
    # Move to GPU *BEFORE* wrapping in DDP
    wrapper = DiffusionWrapper(base_model, diffusion, loss_fn).to(local_rank)
    
    # DDP WRAPPER
    # This handles gradient syncing across GPUs automatically
    model = DDP(wrapper, device_ids=[local_rank])

    # ----------------------
    # D. OPTIMIZER
    # ----------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500000)

    # ----------------------
    # E. TRAINING LOOP
    # ----------------------
    epochs = 500000
    best_loss = float('inf')
    
    for epoch in tldm(range(epochs), desc='Training'):
        # CRITICAL: Tell sampler which epoch it is so it shuffles differently
        train_loader.sampler.set_epoch(epoch)
        
        model.train()
        local_loss_sum = 0.0
        
        for batch in train_loader:
            # Move batch to local GPU
            batch = batch.to(local_rank)
            
            optimizer.zero_grad()
            
            # Forward pass (Wrapper handles noise/t/loss)
            loss = model(batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            local_loss_sum += loss.item()
        # ----------------------
        # F. AGGREGATE LOSS & SAVE
        # ----------------------
        # Calculate local average for this GPU
        local_avg = torch.tensor(local_loss_sum / len(train_loader)).to(local_rank)
        
        # Sync losses from all GPUs to get the true global average
        global_avg_loss = reduce_loss(local_avg, local_rank, world_size)

        # Logging (Only on Rank 0)
        if is_master:
            # --- BEST LOSS LOGIC ---
            current_loss = global_avg_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                # Use model.module to unwrap DDP before saving
                torch.save(model.module.model.state_dict(), './save/fourth_model_v2_best.pth')
                print(f"⭐ New Best! Epoch {epoch} | Loss: {best_loss:.6f}")            
                
            current_lr = optimizer.param_groups[0]['lr']
            
            if epoch % 100000 == 0:
                print(f"Epoch {epoch} | Loss: {current_loss:.6f} | LR: {current_lr:.8f}")
            
            # Save Checkpoint
            if epoch % 25000 == 0:
                # Access .module to save the underlying weights, not the DDP shell
                torch.save(model.module.model.state_dict(), f'./save/fourth_model_v2_{epoch}.pth')

        scheduler.step()

    cleanup()

if __name__ == "__main__":
    main()
    
'''
#-----------------------------#
#---- Use 2 RTX 3090 only ----#
#-----------------------------#

# 1. Force use of only the two matching GPUs (e.g. 0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

# 2. Run with torchrun (Replace 2 with the number of GPUs you exported above)
torchrun --nproc_per_node=2 /home/ssdl/Documents/genSNUPI/04_4th_ddpm3D_blck_selfattention.py

#-----------------------------------#
#---- Use 4: 2*3090 and 2*Titan ----#
#-----------------------------------#

# 1. Unset the restriction (allow all GPUs)
unset CUDA_VISIBLE_DEVICES

# 2. Add the safety flags to prevent crashing
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 3. Run on all 4 GPUs
torchrun --nproc_per_node=4 /home/ssdl/Documents/genSNUPI/04_4th_ddpm3D_blck_selfattention.py
'''