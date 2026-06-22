import torch
import torch.nn as nn
from components import DoubleConv, Up, Down, SelfAttention
from utils import get_wasserstein_grad

class UNet(nn.Module):
    """
    Clean, configurable UNet for conditional diffusion.
    
    Change architecture by modifying `channels` and `attention_levels` only!
    
    Example configurations:
    
    # Tiny (fast, ~1M params)
    channels = [16, 32, 64, 128]
    attention_levels = [2, 3]  # Add attention at 3rd and 4th levels
    
    # Small (balanced, ~5M params)
    channels = [32, 64, 128, 256]
    attention_levels = [1, 2, 3]
    
    # Medium (your current, ~10M params)
    channels = [16, 32, 64, 128]
    attention_levels = [1, 2, 3]
    num_heads = 8
    
    # Large (high quality, ~30M params)
    channels = [64, 128, 256, 512]
    attention_levels = [0, 1, 2, 3]
    num_heads = 16
    """
    
    def __init__(
        self, 
        in_channels=6,           # 3 (coords) + 3 (wasserstein grad)
        out_channels=3,          # Predict 3D noise
        time_dim=256,
        channels=[16, 32, 64, 128],  # Channel progression (determines depth)
        attention_levels=[1, 2, 3],  # Which levels get self-attention (0-indexed)
        num_heads=8,                 # Number of attention heads
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.channels = channels
        self.num_levels = len(channels)

        # time_mlp for better time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.GELU(),
            nn.Linear(time_dim*4, time_dim)
        )
        
        # Initial convolution
        self.initial_conv = DoubleConv(in_channels, channels[0])
        
        # ===== ENCODER =====
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        
        for i in range(self.num_levels - 1):
            # Downsampling block
            down = Down(channels[i], channels[i + 1], emb_dim=time_dim)
            self.encoder_blocks.append(down)
            
            # Optional self-attention
            if i in attention_levels:
                attn = SelfAttention(channels[i + 1], num_heads)
            else:
                attn = nn.Identity()  # No attention
            self.encoder_attentions.append(attn)
        
        # ===== BOTTLENECK =====
        bottleneck_dim = channels[-1]
        self.bottleneck = nn.Sequential(
            DoubleConv(bottleneck_dim, 2*bottleneck_dim),
            DoubleConv(2*bottleneck_dim, 2*bottleneck_dim),
            DoubleConv(2*bottleneck_dim, bottleneck_dim),
        )
        
        # ===== DECODER =====
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        
        for i in reversed(range(self.num_levels - 1)):
            # Input: upsampled features + skip connection
            # upsampled: channels[i+1], skip: channels[i]
            # combined: channels[i+1] + channels[i]
            in_ch = channels[i + 1] + channels[i]
            out_ch = channels[i]
            
            up = Up(in_ch, out_ch, time_dim=time_dim)
            self.decoder_blocks.append(up)
            
            # Optional self-attention
            if i in attention_levels:
                attn = SelfAttention(out_ch, num_heads)
            else:
                attn = nn.Identity()
            self.decoder_attentions.append(attn)
        
        # ===== OUTPUT HEADS =====
        final_channels = channels[0]
        self.noise_head = nn.Conv1d(final_channels, out_channels, kernel_size=1)
        self.label_head = nn.Conv1d(final_channels, 1, kernel_size=1)
    
    def pos_encoding(self, t, channels):
        """Sinusoidal position encoding for timesteps"""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t, target_coords, source_valid_mask, target_valid_mask):
        """
        Args:
            x: [B, L, 3] - noisy coordinates
            t: [B] - timesteps
            target_coords: [B, L, 3] - target structure
            source_valid_mask: [B, L] - valid positions in source
            target_valid_mask: [B, L] - valid positions in target
            
        Returns:
            noise_pred: [B, L, 3] - predicted noise
            label_logits: [B, L, 1] - paired/unpaired classification
        """
        # 1. Time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t_emb = self.pos_encoding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # 2. Wasserstein gradient (conditioning)
        with torch.no_grad():
            wasserstein_grad = get_wasserstein_grad(
                x, target_coords, 
                source_valid_mask, target_valid_mask
            )
        
        # 3. Concatenate input with gradient
        x_concat = torch.cat([x, wasserstein_grad], dim=-1)  # [B, L, 6]
        x = x_concat.transpose(1, 2)  # [B, 6, L] for Conv1d
        
        # 4. Initial convolution
        x = self.initial_conv(x)  # [B, channels[0], L]
        
        # 5. ENCODER with skip connections
        skip_connections = [x]
        
        for down, attn in zip(self.encoder_blocks, self.encoder_attentions):
            x = down(x, t_emb)
            x = attn(x)
            skip_connections.append(x)
        
        # 6. BOTTLENECK
        x = self.bottleneck(x)
        
        # 7. DECODER with skip connections
        skip_connections = skip_connections[:-1]  # Remove last (we're at bottleneck)
        
        for up, attn in zip(self.decoder_blocks, self.decoder_attentions):
            skip_x = skip_connections.pop()  # Get corresponding skip connection
            x = up(x, skip_x, t_emb)
            x = attn(x)
        
        # 8. OUTPUT HEADS
        noise_pred = self.noise_head(x).transpose(1, 2)      # [B, L, 3]
        label_logits = self.label_head(x).transpose(1, 2)    # [B, L, 1]
        
        return noise_pred, label_logits
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        """Print architecture summary"""
        print("\n" + "="*60)
        print("UNet Architecture Summary")
        print("="*60)
        print(f"Input channels: {self.initial_conv.double_conv[0].in_channels}")
        print(f"Output channels: {self.noise_head.out_channels}")
        print(f"Time embedding dim: {self.time_dim}")
        print(f"\nEncoder/Decoder depth: {self.num_levels - 1} levels")
        print(f"Channel progression: {' -> '.join(map(str, self.channels))}")
        print(f"Attention levels: {self.encoder_attentions}")
        print(f"\nTotal parameters: {self.count_parameters():,}")
        print("="*60 + "\n")


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_unet(config='medium', device='cuda'):
    """
    Get a pre-configured UNet model.
    
    Args:
        config: 'tiny', 'small', 'medium', 'large', or 'xlarge'
        device: 'cuda' or 'cpu'
    
    Returns:
        Configured UNet model
    """
    configs = {
        'tiny': {
            'channels': [16, 32, 64],
            'attention_levels': [1, 2],
            'num_heads': 4,
            'time_dim': 128,
        },
        'small': {
            'channels': [16, 32, 64, 128],
            'attention_levels': [2, 3],
            'num_heads': 8,
            'time_dim': 256,
        },
        'medium': {  # Your current architecture
            'channels': [16, 32, 64, 128],
            'attention_levels': [1, 2, 3],
            'num_heads': 8,
            'time_dim': 256,
        },
        'large': {
            'channels': [32, 64, 128, 256],
            'attention_levels': [1, 2, 3],
            'num_heads': 16,
            'time_dim': 512,
        },
        'xlarge': {
            'channels': [64, 128, 256, 512],
            'attention_levels': [0, 1, 2, 3],
            'num_heads': 16,
            'time_dim': 512,
        }
    }
    
    if config not in configs:
        raise ValueError(f"Config '{config}' not found. Choose from: {list(configs.keys())}")
    
    cfg = configs[config]
    model = ConfigurableUNet(
        in_channels=6,
        out_channels=3,
        channels=cfg['channels'],
        attention_levels=cfg['attention_levels'],
        num_heads=cfg['num_heads'],
        time_dim=cfg['time_dim'],
        device=device
    )
    
    model.print_architecture()
    
    return model


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    
    # Example 1: Use preset configuration
    print("\n" + "="*60)
    print("EXAMPLE 1: Using preset configurations")
    print("="*60)
    
    model_tiny = get_unet('tiny')
    model_medium = get_unet('medium')
    model_large = get_unet('large')
    
    
    # Example 2: Custom configuration
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom configuration")
    print("="*60)
    
    custom_model = ConfigurableUNet(
        in_channels=6,
        out_channels=3,
        channels=[8, 16, 32, 64, 128],  # 5 levels (deeper network!)
        attention_levels=[2, 3, 4],      # Attention on last 3 levels
        num_heads=4,
        time_dim=256,
        device='cuda'
    )
    
    custom_model.print_architecture()
    
    
    # Example 3: Test forward pass
    print("\n" + "="*60)
    print("EXAMPLE 3: Test forward pass")
    print("="*60)
    
    model = get_unet('medium', device='cpu')
    
    # Dummy inputs
    batch_size = 2
    seq_len = 500
    
    x = torch.randn(batch_size, seq_len, 3)
    t = torch.randint(0, 1000, (batch_size,))
    target_coords = torch.randn(batch_size, 1000, 3)  # GMM downsampled target
    source_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    target_mask = torch.ones(batch_size, 1000, dtype=torch.bool)
    
    # Forward pass
    noise_pred, label_logits = model(x, t, target_coords, source_mask, target_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Noise prediction shape: {noise_pred.shape}")
    print(f"Label logits shape: {label_logits.shape}")
    print("\n✓ Forward pass successful!")