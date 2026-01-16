###############
## Utilities ##
###############

import torch

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import plotly.graph_objects as go
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow( torch.cat([
        torch.cat([img for img in images.cpu()], dim=-1)
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    
def create_diffusion_animation(img_list, fps=30, skip_frames=1):
    """
    Create animation of diffusion sampling process.
    
    Args:
        img_list: List of image tensors from sampling
        fps: Animation speed
        skip_frames: Show every Nth frame (default: 10)
    """
    # Take every Nth frame and first image from batch
    frames = []
    for i in range(0, len(img_list), skip_frames):
        img = img_list[i][0].cpu().permute(1, 2, 0).numpy()  # [3,H,W] -> [H,W,3]
        frames.append(img)
    
    # Create figure
    fig = go.Figure(
        data=[go.Image(z=frames[0])],
        layout=go.Layout(
            title="Diffusion Denoising Process",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        ),
        frames=[go.Frame(data=[go.Image(z=frame)]) for frame in frames]
    )
    
    # Add play button
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 1000/fps}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ],
            "type": "buttons"
        }]
    )
    
    return fig

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(img_size, batch_size, img_list=None, path=None):
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
    ])
    
    if (img_list is None) and (path is None):
        logger.error("No data was given")
    
    if path is not None:
        dataset = torchvision.datasets.ImageFolder(path, transform=preprocess)
    else:
        img_transformed = torch.stack([preprocess(img) for img in img_list])
        dataset = torch.utils.data.TensorDataset(img_transformed)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader