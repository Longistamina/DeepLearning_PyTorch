import os
import torch
import torch.nn as nn
from matplotlib import pyplot as pyplot
from tqdm.auto import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Diffusion:
    '''
    This class contains these functions:
    + noise scheduler
    + noising images
    + sampling images (generate)
    '''
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device=device):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) # Linear scheduling
    
    def noise_images(self, x, t): # nois image is x_t
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # the resulting shape will be (batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        
        x_t = sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*noise
        return x_t, noise
    
    def sample_timesteps(self, n): # create the timesteps for sampling
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        logger.info(f"Sampling {n} new images...")
        
        _ = model.eval()
        with torch.inference_mode():
            x = torch.randn(size=(n, 3, self.img_size, self.img_size)).to(self.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Sampling"):
                t = (torch.ones(n) * i).long().to(self.device) # torch.ones create [1., 1., 1., ...], multiply i creates [i., i., i., ...], .long() for integer
                predicted_noise = model(x, t)                  # example, at timestep i=5, with n=3 images, the resulted t=[5, 5, 5]
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][: None, None, None]
                beta = self.beta[t][: None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = (1/torch.sqrt(alpha)) * (x - ((1 - alpha)/torch.sqrt(1 - alpha_hat))*predicted_noise) +  torch.sqrt(beta)*noise
        
        _ = model.train()
        x = (x.clamp(-1, 1) + 1) / 2   # bring back to ranges [0, 1]
        x = (x * 255).type(torch.uint8) # convert to RGB pixel values (0-255)
        return x