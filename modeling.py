import torch
from torch import nn, optim
import torch.nn.functional as F

import gymnasium as gym
from vizdoom import gymnasium_wrapper

class UpsampleThenConv(nn.Module):
    def __init__(self, shape, fan_in, fan_out, kernel_size):
        super().__init__()
        self.upsample = nn.Upsample((shape + kernel_size - 1, shape + kernel_size - 1))
        self.conv = nn.Conv2d(fan_in, fan_out, kernel_size=kernel_size)
        
    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)

class VAE(nn.Module):
    def __init__(self):
        super().__init__() 
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),   # 64x64x3  -> 31x31x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 31x31x32 -> 14x14x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # 14x14x64 -> 6x6x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),# 6x6x128 -> 2x2x256
            nn.ReLU(),
            nn.Flatten()                                 # Flattens out to 1024 (2*2*256)
        )
        self.mu_dense = nn.Linear(2 * 2 * 256, 32)
        self.log_sigma_dense = nn.Linear(2 * 2 * 256, 32)
        
        self.z_to_convdense = nn.Linear(32, 1024)
        
        self.decoder = nn.Sequential(
            UpsampleThenConv(5, 1024, 128, 5),
            nn.ReLU(),
            UpsampleThenConv(13, 128, 64, 5),
            nn.ReLU(),
            UpsampleThenConv(30, 64, 32, 6),
            nn.ReLU(),
            UpsampleThenConv(64, 32, 3, 6),
            nn.Sigmoid()            
        )
    
    def forward(self, x, get_loss=False):
        encoder_out = self.encoder(x)

        mu = self.mu_dense(encoder_out)
        log_sigma = self.log_sigma_dense(encoder_out)
        sigma = torch.exp(log_sigma)
        
        z = torch.randn_like(sigma) * sigma + mu
        
        z = self.z_to_convdense(z).view(-1, 1024, 1, 1)
        out = self.decoder(z)
        
        if get_loss:
            recon_loss = F.mse_loss(out, x, reduction='sum') / x.size(0)
            
            kl_loss = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp(), dim=1).mean()
            loss = recon_loss + kl_loss
            return out, loss, z
        
        return out, z