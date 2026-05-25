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
    def __init__(self, latent_dim=32):
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
        self.mu_dense = nn.Linear(2 * 2 * 256, latent_dim)
        self.log_sigma_dense = nn.Linear(2 * 2 * 256, latent_dim)
        
        self.z_to_convdense = nn.Linear(latent_dim, 1024)
        
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
        z, mu, log_sigma, z_flat = self.forward_z(x)
        
        out = self.decoder(z)
        
        if get_loss:
            recon_loss = F.mse_loss(out, x, reduction='sum') / x.size(0)
            
            kl_loss = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp(), dim=1).mean()
            loss = recon_loss + kl_loss
            return out, loss, z_flat
        
        return out, z_flat
    
    def forward_z(self, x):
        encoder_out = self.encoder(x)

        mu = self.mu_dense(encoder_out)
        log_sigma = self.log_sigma_dense(encoder_out)
        sigma = torch.exp(log_sigma)
        
        z = torch.randn_like(sigma) * sigma + mu
        
        # Flattened z for RNN/Controller
        z_flat = z.clone()
        
        z = self.z_to_convdense(z).view(-1, 1024, 1, 1)
        return z, mu, log_sigma, z_flat
        
        
    
class MDN_RNN(nn.Module):
    def __init__(self, action_dim, latent_dim=32, hidden_dim=256, num_mixtures=5, tau=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        self.hidden_dim = hidden_dim
        
        self.tau = tau
        
        self.lstm = nn.LSTM(action_dim + latent_dim, hidden_dim, batch_first=True)
        
        self.fc_mu = nn.Linear(hidden_dim, num_mixtures * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_mixtures * latent_dim)
        self.fc_pi = nn.Linear(hidden_dim, num_mixtures)
        
        self.fc_reward = nn.Linear(hidden_dim, 1)
        self.fc_done = nn.Linear(hidden_dim, 1) 
        
    def forward(self, z, a, h=None):
        # z: (batch, seq, latent_dim)
        # a: (batch, seq, action_dim)
        inputs = torch.cat([z, a], dim=-1)
        
        y, h = self.lstm(inputs, h)
        
        pi_logits = self.fc_pi(y)
        mu = self.fc_mu(y)
        sigma = self.fc_sigma(y)
        
        reward = self.fc_reward(y)
        done = self.fc_done(y)
        
        # Reshape to (batch, seq, num_mixtures, latent_dim)
        mu = mu.view(-1, y.size(1), self.num_mixtures, self.latent_dim)
        sigma = torch.exp(sigma).view(-1, y.size(1), self.num_mixtures, self.latent_dim)
        pi = F.softmax(pi_logits, dim=-1)
        
        return pi, mu, sigma, h, pi_logits, reward, done
    
    def sample(self, z, a, h=None, tau=1.0):
        
        pi, mu, sigma, h, pi_logits, reward, done = self.forward(z, a, h)
        
        # Fetch last step only
        pi_logits = pi_logits[:,-1,:]
        mu = mu[:, -1, :]
        sigma = sigma[:, -1, :]
        
        if tau == 0:
            pi_dist = torch.argmax(pi_logits, dim=-1)
            
        else:
            pi_scaled = F.softmax(pi_logits / tau, dim =-1)
            categorical = torch.distributions.Categorical(probs = pi_scaled)
            pi_dist = categorical.sample()
            
        batch_size = mu.size(0)
        batch_indices = torch.arange(0, batch_size)
        
        mu_chosen = mu[batch_indices, pi_dist, :]
        sigma_chosen = sigma[batch_indices, pi_dist, :] * tau
        
        if tau == 0:
            z_next = mu_chosen
        else:
            epsilon = torch.randn_like(sigma_chosen)
            z_next = mu_chosen + sigma_chosen * epsilon
            
        return z_next, h

class Controller(nn.Module):
    def __init__(self, action_dim, latent_dim, hidden_dim, action_type="continuous"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, action_dim),
            nn.Tanh()
        ) 
            
    def forward(self, z, h):
        # z: (batch, latent_dim)
        # h: (batch, hidden_dim) - the hidden state (h_n) of LSTM
        inp = torch.cat([z, h], dim=-1)
        return self.net(inp)

