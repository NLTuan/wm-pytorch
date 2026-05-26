import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from modeling import VAE

# Configuration
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAEDataset(Dataset):
    def __init__(self, data_dir):
        file_paths = glob.glob(os.path.join(data_dir, "rollout_*.npz"))
        
        all_obs = []
        for path in file_paths:
            with np.load(path) as data:
                all_obs.append(data['obs'])
                
        self.images = np.concatenate(all_obs, axis=0)
        
    def __len__(self):
        return self.images.shape[0]
        
    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32) / 255.0

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    dataset = VAEDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = VAE(latent_dim=32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training VAE on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, x in enumerate(dataloader):
            x = x.to(DEVICE)
            
            out, loss, z_flat = model(x, get_loss=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "vae.pth"))

if __name__ == "__main__":
    train()
