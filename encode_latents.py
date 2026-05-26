import torch
import numpy as np
import glob
import os
from wm.modeling import VAE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode_latents(vae, path_to_encode='data/', path_to_store='data_z/'):
    os.makedirs(path_to_store, exist_ok=True)
    filepaths = glob.glob(os.path.join(path_to_encode, 'rollout_*.npz'))
    vae.eval()
        
    for fn in filepaths:
        basename = os.path.basename(fn).replace('.npz', '.pt')
        save_path = os.path.join(path_to_store, basename)
        
        with np.load(fn) as data:
            obs = data['obs']
            action = data['action']
            reward = data['reward']
            done = data['done']
            
            # Normalize to [0, 1] and convert obs to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(DEVICE) / 255.0
            
            # Permute from (T, H, W, C) to (T, C, H, W)
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            
            with torch.no_grad():
                out, z_flat = vae(obs_tensor)

            # Move latents back to CPU and save along with other rollout data
            torch.save({
                'z': z_flat.cpu(),
                'action': torch.tensor(action),
                'reward': torch.tensor(reward),
                'done': torch.tensor(done)
            }, save_path)
            
        print(f"Encoded {fn} -> {save_path}")

if __name__ == '__main__':
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load("checkpoints/vae.pth", map_location=DEVICE, weights_only=True))
    
    encode_latents(vae, path_to_encode='data/', path_to_store='data_z/')
