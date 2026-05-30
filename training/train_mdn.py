import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from torch import optim

import glob
import os

from wm.modeling import MDN_RNN
from wm.env import make_env

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3



class LatentDataset(Dataset):
    def __init__(self, path_encoded='data_z/'):
        super().__init__()
        self.filepaths = glob.glob(os.path.join(path_encoded, '*.pt'))
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index):
        # Load the dictionary of tensors
        data = torch.load(self.filepaths[index], weights_only=True)
        return data['z'], data['action'], data['reward'], data['done']

if __name__ == '__main__':
    ENV_NAME = "VizdoomTakeCover-v1" # Can swap to CarRacing-v2 or others
    
    # Figure out the action space dynamically
    dummy_env = make_env(ENV_NAME)
    if hasattr(dummy_env.action_space, 'n'):
        ACTION_DIM = dummy_env.action_space.n
        IS_DISCRETE = True
    else:
        ACTION_DIM = dummy_env.action_space.shape[0]
        IS_DISCRETE = False
    dummy_env.close()

    dataset = LatentDataset(path_encoded='data_z/')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    mdn_rnn = MDN_RNN(action_dim=ACTION_DIM).to(DEVICE)
    
    optimizer = optim.AdamW(mdn_rnn.parameters(), lr=LEARNING_RATE)
    # Training loop for MDN-RNN goes here
    for i in range(NUM_EPOCHS):
        for e, (z, actions, rewards, dones) in enumerate(dataloader):
            z_in = z[:, :-1, :].to(DEVICE)
            
            # act_in shape starts as (Batch, Time)
            act_in = actions[:, :-1].to(DEVICE)

            # Convert to one-hot if it's a discrete button environment, else use floats directly
            if IS_DISCRETE:
                act_in = F.one_hot(act_in.long(), num_classes=ACTION_DIM).float()
            else:
                act_in = act_in.float()

            z_targ = z[:, 1:, :].to(DEVICE)
            rew_targ = rewards[:, 1:].to(DEVICE).unsqueeze(-1)
            dones_targ = dones[:, 1:].to(DEVICE).unsqueeze(-1).to(torch.float32)
            
            
            pi, mu, sigma, h, pi_logits, pred_reward, pred_done = mdn_rnn.forward(z_in, act_in)

            loss_reward = F.mse_loss(pred_reward, rew_targ)
            
            loss_dones = F.binary_cross_entropy_with_logits(pred_done, dones_targ)

            z_reshaped = z_targ.unsqueeze(2)
            dist = torch.distributions.Normal(mu, sigma)

            log_pi = F.log_softmax(pi_logits, dim=-1)
            
            log_probs = dist.log_prob(z_reshaped).sum(dim=-1)
            
            loss_mdn = -torch.logsumexp(log_pi + log_probs, dim=2).mean()
            

            loss = loss_reward + loss_dones + loss_mdn
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
        