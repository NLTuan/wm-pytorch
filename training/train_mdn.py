import torch
from torch.utils.data import DataLoader, Dataset
import glob
import os

from wm.modeling import MDN_RNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    dataset = LatentDataset(path_encoded='data_z/')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    mdn = MDN_RNN().to(DEVICE)
    
    # Training loop for MDN-RNN goes here
    pass