import torch
from torch.utils.data import Dataset
import h5py

class MLMDataset(Dataset):
    def __init__(self, path, device=torch.device('cpu')):
        self.path = path
        self.device = device

        with h5py.File(path, 'r') as f:
            self.dataset_names = list(f.keys())
            self.length = len(f[self.dataset_names[0]])
            for name in self.dataset_names:
                if f[name].shape[0] != self.length:
                    raise ValueError('datasets must have the same length')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as f:
            data, label, mask =  (torch.tensor(f[name][idx], device=self.device) for name in self.dataset_names)
            return {
                'x': data,
                'label_ids': label,
            }