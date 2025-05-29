import torch

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, X, p, v):
        self.X = X
        self.p = p
        self.v = v
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (
            self.X[idx].float(),    # input images
            self.p[idx].long(),     # policy labels
            self.v[idx].float()     # value labels
        )



