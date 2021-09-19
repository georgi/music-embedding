from typing import List
from torch.utils.data import Dataset
import numpy as np
import torch
from ...tracks import Track, Segment

class TrackDataset(Dataset):
    segments: List[Segment]
    seq_len: int

    def __init__(self, tracks: List[Track], seq_len: int):
        super().__init__()
        self.segments = [s for t in tracks for s in t.segments]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index: int):
        seg = self.segments[index]
        x = seg.spec.T[:self.seq_len]
        if x.shape[0] < self.seq_len:
            x = np.pad(x, [(0, self.seq_len - x.shape[0]), (0, 0)], mode='linear_ramp')
        return torch.tensor(x).unsqueeze(0)
