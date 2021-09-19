from typing import List
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from ...tracks import Track

class TrackDataset(Dataset):
    tracks: List[Track]
    seq_len: int

    def __init__(self, tracks: List[Track], seq_len: int):
        super().__init__()
        self.tracks = tracks
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index: int):
        track = self.tracks[index]
        i = random.randint(0, len(track.segments) - 1)
        x = track.segments[i].spec.T[:self.seq_len]
        if x.shape[0] < self.seq_len:
            x = np.pad(x, [(0, self.seq_len - x.shape[0]), (0, 0)], mode='linear_ramp')
        return torch.tensor(x).unsqueeze(0)
