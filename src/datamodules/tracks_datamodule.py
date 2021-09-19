from typing import Optional, Tuple, List
import os
import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from random import randrange

from .datasets.track_dataset import TrackDataset
from ..tracks import SEQ_LEN, Track


def train_test_split(dataset: List[Track], split: float = 0.80) -> Tuple[List[Track], List[Track]]:
    train = []
    test = list(dataset)
    train_size = split * len(dataset)
    while len(train) < train_size:
        index = randrange(len(test))
        train.append(test.pop(index))
    return train, test


class TracksDataModule(LightningDataModule):
    """
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    train_tracks: List[Track] = []
    data_val: List[Track] = []
    batch_size: int
    num_workers: int
    pin_memory: bool

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, SEQ_LEN, 256)


    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        files = [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith('mp3') or f.endswith('wav')
        ]
        tracks = []
        for file_path in files:
            track = Track(**torch.load(file_path))
            track.segments = track.segments
            tracks.append(track)

        self.train_tracks, self.val_tracks = train_test_split(tracks)

    def train_dataloader(self):
        return DataLoader(
            dataset=TrackDataset(self.train_tracks, SEQ_LEN),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TrackDataset(self.val_tracks, SEQ_LEN),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
