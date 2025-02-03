import numpy as np
import torch as th
import os
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from torch.utils.data import (
    Dataset,
    DataLoader
)

class SSPCDataset(Dataset):

    def __init__(
        self,
        source_folder: str,
        split: str
    ):

        super().__init__()
        velodyne_unlabeled = os.path.join(
            source_folder,
            split,
            "velodyne_unlabeled"
        )
        self.paths_ = [
            os.path.join(velodyne_unlabeled, path)
            for path in os.listdir(velodyne_unlabeled)
        ]
    
    def __len__(self) -> None:
        return len(self.paths_)

    def __getitem__(self, idx: int) -> tuple[th.Tensor]:

        unlabeled_path = self.paths_[idx]
        labeled_path = unlabeled_path.replace("velodyne_unlabeled", "velodyne_labeled")
        
        return (
            th.load(unlabeled_path, weights_only=True),
            th.load(labeled_path, weights_only=True)
        )
        
        




