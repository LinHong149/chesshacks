import torch
import os
import json
from torch.utils.data import Dataset
from .config import PROCESSED_DIR, MOVE_INDEX_PATH


class ChessDataset(Dataset):
    def __init__(self):
        with open(MOVE_INDEX_PATH) as f:
            self.move_index_map = json.load(f)

        self.files = sorted([
            os.path.join(PROCESSED_DIR, f)
            for f in os.listdir(PROCESSED_DIR)
            if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj = torch.load(self.files[idx])
        return obj["board"], obj["move"]