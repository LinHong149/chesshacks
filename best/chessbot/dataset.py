import torch
import os
import json
from torch.utils.data import Dataset
from .config import PROCESSED_DIR, MOVE_INDEX_PATH


class ChessDataset(Dataset):
    def __init__(self):
        with open(MOVE_INDEX_PATH) as f:
            self.move_index_map = json.load(f)

        # Load all batch files
        self.files = sorted([
            os.path.join(PROCESSED_DIR, f)
            for f in os.listdir(PROCESSED_DIR)
            if f.endswith(".pt")
        ])
        
        # Pre-compute total length by loading first batch to check structure
        self._total_length = None
        self._batch_cache = {}  # Cache loaded batches
        
    def __len__(self):
        if self._total_length is None:
            # Calculate total length by summing positions in all batches
            total = 0
            for file_path in self.files:
                if file_path not in self._batch_cache:
                    batch = torch.load(file_path)
                    self._batch_cache[file_path] = batch
                    if isinstance(batch, list):
                        total += len(batch)
                    else:
                        # Old format: single position per file
                        total += 1
                else:
                    batch = self._batch_cache[file_path]
                    if isinstance(batch, list):
                        total += len(batch)
                    else:
                        total += 1
            self._total_length = total
        return self._total_length

    def __getitem__(self, idx):
        # Find which batch file contains this index
        current_idx = 0
        for file_path in self.files:
            # Load batch if not cached
            if file_path not in self._batch_cache:
                batch = torch.load(file_path)
                self._batch_cache[file_path] = batch
            else:
                batch = self._batch_cache[file_path]
            
            # Check if this is a batched file (list) or old format (dict)
            if isinstance(batch, list):
                # Batched format
                if idx < current_idx + len(batch):
                    position = batch[idx - current_idx]
                    return position["board"], position["move"]
                current_idx += len(batch)
            else:
                # Old format: single position per file
                if idx == current_idx:
                    return batch["board"], batch["move"]
                current_idx += 1
        
        raise IndexError(f"Index {idx} out of range")