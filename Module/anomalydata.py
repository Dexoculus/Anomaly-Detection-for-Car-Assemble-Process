import numpy as np
import torch
import pandas as pd

from .dataset import ActionDataset

class AnomalDataset(ActionDataset):
    def __init__(self, root_dir, seq_length=None, processing=None, mode='train'):
        """
        A dataset for train Auto-Encoder and test anomaly detection.
        The AnomalDataset inherits from the ActionDataset class. 
        If train = True:
            Only use Ok (label=0) data for train Auto-Encoder model.
        else: (train = False)
        Use All OK/NG label data to test anomaly detection.
        
        Args:
            root_dir (str) : Root directory path of Dataset.
            seq_length (int, optional) : Constant length of time series data.
            processing (callable, optional) : Preprocessing func for data.
            mode (str) : 'train', or 'test'
        """
        super().__init__(root_dir, seq_length, processing)
        self.mode = mode

        if self.mode == 'train':
            # Filter Only OK(label=0) data
            mask = np.array(self.labels) == 0
            self.file_paths = np.array(self.file_paths)[mask].tolist()
            self.labels = np.array(self.labels)[mask].tolist()
        # else: Both OK/NG for Test

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]  # 0 or 1

        df = pd.read_excel(file_path, skiprows=13, header=None)
        df = df.drop(df.columns[0], axis=1)
        data = df.values  # shape: (sequence_len, num_features)

        if self.seq_length is not None:
            data = self.pad_or_truncate(data, self.seq_length)

        data = torch.tensor(data, dtype=torch.float32)

        if self.processing:
            data = self.processing(data)

        if self.mode == 'train':
            return data, data
        else:
            return data, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.file_paths)

