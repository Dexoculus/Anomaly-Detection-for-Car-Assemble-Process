import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class ActionDataset(Dataset):
    """
    A dataset of Robot Arm sensor data.
    
    This dataset assumes that the data are stored in a directory structure like:
    root_dir/
        OK/
            ... .xls files ...
        NG/
            NG_+Z_inner_path/
                ... .xls files ...
            NG_-X_Y_offset/
                ... .xls files ...
            NG_-Z_outer_path/
                ... .xls files ...
            NG_X_-Y_offset/
                ... .xls files ...
    """

    def __init__(self, root_dir, seq_length=None, processing=None,
                 mode='train', train_ratio=0.7, valid_ratio=0.15, seed=42):
        """
        Args:
            root_dir (str) : Root directory path of Dataset.
            seq_length (int, optional) : Constant length of time series data.
            processing (callable, optional) : Preprocessing func for data.
            mode (str) : 'train', 'valid', or 'test'
            train_ratio (float) : Ratio of training data from total
            valid_ratio (float) : Ratio of validation data from total
            seed (int) : random seed for reproducibility
        """
        self.seq_length = seq_length
        self.processing = processing
        self.mode = mode
        
        self.label_mapping = {
            'OK': 0,
            'NG': 1,
            'NG_+Z_inner_path': 2,
            'NG_-X_Y_offset': 3,
            'NG_-Z_outer_path': 4,
            'NG_X_-Y_offset': 5
        }

        # Collect all files and labels
        all_file_paths = []
        all_labels = []

        for root, dirs, files in os.walk(root_dir):
            relative_path = os.path.relpath(root, root_dir)
            path_parts = relative_path.split(os.sep)

            # Skip top-level or hidden directories
            if len(path_parts) < 1 or path_parts[0] == '.':
                continue

            class_name = path_parts[0]
            label = self.label_mapping.get(class_name)
            if label is None:
                continue

            for file in files:
                if file.endswith('.xls') or file.endswith('.xlsx'):
                    file_path = os.path.join(root, file)
                    all_file_paths.append(file_path)
                    all_labels.append(label)

        # Convert to numpy for easy indexing
        all_file_paths = np.array(all_file_paths)
        all_labels = np.array(all_labels)

        # Shuffle before splitting
        np.random.seed(seed)
        perm = np.random.permutation(len(all_file_paths))
        all_file_paths = all_file_paths[perm]
        all_labels = all_labels[perm]

        # Compute split indices
        total = len(all_file_paths)
        train_end = int(total * train_ratio)
        valid_end = int(total * (train_ratio + valid_ratio))

        if self.mode == 'train':
            self.file_paths = all_file_paths[:train_end]
            self.labels = all_labels[:train_end]
        elif self.mode == 'valid':
            self.file_paths = all_file_paths[train_end:valid_end]
            self.labels = all_labels[train_end:valid_end]
        elif self.mode == 'test':
            self.file_paths = all_file_paths[valid_end:]
            self.labels = all_labels[valid_end:]
        else:
            raise ValueError("mode should be one of 'train', 'valid', or 'test'")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Read the Excel file, skipping first 13 rows
        try:
            df = pd.read_excel(file_path, skiprows=13, header=None)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            raise

        df = df.drop(df.columns[0], axis=1)

        # Convert DataFrame to numpy array
        data = df.values

        # If seq_length is specified, pad or truncate
        if self.seq_length is not None:
            data = self.pad_or_truncate(data, self.seq_length)

        # Convert to torch tensor
        data = torch.tensor(data, dtype=torch.float32)

        # Apply processing function if provided
        if self.processing:
            data = self.processing(data)

        return data, torch.tensor(label, dtype=torch.long)

    def pad_or_truncate(self, data, length):
        """
        Pads or truncates the time series data to a fixed length.
        """
        if data.shape[0] > length:
            # Truncate
            data = data[:length, :]
        elif data.shape[0] < length:
            # Pad with zeros
            padding = np.zeros((length - data.shape[0], data.shape[1]))
            data = np.vstack((data, padding))
        return data


class ImageDataset(Dataset):
    """
    To be developed...
    """
    def __init__(self):
        """
        """
        ng_path = 'data/robot_sticker/ri_ng'
        ok_path = 'data/robot_sticker/ri_ok'