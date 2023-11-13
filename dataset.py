import os
import pandas as pd
from torch.utils.data import Dataset


class SeriesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = pd.read_csv(file_path).values  # Assuming CSV files have numeric values
        x = torch.FloatTensor(data[:, :-1])  # Input features
        y = torch.LongTensor(data[:, -1])  # Labels
        return x, y

