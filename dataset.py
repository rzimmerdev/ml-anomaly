import os
from enum import Enum

import pandas as pd
import torch
from torch.utils.data import Dataset


class Labels(Enum):
    NORMAL = 0
    ANOMALY_1 = 1
    ANOMALY_2 = 2
    ANOMALY_3 = 3
    ANOMALY_4 = 4


class SeriesDataset(Dataset):
    label_map = {
        "Situação referencia index": 0,
        "Situação rolo 3 da esquerda levantado index": 1,
        "Situação rolo 3 da esquerda removido index": 2,
        "Situação rolo 3 da direita levantado index": 3,
        "Situação rolo 3 da direita removido index": 4,
    }

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # filename = <LABEL> <ID>.csv
        file_path = os.path.join(self.data_dir, self.file_list[idx])

        label = " ".join(self.file_list[idx].split('.')[0].split(' ')[:-1])
        y = torch.zeros(5).scatter_(0, torch.LongTensor([self.label_map[label]]), 1)

        data = pd.read_csv(file_path).values
        x = torch.FloatTensor(data)

        return x, y
