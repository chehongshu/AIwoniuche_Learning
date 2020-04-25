from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch


class MyDataset(Dataset):

    def __init__(self):
        data_df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')
        feature_col = ['No', 'year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
        data_df_x = data_df.loc[:127, feature_col]
        label_col = ['pm2.5']
        data_df_y = data_df.loc[:127, label_col]
        data_numpy_x = data_df_x.values
        data_numpy_y = data_df_y.values
        self.X = torch.from_numpy(data_numpy_x)
        self.Y = torch.from_numpy(data_numpy_y)
        self.len = data_numpy_x.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


# dataset instantiation
dataset2 = MyDataset()

# dataloader
data_loader2 = DataLoader(dataset=dataset2,
                          batch_size=64,
                          shuffle=True)

# index
for i, data in enumerate(data_loader2):
    print(i)
    x, y = data
    print(type(x), type(y))
    print(x.data.size(), y.data.size())
