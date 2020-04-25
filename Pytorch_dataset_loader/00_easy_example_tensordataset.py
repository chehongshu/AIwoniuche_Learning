from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import torch


if __name__ == '__main__':
    data_df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')

    feature_col =  ['No', 'year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    data_df_x = data_df.loc[:127,feature_col]
    label_col = ['pm2.5']
    data_df_y = data_df.loc[:127, label_col]

    data_numpy_x = data_df_x.values
    data_numpy_y = data_df_y.values

    X = torch.from_numpy(data_numpy_x)
    Y = torch.from_numpy(data_numpy_y)

    dataset = TensorDataset(X, Y)

    # if use num_workers ,you need "if __name__ == '__main__':"
    data_loader = DataLoader(dataset=dataset,
              batch_size=64,
              shuffle=True,
              num_workers=2)
    
    for i, data in enumerate(data_loader):
        print(i)
        x, y = data
        print(type(x), type(y))
        print(x.data.size(), y.data.size())