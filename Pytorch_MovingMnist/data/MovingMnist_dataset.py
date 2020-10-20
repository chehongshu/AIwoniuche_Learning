import numpy as np
from torch.utils.data import Dataset
import torch

def MNISTdataLoader(path):
    # load moving mnist data, data shape = [time steps, batch size, width, height] = [20, batch_size, 64, 64]
    # B S H W -> S B H W
    data = np.load(path)
    data_trans = data.transpose(1, 0, 2, 3)
    return data_trans

class MovingMNISTdataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = MNISTdataLoader(path)

    def __len__(self):
        return len(self.data[:, 0, 0, 0])

    def __getitem__(self, indx):
        self.trainsample_ = self.data[indx, ...]
        # self.sample_ = self.trainsample_/255.0   # normalize
        self.sample_ = self.trainsample_
        self.sample = torch.from_numpy(np.expand_dims(self.sample_,  axis=1)).float()
        return self.sample

if __name__ == '__main__':
    mnistdata = MovingMNISTdataset("./mnist_test_seq.npy")
    print(mnistdata.__getitem__(0)[:10].shape)  # input
    print(mnistdata.__getitem__(0)[10:].shape)  # target
