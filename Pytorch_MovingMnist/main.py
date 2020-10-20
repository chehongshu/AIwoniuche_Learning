import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import tqdm
valid_size = 0.2
batch_size = 64
shuffle_dataset = True
random_seed = 1222

if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# training set or testing set, val set
mnistdata = MovingMNISTdataset("./data/mnist_test_seq.npy")
train_size = int(0.8 * len(mnistdata))
test_size = len(mnistdata) - train_size
torch.manual_seed(torch.initial_seed())
train_dataset, test_dataset = random_split(mnistdata, [train_size, test_size])

num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# load training data in batches
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          sampler=train_sampler)

# load validation data in batches
valid_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          sampler=valid_sampler)

# load test data in batches
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size)

# """
# model
# loss
# optimizer
# """
# n_epochs = 200
# for epoch in range(1, n_epochs + 1):
#
#     t = tqdm(train_loader, leave=False, total=len(train_loader))
#
#     for data in t:
#         input = data[:, 0:10, ...].cuda()
#         # print("mean")
#         # print(np.mean(input.data.cpu().numpy()))
#         label = data[:, 10:20, ...].cuda()