import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class MyDataset(Dataset):

    def __init__(self, path, input_seqlen=10, output_seqlen=1, fea_num=1, train_precent=0.8, isTrain=True):
        data_df = pd.read_csv(path)
        Temp = data_df['Temp'].values

        self.data_num = len(Temp)
        self.input_seqlen = input_seqlen
        self.output_seqlen = output_seqlen
        self.fea_num = fea_num
        self.all_seqlen = self.input_seqlen + self.output_seqlen
        self.train_index = int(self.data_num*train_precent)

        self.data_seq = []
        self.target_seq = []

        for i in range(self.data_num - self.all_seqlen):
            self.data_seq.append(list(Temp[i:i + self.input_seqlen]))
            self.target_seq.append(list(Temp[i + self.input_seqlen: i + self.all_seqlen]))

        if isTrain is True:
            self.data_seq = self.data_seq[:self.train_index]
            self.target_seq = self.target_seq[:self.train_index]

        else:
            self.data_seq = self.data_seq[self.train_index:]
            self.target_seq = self.target_seq[self.train_index:]

        self.data_seq = np.array(self.data_seq).reshape((len(self.data_seq), -1, fea_num))
        self.target_seq = np.array(self.target_seq).reshape((len(self.target_seq), -1, fea_num))

        self.data_seq = torch.from_numpy(self.data_seq).type(torch.float32)
        self.target_seq = torch.from_numpy(self.target_seq).type(torch.float32)

    def __getitem__(self, index):
        return self.data_seq[index], self.target_seq[index]

    def __len__(self):
        return len(self.data_seq)

class RegrNet(nn.Module):
    def __init__(self):
        super(RegrNet, self).__init__()
        self.hidden1 = nn.LSTM(input_size=1, hidden_size=256, num_layers=2)
        self.hidden2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.hidden1(x)
        out = out[-1:]
        out = out.reshape(-1, 256)
        out = self.hidden2(out)
        out = self.out(out)
        out = out.reshape(1, -1, 1)
        return out

# parameters
path = './data/daily-min-temperatures.csv'
INPUT_SEQLEN = 90
OUTPUT_SEQLEN = 1
EPOCH = 50
LR = 0.001

# dataset instantiation
mydata = MyDataset(path, input_seqlen=INPUT_SEQLEN, output_seqlen=OUTPUT_SEQLEN)

# input, target = mydata.__getitem__(2)

# dataloader
data_loader = DataLoader(dataset=mydata,
                          batch_size=64,
                          shuffle=True)


model = RegrNet().cuda()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
mse = torch.nn.MSELoss()

writer = SummaryWriter()

for epoch in range(EPOCH):
    # print('epoch {}'.format(epoch + 1))
    # training process
    train_loss = 0.
    train_acc = 0.
    model.train()
    for i, (batch_x, batch_y) in enumerate(data_loader):
        #print(batch_x.shape)
        #print(batch_y.shape)
        batch_x = batch_x.permute(1, 0, 2).cuda()
        batch_y = batch_y.permute(1, 0, 2).cuda()
        out = model(batch_x)
        #print(out.shape)
        loss = mse(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar("loss_train", loss.item(), epoch)
    print('epoch : {}, Train Loss: {:.6f}'.format(epoch, loss.item()))

writer.close()