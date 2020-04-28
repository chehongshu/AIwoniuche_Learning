import torch
import torchvision
import torch.utils.data.dataloader as Data
import torch.nn as nn

train_data = torchvision.datasets.MNIST(
    './mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.MNIST(
    './mnist', train=False, transform=torchvision.transforms.ToTensor()
)


train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64)


class ClasNet(nn.Module):
    def __init__(self):
        super(ClasNet, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        self.output = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.hidden(x)
        out = self.output(x)
        return out


model = ClasNet()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_func = nn.CrossEntropyLoss()


for epoch in range(35):
    print('epoch {}'.format(epoch + 1))
    # training process
    train_loss = 0.
    train_acc = 0.
    model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        # batch_x = batch_x.cuda()
        # batch_y = batch_y.cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            train_data)), train_acc / (len(train_data))))

    # eval process
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for j,(batch_x, batch_y) in enumerate(test_loader):
        # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()

    print('eval Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))

# writer.close()