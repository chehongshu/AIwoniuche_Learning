from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np


class MyData_Animal(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_path_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, index):
        image_index = self.images_path_list[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        img = img.resize((64, 64))
        label = img_path.split('\\')[-1].split('.')[0]

        if self.transform:
            img = self.transform(img)

        return img, label


transform = transforms.Compose([
#     transforms.Resize(64),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
])

mydataset_animal = MyData_Animal('./train', transform)

data_loader3 = DataLoader(dataset=mydataset_animal,
                          batch_size=4,
                          shuffle=True)



for i, (img, label) in enumerate(data_loader3):
    print(i)
    print(type(img), type(label))
    print(img.data.size(), label)