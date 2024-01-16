import torch
from torch import nn


class DqlNet(nn.Module):
    def __init__(self, img_size: tuple, out_channels: int):
        super(DqlNet, self).__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, out_channels)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
