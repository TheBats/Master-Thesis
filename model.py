import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnMist(nn.Module):

    def __init__(self):
        super(CnnMist, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1, 0)
        self.conv2 = nn.Conv2d(20, 20, 5, 1, 0)
        self.fc1 = nn.Linear(320, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out1 = self.conv1(x)
        activ1 = F.relu(out1)
        max_pool1 = F.max_pool2d(activ1, kernel_size=2)

        out2 = self.conv2(max_pool1)
        activ2 = F.relu(out2)
        max_pool2 = F.max_pool2d(activ2, kernel_size=2)
        flattened = max_pool2.reshape(-1, 320)

        out3 = self.fc1(flattened)
        activ3 = F.relu(out3)

        out4 = self.fc2(activ3)

        return F.softmax(out4, dim=1)
