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


class CnnCifar10(nn.Module):

    def __init__(self):
        super(CnnCifar10, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)

        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, 10)

        self.batchnorm64_1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.batchnorm128_1 = nn.BatchNorm2d(128, track_running_stats=False)
        self.batchnorm128_2 = nn.BatchNorm2d(128, track_running_stats=False)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        out1 = self.conv1(x)
        activ1 = F.relu(out1)
        batch1 = self.dropout(activ1)

        out2 = self.conv2(batch1)
        activ2 = F.relu(out2)
        batch2 = self.batchnorm64_1(activ2)
        max_pool2 = F.max_pool2d(batch2, kernel_size=2)
        dropout2 = self.dropout(max_pool2)

        out3 = self.conv3(dropout2)
        activ3 = F.relu(out3)
        batch3 = self.batchnorm128_1(activ3)

        out4 = self.conv4(batch3)
        activ4 = F.relu(out4)
        batch4 = self.batchnorm128_2(activ4)
        max_pool4 = F.max_pool2d(batch4, kernel_size=2)
        dropout4 = self.dropout(max_pool4)

        flattened = torch.flatten(dropout4, 1)

        linear1 = self.fc1(flattened)
        activ5 = F.relu(linear1)
        dropout5 = self.dropout(activ5)

        linear2 = self.fc2(dropout5)

        return F.softmax(linear2, dim=1)
