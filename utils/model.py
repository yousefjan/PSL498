import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNN(nn.Module):
    def __init__(self, num_features=27):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # Reshape (batch_size, num_features, 1)
        x = x.view(x.size(0), x.size(1), 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()

        self.fc1 = nn.Linear(27, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


