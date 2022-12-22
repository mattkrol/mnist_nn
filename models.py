import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
