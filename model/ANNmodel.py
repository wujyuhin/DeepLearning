import torch
import torch.nn as nn
# 调用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ANNmodel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANNmodel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        x = self.fc3(x)
        return x
