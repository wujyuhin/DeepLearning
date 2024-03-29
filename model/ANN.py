import torch
import torch.nn as nn
class ANN(nn.model):
    def __init__(self,input_dim,output_dim):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(input_dim,128)
        self.relu1 = nn.ReLU(128,128)
        self.fc2 = nn.Linear(128,output_dim)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x