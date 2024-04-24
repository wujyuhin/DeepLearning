<<<<<<< HEAD
import torch
import torch.nn as nn
import pytorch_lightning as pl
=======
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

# 调用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>>>>>> dad1c1502c4a8901b4ffa4702e0e6a35af4993da


class ANNmodel(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(ANNmodel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        # softmax
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # 分类
        x = self.softmax(x)
        return x

<<<<<<< HEAD
=======

>>>>>>> dad1c1502c4a8901b4ffa4702e0e6a35af4993da
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
<<<<<<< HEAD
=======
        # print(y_hat.device)
>>>>>>> dad1c1502c4a8901b4ffa4702e0e6a35af4993da
        loss = self.loss(y_hat, y)
        return loss
