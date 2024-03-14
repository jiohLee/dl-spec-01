import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.At = nn.Linear(in_features=36, out_features=350, bias=False)

        def conv_block(in_channels, out_channels, kernel_size, pool_window):

            return [
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_window, stride=pool_window),
            ]

        def fc_block(in_features, out_features):
            return [
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.ReLU(),
            ]

        self.conv_layers = nn.Sequential(
            *conv_block(1, 128, 3, 2),
            *conv_block(128, 128, 3, 2),
            *conv_block(128, 256, 3, 2),
            *conv_block(256, 128, 3, 2),
            *conv_block(128, 128, 3, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            *fc_block(128 * 9, 512),
        )

        self.out = nn.Linear(512, 350)

    def forward(self, y):

        ext = self.At(y)

        x = self.conv_layers(ext)
        x = self.fc_layers(x)

        return self.out(x).unsqueeze(1)


def build_model(sensing_matrix_path=None) -> nn.Module:

    model = Model()

    if sensing_matrix_path:
        A = loadmat(sensing_matrix_path)["sensing_matrix"]
        T = torch.tensor(np.matmul(A.T, np.linalg.inv(np.matmul(A, A.T))), dtype=torch.float32)
        model.At.weight = nn.Parameter(T)

    return model


def main():

    device = torch.device("cuda", index=0)
    model = build_model("./sensing_matrix.mat").to(device)
    summary(model, input_size=(10, 1, 36), device=device)


if __name__ == "__main__":
    main()


"""
==========================================================================================
(base) root@ac87b8639f7d:~/spec/models# python cnn.py    
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [10, 1, 350]              --
├─Linear: 1-1                            [10, 1, 350]              12,600
├─Sequential: 1-2                        [10, 128, 9]              --
│    └─Conv1d: 2-1                       [10, 128, 348]            512
│    └─ReLU: 2-2                         [10, 128, 348]            --
│    └─MaxPool1d: 2-3                    [10, 128, 174]            --
│    └─Conv1d: 2-4                       [10, 128, 172]            49,280
│    └─ReLU: 2-5                         [10, 128, 172]            --
│    └─MaxPool1d: 2-6                    [10, 128, 86]             --
│    └─Conv1d: 2-7                       [10, 256, 84]             98,560
│    └─ReLU: 2-8                         [10, 256, 84]             --
│    └─MaxPool1d: 2-9                    [10, 256, 42]             --
│    └─Conv1d: 2-10                      [10, 128, 40]             98,432
│    └─ReLU: 2-11                        [10, 128, 40]             --
│    └─MaxPool1d: 2-12                   [10, 128, 20]             --
│    └─Conv1d: 2-13                      [10, 128, 18]             49,280
│    └─ReLU: 2-14                        [10, 128, 18]             --
│    └─MaxPool1d: 2-15                   [10, 128, 9]              --
├─Sequential: 1-3                        [10, 512]                 --
│    └─Flatten: 2-16                     [10, 1152]                --
│    └─Linear: 2-17                      [10, 512]                 590,336
│    └─ReLU: 2-18                        [10, 512]                 --
├─Linear: 1-4                            [10, 350]                 179,550
==========================================================================================
Total params: 1,078,550
Trainable params: 1,078,550
Non-trainable params: 0
Total mult-adds (M): 225.40
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 7.74
Params size (MB): 4.31
Estimated Total Size (MB): 12.05
==========================================================================================
"""
