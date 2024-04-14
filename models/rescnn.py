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
                nn.MaxPool1d(kernel_size=pool_window, stride=pool_window),
                nn.ReLU(),
            ]

        def fc_block(in_features, out_features):
            return [
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.ReLU(),
            ]

        self.conv_layers = nn.Sequential(
            *conv_block(1, 32, 27, 2),
            *conv_block(32, 64, 18, 2),
            *conv_block(64, 128, 9, 2),
            *conv_block(128, 256, 3, 2),
            *conv_block(256, 512, 3, 3),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            *fc_block(2048, 1024),
            *fc_block(1024, 512),
            *fc_block(512, 350),
        )

        self.out = nn.Linear(350, 350)

    def forward(self, y):
        # y: [N, 1, 36]
        ext = self.At(y)

        x = self.conv_layers(ext)
        x = self.fc_layers(x)

        return self.out(x).unsqueeze(1) + ext
        return F.relu(self.out(x).unsqueeze(1) + ext)


def build_model(sensing_matrix_path=None) -> nn.Module:

    model = Model()

    if sensing_matrix_path:
        T = loadmat(sensing_matrix_path)["sensing_matrix"]
        A = torch.tensor(np.matmul(T.T, np.linalg.inv(np.matmul(T, T.T))), dtype=torch.float32)
        model.At.weight = nn.Parameter(A)

    return model


def main():

    device = torch.device("cuda", index=0)
    model = build_model("./sensing_matrix.mat").to(device)
    summary(model, input_size=(10, 1, 36), device=device)


if __name__ == "__main__":
    main()


"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [10, 1, 350]              --
├─Linear: 1-1                            [10, 1, 350]              12,600
├─Sequential: 1-2                        [10, 512, 4]              --
│    └─Conv1d: 2-1                       [10, 32, 324]             896
│    └─MaxPool1d: 2-2                    [10, 32, 162]             --
│    └─ReLU: 2-3                         [10, 32, 162]             --
│    └─Conv1d: 2-4                       [10, 64, 145]             36,928
│    └─MaxPool1d: 2-5                    [10, 64, 72]              --
│    └─ReLU: 2-6                         [10, 64, 72]              --
│    └─Conv1d: 2-7                       [10, 128, 64]             73,856
│    └─MaxPool1d: 2-8                    [10, 128, 32]             --
│    └─ReLU: 2-9                         [10, 128, 32]             --
│    └─Conv1d: 2-10                      [10, 256, 30]             98,560
│    └─MaxPool1d: 2-11                   [10, 256, 15]             --
│    └─ReLU: 2-12                        [10, 256, 15]             --
│    └─Conv1d: 2-13                      [10, 512, 13]             393,728
│    └─MaxPool1d: 2-14                   [10, 512, 4]              --
│    └─ReLU: 2-15                        [10, 512, 4]              --
├─Sequential: 1-3                        [10, 350]                 --
│    └─Flatten: 2-16                     [10, 2048]                --
│    └─Linear: 2-17                      [10, 1024]                2,098,176
│    └─ReLU: 2-18                        [10, 1024]                --
│    └─Linear: 2-19                      [10, 512]                 524,800
│    └─ReLU: 2-20                        [10, 512]                 --
│    └─Linear: 2-21                      [10, 350]                 179,550
│    └─ReLU: 2-22                        [10, 350]                 --
├─Linear: 1-4                            [10, 350]                 122,850
==========================================================================================
Total params: 3,541,944
Trainable params: 3,541,944
Non-trainable params: 0
Total mult-adds (M): 213.85
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 3.58
Params size (MB): 14.17
Estimated Total Size (MB): 17.75
==========================================================================================
"""
