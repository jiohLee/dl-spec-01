import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Bridge(nn.Module):
    def __init__(self, in_channels, bridge, upsample_padding=0, upsample_output_padding=0) -> None:
        super().__init__()

        out_channels = in_channels * 2

        self.down = ResidualConv(in_channels, out_channels, stride=2)
        self.bridge = bridge
        self.upsample = nn.ConvTranspose1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=upsample_padding,
            output_padding=upsample_output_padding,
        )
        self.up = ResidualConv(out_channels + in_channels, in_channels, stride=1)

    def forward(self, x):
        d = self.upsample(self.bridge(self.down(x)))
        cat = torch.concat([x, d], dim=1)

        return self.up(cat)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.At = nn.Linear(in_features=36, out_features=350, bias=False)

        self.input_layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, padding=1))

        s = Bridge(512, nn.Identity())
        s = Bridge(256, s)
        s = Bridge(128, s, upsample_padding=1, upsample_output_padding=1)
        self.backbone = Bridge(64, s)

        self.fc = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, y):
        ext = self.At(y)
        ext1 = F.relu(ext)
        y = self.input_layer(ext) + self.input_skip(ext)
        y = self.backbone(y)
        return self.fc(y) + ext1


def build_model(sensing_matrix_path=None) -> nn.Module:

    model = Model()

    if sensing_matrix_path:
        A = loadmat(sensing_matrix_path)["sensing_matrix"]
        T = torch.tensor(np.matmul(A.T, np.linalg.inv(np.matmul(A, A.T))), dtype=torch.float32)
        model.At.weight = nn.Parameter(T)
        # model.At.weight.requires_grad = False

    return model


def main():
    device = torch.device("cuda", index=0)
    model = build_model("./sensing_matrix.mat").to(device)
    summary(model, input_size=(10, 1, 36), device=device)


if __name__ == "__main__":
    main()


"""
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
Model                                                   [10, 1, 350]              --
├─Linear: 1-1                                           [10, 1, 350]              12,600
├─Sequential: 1-2                                       [10, 64, 350]             --
│    └─Conv1d: 2-1                                      [10, 64, 350]             256
│    └─BatchNorm1d: 2-2                                 [10, 64, 350]             128
│    └─ReLU: 2-3                                        [10, 64, 350]             --
│    └─Conv1d: 2-4                                      [10, 64, 350]             12,352
├─Sequential: 1-3                                       [10, 64, 350]             --
│    └─Conv1d: 2-5                                      [10, 64, 350]             256
├─Bridge: 1-4                                           [10, 64, 350]             --
│    └─ResidualConv: 2-6                                [10, 128, 175]            --
│    │    └─Sequential: 3-1                             [10, 128, 175]            74,368
│    │    └─Sequential: 3-2                             [10, 128, 175]            24,960
│    └─Bridge: 2-7                                      [10, 128, 175]            --
│    │    └─ResidualConv: 3-3                           [10, 256, 88]             395,264
│    │    └─Bridge: 3-4                                 [10, 256, 88]             17,390,848
│    │    └─ConvTranspose1d: 3-5                        [10, 256, 175]            131,328
│    │    └─ResidualConv: 3-6                           [10, 128, 175]            345,728
│    └─ConvTranspose1d: 2-8                             [10, 128, 350]            32,896
│    └─ResidualConv: 2-9                                [10, 64, 350]             --
│    │    └─Sequential: 3-7                             [10, 64, 350]             49,792
│    │    └─Sequential: 3-8                             [10, 64, 350]             37,056
├─Conv1d: 1-5                                           [10, 1, 350]              65
=========================================================================================================
Total params: 18,507,897
Trainable params: 18,507,897
Non-trainable params: 0
Total mult-adds (G): 8.91
=========================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 122.29
Params size (MB): 74.03
Estimated Total Size (MB): 196.32
=========================================================================================================
"""
