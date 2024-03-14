import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


class BasicConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=in_channels),
            BasicConv(in_channels=in_channels, out_channels=in_channels),
            nn.Conv1d(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding="same", padding_mode="reflect"
            ),
        )

        self.output_layers = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.output_layers(x + self.layers(x))


def block_down(in_channels, out_channels):
    return nn.Sequential(
        nn.MaxPool1d(kernel_size=2, stride=2),
        BasicConv(in_channels=in_channels, out_channels=out_channels),
        ResBlock(in_channels=out_channels),
    )


def connection(channels):
    return nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding="same", padding_mode="reflect")


def block_up(in_channels, out_channels, output_padding=1):
    return nn.Sequential(
        BasicConv(in_channels=in_channels * 2, out_channels=in_channels),
        ResBlock(in_channels=in_channels),
        nn.ConvTranspose1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, output_padding=output_padding
        ),
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.At = nn.Linear(in_features=36, out_features=350, bias=False)

        self.block_down_1 = nn.Sequential(BasicConv(1, 64), ResBlock(64))
        self.block_down_2 = block_down(in_channels=64, out_channels=128)
        self.block_down_3 = block_down(in_channels=128, out_channels=256)
        self.block_down_4 = block_down(in_channels=256, out_channels=512)
        self.block_down_5 = block_down(in_channels=512, out_channels=1024)

        self.connections = nn.ModuleList([connection(ch) for ch in [1, 64, 128, 256, 512, 1024]])

        self.bridge = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            BasicConv(1024, 1024),
            connection(1024),
            ResBlock(1024),
            nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, output_padding=1),
        )

        self.block_up_5 = block_up(1024, 512)
        self.block_up_4 = block_up(512, 256)
        self.block_up_3 = block_up(256, 128)
        self.block_up_2 = block_up(128, 64, output_padding=0)
        self.block_up_1 = nn.Sequential(
            BasicConv(64 * 2, 64),
            BasicConv(64, 1),
        )

        self.output_layer = BasicConv(1 * 2, 1)

    def forward(self, x):
        x = self.At(x)

        x1 = self.block_down_1(x)
        x2 = self.block_down_2(x1)
        x3 = self.block_down_3(x2)
        x4 = self.block_down_4(x3)
        x5 = self.block_down_5(x4)

        x_ = self.bridge(x5)

        x_ = self.block_up_5(torch.cat([x_, self.connections[5](x5)], dim=1))
        x_ = self.block_up_4(torch.cat([x_, self.connections[4](x4)], dim=1))
        x_ = self.block_up_3(torch.cat([x_, self.connections[3](x3)], dim=1))
        x_ = self.block_up_2(torch.cat([x_, self.connections[2](x2)], dim=1))
        x_ = self.block_up_1(torch.cat([x_, self.connections[1](x1)], dim=1))

        return self.output_layer(torch.cat([x_, self.connections[0](x)], dim=1))


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

    # num_Conv1d = sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))
    # num_ConvTranspose1d = sum(1 for m in model.modules() if isinstance(m, nn.ConvTranspose1d))
    # print(num_Conv1d, num_ConvTranspose1d)


if __name__ == "__main__":
    main()


"""
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
Model                                              [10, 1, 350]              --
├─Linear: 1-1                                      [10, 1, 350]              12,600
├─Sequential: 1-2                                  [10, 64, 350]             --
│    └─BasicConv: 2-1                              [10, 64, 350]             --
│    │    └─Sequential: 3-1                        [10, 64, 350]             384
│    └─ResBlock: 2-2                               [10, 64, 350]             --
│    │    └─Sequential: 3-2                        [10, 64, 350]             37,312
│    │    └─Sequential: 3-3                        [10, 64, 350]             128
├─Sequential: 1-3                                  [10, 128, 175]            --
│    └─MaxPool1d: 2-3                              [10, 64, 175]             --
│    └─BasicConv: 2-4                              [10, 128, 175]            --
│    │    └─Sequential: 3-4                        [10, 128, 175]            24,960
│    └─ResBlock: 2-5                               [10, 128, 175]            --
│    │    └─Sequential: 3-5                        [10, 128, 175]            148,352
│    │    └─Sequential: 3-6                        [10, 128, 175]            256
├─Sequential: 1-4                                  [10, 256, 87]             --
│    └─MaxPool1d: 2-6                              [10, 128, 87]             --
│    └─BasicConv: 2-7                              [10, 256, 87]             --
│    │    └─Sequential: 3-7                        [10, 256, 87]             99,072
│    └─ResBlock: 2-8                               [10, 256, 87]             --
│    │    └─Sequential: 3-8                        [10, 256, 87]             591,616
│    │    └─Sequential: 3-9                        [10, 256, 87]             512
├─Sequential: 1-5                                  [10, 512, 43]             --
│    └─MaxPool1d: 2-9                              [10, 256, 43]             --
│    └─BasicConv: 2-10                             [10, 512, 43]             --
│    │    └─Sequential: 3-10                       [10, 512, 43]             394,752
│    └─ResBlock: 2-11                              [10, 512, 43]             --
│    │    └─Sequential: 3-11                       [10, 512, 43]             2,362,880
│    │    └─Sequential: 3-12                       [10, 512, 43]             1,024
├─Sequential: 1-6                                  [10, 1024, 21]            --
│    └─MaxPool1d: 2-12                             [10, 512, 21]             --
│    └─BasicConv: 2-13                             [10, 1024, 21]            --
│    │    └─Sequential: 3-13                       [10, 1024, 21]            1,575,936
│    └─ResBlock: 2-14                              [10, 1024, 21]            --
│    │    └─Sequential: 3-14                       [10, 1024, 21]            9,444,352
│    │    └─Sequential: 3-15                       [10, 1024, 21]            2,048
├─Sequential: 1-7                                  [10, 1024, 21]            --
│    └─MaxPool1d: 2-15                             [10, 1024, 10]            --
│    └─BasicConv: 2-16                             [10, 1024, 10]            --
│    │    └─Sequential: 3-16                       [10, 1024, 10]            3,148,800
│    └─Conv1d: 2-17                                [10, 1024, 10]            3,146,752
│    └─ResBlock: 2-18                              [10, 1024, 10]            --
│    │    └─Sequential: 3-17                       [10, 1024, 10]            9,444,352
│    │    └─Sequential: 3-18                       [10, 1024, 10]            2,048
│    └─ConvTranspose1d: 2-19                       [10, 1024, 21]            2,098,176
├─ModuleList: 1-18                                 --                        (recursive)
│    └─Conv1d: 2-20                                [10, 1024, 21]            3,146,752
├─Sequential: 1-9                                  [10, 512, 43]             --
│    └─BasicConv: 2-21                             [10, 1024, 21]            --
│    │    └─Sequential: 3-19                       [10, 1024, 21]            6,294,528
│    └─ResBlock: 2-22                              [10, 1024, 21]            --
│    │    └─Sequential: 3-20                       [10, 1024, 21]            9,444,352
│    │    └─Sequential: 3-21                       [10, 1024, 21]            2,048
│    └─ConvTranspose1d: 2-23                       [10, 512, 43]             1,049,088
├─ModuleList: 1-18                                 --                        (recursive)
│    └─Conv1d: 2-24                                [10, 512, 43]             786,944
├─Sequential: 1-11                                 [10, 256, 87]             --
│    └─BasicConv: 2-25                             [10, 512, 43]             --
│    │    └─Sequential: 3-22                       [10, 512, 43]             1,574,400
│    └─ResBlock: 2-26                              [10, 512, 43]             --
│    │    └─Sequential: 3-23                       [10, 512, 43]             2,362,880
│    │    └─Sequential: 3-24                       [10, 512, 43]             1,024
│    └─ConvTranspose1d: 2-27                       [10, 256, 87]             262,400
├─ModuleList: 1-18                                 --                        (recursive)
│    └─Conv1d: 2-28                                [10, 256, 87]             196,864
├─Sequential: 1-13                                 [10, 128, 175]            --
│    └─BasicConv: 2-29                             [10, 256, 87]             --
│    │    └─Sequential: 3-25                       [10, 256, 87]             393,984
│    └─ResBlock: 2-30                              [10, 256, 87]             --
│    │    └─Sequential: 3-26                       [10, 256, 87]             591,616
│    │    └─Sequential: 3-27                       [10, 256, 87]             512
│    └─ConvTranspose1d: 2-31                       [10, 128, 175]            65,664
├─ModuleList: 1-18                                 --                        (recursive)
│    └─Conv1d: 2-32                                [10, 128, 175]            49,280
├─Sequential: 1-15                                 [10, 64, 350]             --
│    └─BasicConv: 2-33                             [10, 128, 175]            --
│    │    └─Sequential: 3-28                       [10, 128, 175]            98,688
│    └─ResBlock: 2-34                              [10, 128, 175]            --
│    │    └─Sequential: 3-29                       [10, 128, 175]            148,352
│    │    └─Sequential: 3-30                       [10, 128, 175]            256
│    └─ConvTranspose1d: 2-35                       [10, 64, 350]             16,448
├─ModuleList: 1-18                                 --                        (recursive)
│    └─Conv1d: 2-36                                [10, 64, 350]             12,352
├─Sequential: 1-17                                 [10, 1, 350]              --
│    └─BasicConv: 2-37                             [10, 64, 350]             --
│    │    └─Sequential: 3-31                       [10, 64, 350]             24,768
│    └─BasicConv: 2-38                             [10, 1, 350]              --
│    │    └─Sequential: 3-32                       [10, 1, 350]              195
├─ModuleList: 1-18                                 --                        (recursive)
│    └─Conv1d: 2-39                                [10, 1, 350]              4
├─BasicConv: 1-19                                  [10, 1, 350]              --
│    └─Sequential: 2-40                            [10, 1, 350]              --
│    │    └─Conv1d: 3-33                           [10, 1, 350]              7
│    │    └─BatchNorm1d: 3-34                      [10, 1, 350]              2
│    │    └─ReLU: 3-35                             [10, 1, 350]              --
====================================================================================================
Total params: 59,059,720
Trainable params: 59,059,720
Non-trainable params: 0
Total mult-adds (G): 15.07
====================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 156.04
Params size (MB): 236.24
Estimated Total Size (MB): 392.28
====================================================================================================
"""
