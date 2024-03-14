import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn

from torchinfo import summary


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
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


class Model(nn.Module):
    def __init__(self, channel=1, filters=[64, 128, 256, 512, 1024]):
        super(Model, self).__init__()

        self.At = nn.Linear(in_features=36, out_features=350, bias=True)
        # self.act = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

        self.input_layer = nn.Sequential(
            nn.Conv1d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(),
            nn.Conv1d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(nn.Conv1d(channel, filters[0], kernel_size=3, padding=1))

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=2, padding=1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=2, padding=1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], stride=2, padding=1)

        self.bridge = ResidualConv(filters[3], filters[4], stride=2, padding=1)

        self.upsample_1 = nn.ConvTranspose1d(
            in_channels=filters[4],
            out_channels=filters[4],
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)

        self.upsample_2 = nn.ConvTranspose1d(
            in_channels=filters[3],
            out_channels=filters[3],
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_3 = nn.ConvTranspose1d(
            in_channels=filters[2],
            out_channels=filters[2],
            kernel_size=2,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_4 = nn.ConvTranspose1d(
            in_channels=filters[1],
            out_channels=filters[1],
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.fc = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, y):
        # Encode

        ext = self.At(y)
        ext1 = self.act(ext)
        # ext2 = self.re(ext)
        x1 = self.input_layer(ext) + self.input_skip(ext)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        # Bridge
        x5 = self.bridge(x4)
        # Decode
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)

        x7 = self.up_residual_conv1(x6)

        x8 = self.upsample_2(x7)

        x9 = torch.cat([x8, x3], dim=1)

        x10 = self.up_residual_conv2(x9)

        x11 = self.upsample_3(x10)
        x12 = torch.cat([x11, x2], dim=1)

        x13 = self.up_residual_conv3(x12)

        x14 = self.upsample_4(x13)
        x15 = torch.cat([x14, x1], dim=1)

        x16 = self.up_residual_conv4(x15)
        output = self.fc(x16)

        return output + ext1
        # return output


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
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [10, 1, 350]              --
├─Linear: 1-1                            [10, 1, 350]              12,950
├─ReLU: 1-2                              [10, 1, 350]              --
├─Sequential: 1-3                        [10, 64, 350]             --
│    └─Conv1d: 2-1                       [10, 64, 350]             256
│    └─BatchNorm1d: 2-2                  [10, 64, 350]             128
│    └─ReLU: 2-3                         [10, 64, 350]             --
│    └─Conv1d: 2-4                       [10, 64, 350]             12,352
├─Sequential: 1-4                        [10, 64, 350]             --
│    └─Conv1d: 2-5                       [10, 64, 350]             256
├─ResidualConv: 1-5                      [10, 128, 175]            --
│    └─Sequential: 2-6                   [10, 128, 175]            --
│    │    └─BatchNorm1d: 3-1             [10, 64, 350]             128
│    │    └─ReLU: 3-2                    [10, 64, 350]             --
│    │    └─Conv1d: 3-3                  [10, 128, 175]            24,704
│    │    └─BatchNorm1d: 3-4             [10, 128, 175]            256
│    │    └─ReLU: 3-5                    [10, 128, 175]            --
│    │    └─Conv1d: 3-6                  [10, 128, 175]            49,280
│    └─Sequential: 2-7                   [10, 128, 175]            --
│    │    └─Conv1d: 3-7                  [10, 128, 175]            24,704
│    │    └─BatchNorm1d: 3-8             [10, 128, 175]            256
├─ResidualConv: 1-6                      [10, 256, 88]             --
│    └─Sequential: 2-8                   [10, 256, 88]             --
│    │    └─BatchNorm1d: 3-9             [10, 128, 175]            256
│    │    └─ReLU: 3-10                   [10, 128, 175]            --
│    │    └─Conv1d: 3-11                 [10, 256, 88]             98,560
│    │    └─BatchNorm1d: 3-12            [10, 256, 88]             512
│    │    └─ReLU: 3-13                   [10, 256, 88]             --
│    │    └─Conv1d: 3-14                 [10, 256, 88]             196,864
│    └─Sequential: 2-9                   [10, 256, 88]             --
│    │    └─Conv1d: 3-15                 [10, 256, 88]             98,560
│    │    └─BatchNorm1d: 3-16            [10, 256, 88]             512
├─ResidualConv: 1-7                      [10, 512, 44]             --
│    └─Sequential: 2-10                  [10, 512, 44]             --
│    │    └─BatchNorm1d: 3-17            [10, 256, 88]             512
│    │    └─ReLU: 3-18                   [10, 256, 88]             --
│    │    └─Conv1d: 3-19                 [10, 512, 44]             393,728
│    │    └─BatchNorm1d: 3-20            [10, 512, 44]             1,024
│    │    └─ReLU: 3-21                   [10, 512, 44]             --
│    │    └─Conv1d: 3-22                 [10, 512, 44]             786,944
│    └─Sequential: 2-11                  [10, 512, 44]             --
│    │    └─Conv1d: 3-23                 [10, 512, 44]             393,728
│    │    └─BatchNorm1d: 3-24            [10, 512, 44]             1,024
├─ResidualConv: 1-8                      [10, 1024, 22]            --
│    └─Sequential: 2-12                  [10, 1024, 22]            --
│    │    └─BatchNorm1d: 3-25            [10, 512, 44]             1,024
│    │    └─ReLU: 3-26                   [10, 512, 44]             --
│    │    └─Conv1d: 3-27                 [10, 1024, 22]            1,573,888
│    │    └─BatchNorm1d: 3-28            [10, 1024, 22]            2,048
│    │    └─ReLU: 3-29                   [10, 1024, 22]            --
│    │    └─Conv1d: 3-30                 [10, 1024, 22]            3,146,752
│    └─Sequential: 2-13                  [10, 1024, 22]            --
│    │    └─Conv1d: 3-31                 [10, 1024, 22]            1,573,888
│    │    └─BatchNorm1d: 3-32            [10, 1024, 22]            2,048
├─ConvTranspose1d: 1-9                   [10, 1024, 44]            2,098,176
├─ResidualConv: 1-10                     [10, 512, 44]             --
│    └─Sequential: 2-14                  [10, 512, 44]             --
│    │    └─BatchNorm1d: 3-33            [10, 1536, 44]            3,072
│    │    └─ReLU: 3-34                   [10, 1536, 44]            --
│    │    └─Conv1d: 3-35                 [10, 512, 44]             2,359,808
│    │    └─BatchNorm1d: 3-36            [10, 512, 44]             1,024
│    │    └─ReLU: 3-37                   [10, 512, 44]             --
│    │    └─Conv1d: 3-38                 [10, 512, 44]             786,944
│    └─Sequential: 2-15                  [10, 512, 44]             --
│    │    └─Conv1d: 3-39                 [10, 512, 44]             2,359,808
│    │    └─BatchNorm1d: 3-40            [10, 512, 44]             1,024
├─ConvTranspose1d: 1-11                  [10, 512, 88]             524,800
├─ResidualConv: 1-12                     [10, 256, 88]             --
│    └─Sequential: 2-16                  [10, 256, 88]             --
│    │    └─BatchNorm1d: 3-41            [10, 768, 88]             1,536
│    │    └─ReLU: 3-42                   [10, 768, 88]             --
│    │    └─Conv1d: 3-43                 [10, 256, 88]             590,080
│    │    └─BatchNorm1d: 3-44            [10, 256, 88]             512
│    │    └─ReLU: 3-45                   [10, 256, 88]             --
│    │    └─Conv1d: 3-46                 [10, 256, 88]             196,864
│    └─Sequential: 2-17                  [10, 256, 88]             --
│    │    └─Conv1d: 3-47                 [10, 256, 88]             590,080
│    │    └─BatchNorm1d: 3-48            [10, 256, 88]             512
├─ConvTranspose1d: 1-13                  [10, 256, 175]            131,328
├─ResidualConv: 1-14                     [10, 128, 175]            --
│    └─Sequential: 2-18                  [10, 128, 175]            --
│    │    └─BatchNorm1d: 3-49            [10, 384, 175]            768
│    │    └─ReLU: 3-50                   [10, 384, 175]            --
│    │    └─Conv1d: 3-51                 [10, 128, 175]            147,584
│    │    └─BatchNorm1d: 3-52            [10, 128, 175]            256
│    │    └─ReLU: 3-53                   [10, 128, 175]            --
│    │    └─Conv1d: 3-54                 [10, 128, 175]            49,280
│    └─Sequential: 2-19                  [10, 128, 175]            --
│    │    └─Conv1d: 3-55                 [10, 128, 175]            147,584
│    │    └─BatchNorm1d: 3-56            [10, 128, 175]            256
├─ConvTranspose1d: 1-15                  [10, 128, 350]            32,896
├─ResidualConv: 1-16                     [10, 64, 350]             --
│    └─Sequential: 2-20                  [10, 64, 350]             --
│    │    └─BatchNorm1d: 3-57            [10, 192, 350]            384
│    │    └─ReLU: 3-58                   [10, 192, 350]            --
│    │    └─Conv1d: 3-59                 [10, 64, 350]             36,928
│    │    └─BatchNorm1d: 3-60            [10, 64, 350]             128
│    │    └─ReLU: 3-61                   [10, 64, 350]             --
│    │    └─Conv1d: 3-62                 [10, 64, 350]             12,352
│    └─Sequential: 2-21                  [10, 64, 350]             --
│    │    └─Conv1d: 3-63                 [10, 64, 350]             36,928
│    │    └─BatchNorm1d: 3-64            [10, 64, 350]             128
├─Conv1d: 1-17                           [10, 1, 350]              65
==========================================================================================
Total params: 18,508,247
Trainable params: 18,508,247
Non-trainable params: 0
Total mult-adds (G): 8.91
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 122.29
Params size (MB): 74.03
Estimated Total Size (MB): 196.33
==========================================================================================
"""
