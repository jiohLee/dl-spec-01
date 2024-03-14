import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


def block_down(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="reflect",
        ),
        nn.ReLU(),
        nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="reflect",
        ),
        nn.ReLU(),
    )


def block_up(in_channels, out_channels, kernel_size=3, upsample_scale=2, output_padding=1):
    return nn.Sequential(
        nn.Conv1d(
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="reflect",
        ),
        nn.ReLU(),
        nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=upsample_scale,
            stride=upsample_scale,
            output_padding=output_padding,
        ),
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="reflect",
        ),
        nn.ReLU(),
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.At = nn.Linear(in_features=36, out_features=350, bias=False)

        self.block_down_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=11, padding="same", padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=11, padding="same", padding_mode="reflect"),
            nn.ReLU(),
        )
        self.block_down_2 = block_down(8, 16, kernel_size=9)
        self.block_down_3 = block_down(16, 32, kernel_size=7)

        self.bridge = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, padding="same", padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding="same", padding_mode="reflect"),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, padding="same", padding_mode="reflect"),
            nn.ReLU(),
        )

        self.block_up_3 = block_up(32, 16, kernel_size=7)
        self.block_up_2 = block_up(16, 8, kernel_size=9, output_padding=0)
        self.block_up_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=11, padding="same", padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=11, padding="same", padding_mode="reflect"),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.At(x)

        x1 = self.block_down_1(x)
        x2 = self.block_down_2(x1)
        x3 = self.block_down_3(x2)

        x_ = self.bridge(x3)

        x_ = self.block_up_3(torch.cat([x_, x3], dim=1))
        x_ = self.block_up_2(torch.cat([x_, x2], dim=1))
        x_ = self.block_up_1(torch.cat([x_, x1], dim=1))

        return x


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
├─Linear: 1-1                            [10, 1, 350]              12,600
├─Sequential: 1-2                        [10, 8, 350]              --
│    └─Conv1d: 2-1                       [10, 8, 350]              96
│    └─ReLU: 2-2                         [10, 8, 350]              --
│    └─Conv1d: 2-3                       [10, 8, 350]              712
│    └─ReLU: 2-4                         [10, 8, 350]              --
├─Sequential: 1-3                        [10, 16, 175]             --
│    └─MaxPool1d: 2-5                    [10, 8, 175]              --
│    └─Conv1d: 2-6                       [10, 16, 175]             1,168
│    └─ReLU: 2-7                         [10, 16, 175]             --
│    └─Conv1d: 2-8                       [10, 16, 175]             2,320
│    └─ReLU: 2-9                         [10, 16, 175]             --
├─Sequential: 1-4                        [10, 32, 87]              --
│    └─MaxPool1d: 2-10                   [10, 16, 87]              --
│    └─Conv1d: 2-11                      [10, 32, 87]              3,616
│    └─ReLU: 2-12                        [10, 32, 87]              --
│    └─Conv1d: 2-13                      [10, 32, 87]              7,200
│    └─ReLU: 2-14                        [10, 32, 87]              --
├─Sequential: 1-5                        [10, 32, 87]              --
│    └─MaxPool1d: 2-15                   [10, 32, 43]              --
│    └─Conv1d: 2-16                      [10, 128, 43]             20,608
│    └─ReLU: 2-17                        [10, 128, 43]             --
│    └─Conv1d: 2-18                      [10, 128, 43]             82,048
│    └─ReLU: 2-19                        [10, 128, 43]             --
│    └─ConvTranspose1d: 2-20             [10, 128, 87]             49,280
│    └─Conv1d: 2-21                      [10, 32, 87]              28,704
│    └─ReLU: 2-22                        [10, 32, 87]              --
├─Sequential: 1-6                        [10, 16, 175]             --
│    └─Conv1d: 2-23                      [10, 32, 87]              14,368
│    └─ReLU: 2-24                        [10, 32, 87]              --
│    └─ConvTranspose1d: 2-25             [10, 32, 175]             2,080
│    └─Conv1d: 2-26                      [10, 16, 175]             3,600
│    └─ReLU: 2-27                        [10, 16, 175]             --
├─Sequential: 1-7                        [10, 8, 350]              --
│    └─Conv1d: 2-28                      [10, 16, 175]             4,624
│    └─ReLU: 2-29                        [10, 16, 175]             --
│    └─ConvTranspose1d: 2-30             [10, 16, 350]             528
│    └─Conv1d: 2-31                      [10, 8, 350]              1,160
│    └─ReLU: 2-32                        [10, 8, 350]              --
├─Sequential: 1-8                        [10, 1, 350]              --
│    └─Conv1d: 2-33                      [10, 8, 350]              1,416
│    └─ReLU: 2-34                        [10, 8, 350]              --
│    └─Conv1d: 2-35                      [10, 1, 350]              89
│    └─ReLU: 2-36                        [10, 1, 350]              --
==========================================================================================
Total params: 236,217
Trainable params: 236,217
Non-trainable params: 0
Total mult-adds (M): 172.16
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 5.41
Params size (MB): 0.94
Estimated Total Size (MB): 6.35
==========================================================================================
"""
