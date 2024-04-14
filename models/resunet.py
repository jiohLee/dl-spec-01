import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from pprint import pprint


class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class SkipConnection(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, mid_module, mid_padding=0) -> None:
        super().__init__()

        self.down = ResidualConv(in_ch=in_ch, out_ch=out_ch, stride=2)

        self.mid_module = mid_module

        self.upsample = nn.ConvTranspose1d(
            in_channels=mid_ch,
            out_channels=mid_ch,
            kernel_size=2,
            stride=2,
            padding=mid_padding,
            output_padding=mid_padding,
        )

        self.up = ResidualConv(in_ch=mid_ch + out_ch, out_ch=out_ch, stride=1)

    def forward(self, x):
        x_down = self.down(x)
        x_upsample = self.upsample(self.mid_module(x_down))
        return self.up(torch.cat([x_upsample, x_down], dim=1))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        sc = ResidualConv(512, 1024, 2)
        sc = SkipConnection(256, 512, 1024, mid_module=sc)
        sc = SkipConnection(128, 256, 512, mid_module=sc)
        sc = SkipConnection(64, 128, 256, mid_module=sc)
        sc = SkipConnection(1, 64, 128, mid_module=sc, mid_padding=1)

        self.At = nn.Linear(in_features=36, out_features=350, bias=False)
        # self.At = nn.Linear(in_features=36, out_features=350)

        self.layers = nn.Sequential(
            sc,
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
            ),
            nn.Conv1d(
                in_channels=64,
                out_channels=1,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(self.At(x))


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

    # num_Conv1d = sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))
    # num_ConvTranspose1d = sum(1 for m in model.modules() if isinstance(m, nn.ConvTranspose1d))
    # print(num_Conv1d, num_ConvTranspose1d)


if __name__ == "__main__":
    main()


"""
==============================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
==============================================================================================================
Model                                                        [10, 1, 350]              --
├─Linear: 1-1                                                [10, 1, 350]              12,950
├─Sequential: 1-2                                            [10, 1, 350]              --
│    └─SkipConnection: 2-1                                   [10, 64, 175]             --
│    │    └─ResidualConv: 3-1                                [10, 64, 175]             13,056
│    │    └─SkipConnection: 3-2                              [10, 128, 88]             18,352,384
│    │    └─ConvTranspose1d: 3-3                             [10, 128, 175]            32,896
│    │    └─ResidualConv: 3-4                                [10, 64, 175]             86,400
│    └─ConvTranspose1d: 2-2                                  [10, 64, 350]             8,256
│    └─Conv1d: 2-3                                           [10, 1, 350]              65
│    └─ReLU: 2-4                                             [10, 1, 350]              --
==============================================================================================================
Total params: 18,506,007
Trainable params: 18,506,007
Non-trainable params: 0
Total mult-adds (G): 4.48
==============================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 57.65
Params size (MB): 74.02
Estimated Total Size (MB): 131.67
==============================================================================================================
"""
