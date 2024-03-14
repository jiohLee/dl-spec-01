import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary


class BasicConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(BasicConv, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3,), padding="same", padding_mode="reflect"
        )
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels: int):
        super(ResBlock, self).__init__()

        self.bc1 = BasicConv(in_channels=in_channels, out_channels=in_channels)
        self.bc2 = BasicConv(in_channels=in_channels, out_channels=in_channels)

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=(3,), padding="same", padding_mode="reflect"
        )

        self.bn1 = nn.BatchNorm1d(num_features=in_channels)

    def forward(self, x):

        residual = x

        x = self.bc1(x)
        x = self.bc2(x)
        x = self.conv1(x)

        x = residual + x

        x = F.relu(self.bn1(x))

        return x


class Block(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, sampler: nn.Module):
        super(Block, self).__init__()

        self.bc = BasicConv(in_channels=in_channels, out_channels=out_channels)
        self.rb = ResBlock(out_channels)
        self.sampler = sampler

    def forward(self, x):

        x = self.bc(x)
        out_residual = self.rb(x)
        out_sampled = self.sampler(out_residual)

        return {"residual": out_residual, "sampled": out_sampled}


# Res-Unet
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.At = nn.Linear(in_features=36, out_features=350, bias=False)

        # contracting path
        self.db1 = Block(1, 64, nn.MaxPool1d(kernel_size=(2,), stride=(2,)))
        self.db2 = Block(64, 128, nn.MaxPool1d(kernel_size=(2,), stride=(2,)))
        self.db3 = Block(128, 256, nn.MaxPool1d(kernel_size=(2,), stride=(2,)))
        self.db4 = Block(256, 512, nn.MaxPool1d(kernel_size=(2,), stride=(2,)))
        self.db5 = Block(512, 1024, nn.MaxPool1d(kernel_size=(2,), stride=(2,)))

        # bottleneck
        self.bc1 = BasicConv(1024, 1024)
        self.cconv7 = nn.Conv1d(
            in_channels=1024, out_channels=1024, kernel_size=(3,), padding="same", padding_mode="reflect"
        )
        self.rb1 = ResBlock(1024)
        self.upconv1 = nn.ConvTranspose1d(
            in_channels=1024, out_channels=1024, kernel_size=(2,), stride=(2,), output_padding=(1,)
        )

        self.ub2 = Block(256, 128, nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=(2,), stride=(2,)))
        self.ub3 = Block(
            512,
            256,
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=(2,), stride=(2,), output_padding=(1,)),
        )
        self.ub4 = Block(
            1024,
            512,
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=(2,), stride=(2,), output_padding=(1,)),
        )
        self.ub5 = Block(
            2048,
            1024,
            nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=(2,), stride=(2,), output_padding=(1,)),
        )

        self.cconv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(3,), padding="same", padding_mode="reflect")

        self.cconv2 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=(3,), padding="same", padding_mode="reflect"
        )

        self.cconv3 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=(3,), padding="same", padding_mode="reflect"
        )

        self.cconv4 = nn.Conv1d(
            in_channels=256, out_channels=256, kernel_size=(3,), padding="same", padding_mode="reflect"
        )

        self.cconv5 = nn.Conv1d(
            in_channels=512, out_channels=512, kernel_size=(3,), padding="same", padding_mode="reflect"
        )

        self.cconv6 = nn.Conv1d(
            in_channels=1024, out_channels=1024, kernel_size=(3,), padding="same", padding_mode="reflect"
        )

        self.obc1 = BasicConv(128, 64)
        self.obc2 = BasicConv(64, 1)
        self.obc3 = BasicConv(2, 1)

    def forward(self, y):

        ext = self.At(y)

        residual = self.cconv1(ext)

        # contracting path
        db1 = self.db1(ext)
        db1["residual"] = self.cconv2(db1["residual"])
        # print(f"residual shape db1: {db1['residual'].shape}")

        db2 = self.db2(db1["sampled"])
        db2["residual"] = self.cconv3(db2["residual"])
        # print(f"residual shape db2: {db2['residual'].shape}")

        db3 = self.db3(db2["sampled"])
        db3["residual"] = self.cconv4(db3["residual"])
        # print(f"residual shape db3: {db3['residual'].shape}")

        db4 = self.db4(db3["sampled"])
        db4["residual"] = self.cconv5(db4["residual"])
        # print(f"residual shape db4: {db4['residual'].shape}")

        db5 = self.db5(db4["sampled"])
        db5["residual"] = self.cconv6(db5["residual"])
        # print(f"residual shape db5: {db5['residual'].shape}")

        # bottleneck
        x = self.bc1(db5["sampled"])
        x = self.cconv7(x)
        x = self.rb1(x)
        x = self.upconv1(x)

        # expansive path
        x = torch.cat([x, db5["residual"]], dim=1)
        # print(f"ub5 input shape : {x.shape}")
        x = self.ub5(x)

        x = torch.cat([x["sampled"], db4["residual"]], dim=1)
        # print(f"ub4 input shape : {x.shape}")
        x = self.ub4(x)

        x = torch.cat([x["sampled"], db3["residual"]], dim=1)
        # print(f"ub3 input shape : {x.shape}")
        x = self.ub3(x)

        x = torch.cat([x["sampled"], db2["residual"]], dim=1)
        # print(f"ub2 input shape : {x.shape}")
        x = self.ub2(x)

        x = torch.cat([x["sampled"], db1["residual"]], dim=1)
        # print(f"obc1 input shape : {x.shape}")
        x = self.obc1(x)
        x = self.obc2(x)
        x = torch.cat([x, residual], dim=1)
        x = self.obc3(x)

        # x = torch.flatten(x, 1)

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
avg.MSE = 0.0001852572788663309
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [10, 1, 350]              --
├─Linear: 1-1                            [10, 1, 350]              12,600
├─Conv1d: 1-2                            [10, 1, 350]              4
├─Block: 1-3                             [10, 64, 175]             --
│    └─BasicConv: 2-1                    [10, 64, 350]             --
│    │    └─Conv1d: 3-1                  [10, 64, 350]             256
│    │    └─BatchNorm1d: 3-2             [10, 64, 350]             128
│    └─ResBlock: 2-2                     [10, 64, 350]             --
│    │    └─BasicConv: 3-3               [10, 64, 350]             12,480
│    │    └─BasicConv: 3-4               [10, 64, 350]             12,480
│    │    └─Conv1d: 3-5                  [10, 64, 350]             12,352
│    │    └─BatchNorm1d: 3-6             [10, 64, 350]             128
│    └─MaxPool1d: 2-3                    [10, 64, 175]             --
├─Conv1d: 1-4                            [10, 64, 350]             12,352
├─Block: 1-5                             [10, 128, 87]             --
│    └─BasicConv: 2-4                    [10, 128, 175]            --
│    │    └─Conv1d: 3-7                  [10, 128, 175]            24,704
│    │    └─BatchNorm1d: 3-8             [10, 128, 175]            256
│    └─ResBlock: 2-5                     [10, 128, 175]            --
│    │    └─BasicConv: 3-9               [10, 128, 175]            49,536
│    │    └─BasicConv: 3-10              [10, 128, 175]            49,536
│    │    └─Conv1d: 3-11                 [10, 128, 175]            49,280
│    │    └─BatchNorm1d: 3-12            [10, 128, 175]            256
│    └─MaxPool1d: 2-6                    [10, 128, 87]             --
├─Conv1d: 1-6                            [10, 128, 175]            49,280
├─Block: 1-7                             [10, 256, 43]             --
│    └─BasicConv: 2-7                    [10, 256, 87]             --
│    │    └─Conv1d: 3-13                 [10, 256, 87]             98,560
│    │    └─BatchNorm1d: 3-14            [10, 256, 87]             512
│    └─ResBlock: 2-8                     [10, 256, 87]             --
│    │    └─BasicConv: 3-15              [10, 256, 87]             197,376
│    │    └─BasicConv: 3-16              [10, 256, 87]             197,376
│    │    └─Conv1d: 3-17                 [10, 256, 87]             196,864
│    │    └─BatchNorm1d: 3-18            [10, 256, 87]             512
│    └─MaxPool1d: 2-9                    [10, 256, 43]             --
├─Conv1d: 1-8                            [10, 256, 87]             196,864
├─Block: 1-9                             [10, 512, 21]             --
│    └─BasicConv: 2-10                   [10, 512, 43]             --
│    │    └─Conv1d: 3-19                 [10, 512, 43]             393,728
│    │    └─BatchNorm1d: 3-20            [10, 512, 43]             1,024
│    └─ResBlock: 2-11                    [10, 512, 43]             --
│    │    └─BasicConv: 3-21              [10, 512, 43]             787,968
│    │    └─BasicConv: 3-22              [10, 512, 43]             787,968
│    │    └─Conv1d: 3-23                 [10, 512, 43]             786,944
│    │    └─BatchNorm1d: 3-24            [10, 512, 43]             1,024
│    └─MaxPool1d: 2-12                   [10, 512, 21]             --
├─Conv1d: 1-10                           [10, 512, 43]             786,944
├─Block: 1-11                            [10, 1024, 10]            --
│    └─BasicConv: 2-13                   [10, 1024, 21]            --
│    │    └─Conv1d: 3-25                 [10, 1024, 21]            1,573,888
│    │    └─BatchNorm1d: 3-26            [10, 1024, 21]            2,048
│    └─ResBlock: 2-14                    [10, 1024, 21]            --
│    │    └─BasicConv: 3-27              [10, 1024, 21]            3,148,800
│    │    └─BasicConv: 3-28              [10, 1024, 21]            3,148,800
│    │    └─Conv1d: 3-29                 [10, 1024, 21]            3,146,752
│    │    └─BatchNorm1d: 3-30            [10, 1024, 21]            2,048
│    └─MaxPool1d: 2-15                   [10, 1024, 10]            --
├─Conv1d: 1-12                           [10, 1024, 21]            3,146,752
├─BasicConv: 1-13                        [10, 1024, 10]            --
│    └─Conv1d: 2-16                      [10, 1024, 10]            3,146,752
│    └─BatchNorm1d: 2-17                 [10, 1024, 10]            2,048
├─Conv1d: 1-14                           [10, 1024, 10]            3,146,752
├─ResBlock: 1-15                         [10, 1024, 10]            --
│    └─BasicConv: 2-18                   [10, 1024, 10]            --
│    │    └─Conv1d: 3-31                 [10, 1024, 10]            3,146,752
│    │    └─BatchNorm1d: 3-32            [10, 1024, 10]            2,048
│    └─BasicConv: 2-19                   [10, 1024, 10]            --
│    │    └─Conv1d: 3-33                 [10, 1024, 10]            3,146,752
│    │    └─BatchNorm1d: 3-34            [10, 1024, 10]            2,048
│    └─Conv1d: 2-20                      [10, 1024, 10]            3,146,752
│    └─BatchNorm1d: 2-21                 [10, 1024, 10]            2,048
├─ConvTranspose1d: 1-16                  [10, 1024, 21]            2,098,176
├─Block: 1-17                            [10, 512, 43]             --
│    └─BasicConv: 2-22                   [10, 1024, 21]            --
│    │    └─Conv1d: 3-35                 [10, 1024, 21]            6,292,480
│    │    └─BatchNorm1d: 3-36            [10, 1024, 21]            2,048
│    └─ResBlock: 2-23                    [10, 1024, 21]            --
│    │    └─BasicConv: 3-37              [10, 1024, 21]            3,148,800
│    │    └─BasicConv: 3-38              [10, 1024, 21]            3,148,800
│    │    └─Conv1d: 3-39                 [10, 1024, 21]            3,146,752
│    │    └─BatchNorm1d: 3-40            [10, 1024, 21]            2,048
│    └─ConvTranspose1d: 2-24             [10, 512, 43]             1,049,088
├─Block: 1-18                            [10, 256, 87]             --
│    └─BasicConv: 2-25                   [10, 512, 43]             --
│    │    └─Conv1d: 3-41                 [10, 512, 43]             1,573,376
│    │    └─BatchNorm1d: 3-42            [10, 512, 43]             1,024
│    └─ResBlock: 2-26                    [10, 512, 43]             --
│    │    └─BasicConv: 3-43              [10, 512, 43]             787,968
│    │    └─BasicConv: 3-44              [10, 512, 43]             787,968
│    │    └─Conv1d: 3-45                 [10, 512, 43]             786,944
│    │    └─BatchNorm1d: 3-46            [10, 512, 43]             1,024
│    └─ConvTranspose1d: 2-27             [10, 256, 87]             262,400
├─Block: 1-19                            [10, 128, 175]            --
│    └─BasicConv: 2-28                   [10, 256, 87]             --
│    │    └─Conv1d: 3-47                 [10, 256, 87]             393,472
│    │    └─BatchNorm1d: 3-48            [10, 256, 87]             512
│    └─ResBlock: 2-29                    [10, 256, 87]             --
│    │    └─BasicConv: 3-49              [10, 256, 87]             197,376
│    │    └─BasicConv: 3-50              [10, 256, 87]             197,376
│    │    └─Conv1d: 3-51                 [10, 256, 87]             196,864
│    │    └─BatchNorm1d: 3-52            [10, 256, 87]             512
│    └─ConvTranspose1d: 2-30             [10, 128, 175]            65,664
├─Block: 1-20                            [10, 64, 350]             --
│    └─BasicConv: 2-31                   [10, 128, 175]            --
│    │    └─Conv1d: 3-53                 [10, 128, 175]            98,432
│    │    └─BatchNorm1d: 3-54            [10, 128, 175]            256
│    └─ResBlock: 2-32                    [10, 128, 175]            --
│    │    └─BasicConv: 3-55              [10, 128, 175]            49,536
│    │    └─BasicConv: 3-56              [10, 128, 175]            49,536
│    │    └─Conv1d: 3-57                 [10, 128, 175]            49,280
│    │    └─BatchNorm1d: 3-58            [10, 128, 175]            256
│    └─ConvTranspose1d: 2-33             [10, 64, 350]             16,448
├─BasicConv: 1-21                        [10, 64, 350]             --
│    └─Conv1d: 2-34                      [10, 64, 350]             24,640
│    └─BatchNorm1d: 2-35                 [10, 64, 350]             128
├─BasicConv: 1-22                        [10, 1, 350]              --
│    └─Conv1d: 2-36                      [10, 1, 350]              193
│    └─BatchNorm1d: 2-37                 [10, 1, 350]              2
├─BasicConv: 1-23                        [10, 1, 350]              --
│    └─Conv1d: 2-38                      [10, 1, 350]              7
│    └─BatchNorm1d: 2-39                 [10, 1, 350]              2
==========================================================================================
Total params: 59,059,720
Trainable params: 59,059,720
Non-trainable params: 0
Total mult-adds (G): 15.07
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 156.04
Params size (MB): 236.24
Estimated Total Size (MB): 392.28
==========================================================================================
"""
