import os
from argparse import ArgumentParser
from functools import partial
import wandb
import time, datetime
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import show_dict, set_seed
from models import (
    dataset,
    cnn,
    rescnn,
    resunet,
)

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="cnn")
parser.add_argument("--dataset_name", type=str, default="synthetic")
parser.add_argument("--run_name", type=str, default="exp01")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--ckpt_freq", type=int, default=1)
parser.add_argument("--ckpt_delay", type=int, default=50)
parser.add_argument("--log", action="store_true", default=False)

args, _ = parser.parse_known_args()
args.save_path = os.path.join("./results", args.run_name)
os.makedirs(args.save_path, exist_ok=True)
show_dict(dict(args._get_kwargs()))


def build_model(model_cls, requires_grad=False, smp=None) -> nn.Module:

    model = model_cls()

    if smp:
        T = loadmat(smp)["sensing_matrix"]
        A = torch.tensor(np.matmul(T.T, np.linalg.inv(np.matmul(T, T.T))), dtype=torch.float32)
        model.At.weight = nn.Parameter(A)
        model.At.weight.requires_grad = requires_grad

    return model


model_table = {
    "cnn-wa": partial(build_model, cnn.Model, smp="/root/spec/models/sensing_matrix.mat"),
    "cnn-wagu": partial(build_model, cnn.Model, smp="/root/spec/models/sensing_matrix.mat", requires_grad=True),
    "cnn-woa": partial(build_model, cnn.Model),
    "rescnn-wa": partial(build_model, rescnn.Model, smp="/root/spec/models/sensing_matrix.mat"),
    "rescnn-wagu": partial(build_model, rescnn.Model, smp="/root/spec/models/sensing_matrix.mat", requires_grad=True),
    "rescnn-woa": partial(build_model, rescnn.Model),
    "resunet-wa": partial(build_model, resunet.Model, smp="/root/spec/models/sensing_matrix.mat"),
    "resunet-wagu": partial(build_model, resunet.Model, smp="/root/spec/models/sensing_matrix.mat", requires_grad=True),
    "resunet-woa": partial(build_model, resunet.Model),
}

dataset_table = {
    "synthetic": partial(dataset.Spectrum, root="/root/spec/datasets/synthetic.mat"),
    "synthetic-n40db": partial(dataset.Spectrum, root="/root/spec/datasets/synthetic-n40db.mat"),
    "synthetic-n30db": partial(dataset.Spectrum, root="/root/spec/datasets/synthetic-n30db.mat"),
    "synthetic-n20db": partial(dataset.Spectrum, root="/root/spec/datasets/synthetic-n20db.mat"),
    "synthetic-n35db": partial(dataset.Spectrum, root="/root/spec/datasets/synthetic-n35db.mat"),
    "synthetic-n25db": partial(dataset.Spectrum, root="/root/spec/datasets/synthetic-n25db.mat"),
    "synthetic-n15db": partial(dataset.Spectrum, root="/root/spec/datasets/synthetic-n15db.mat"),
    "measured": partial(dataset.Spectrum, root="/root/spec/datasets/measured.mat"),
}

set_seed(0)

if args.log:
    run = wandb.init(
        project="cs-spec",
        name=args.run_name,
        config=dict(args._get_kwargs()),
        save_code=True,
    )
    print(f"start run {run.name}, {run.id}")

rank = os.environ.get("LOCAL_RANK", 0)
device = torch.device("cuda", rank)

model_cls = model_table[args.model_name]
dataset_cls = dataset_table[args.dataset_name]

dataset_train = dataset_cls(split="train")
dataset_valid = dataset_cls(split="valid")
dataset_test = dataset_cls(split="test")
loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
loader_valid = DataLoader(dataset=dataset_valid, batch_size=args.batch_size)
loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size)

model = model_cls().to(device)
model_optim = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-5)
if args.log:
    wandb.watch(model, criterion="all", log_freq=1)


def train_epoch():
    model.train()

    train_loss, cnt = 0, 0
    bar = tqdm(loader_train, desc="[train]")
    for idx, (x, y) in enumerate(bar):
        x, y = x.to(device), y.to(device)

        xr = model(y)
        loss = F.mse_loss(xr, x)

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        cnt_ = x.shape[0]
        cnt += cnt_
        train_loss += loss.item() * cnt_

        result = {"train/loss": train_loss / cnt}
        bar.set_postfix_str(", ".join(f"{k}={v:.05f}" for k, v in result.items()))

    return result


def valid():
    model.eval()

    valid_loss, cnt = 0, 0
    bar = tqdm(loader_valid, desc="[valid]")
    for idx, (x, y) in enumerate(bar):
        x, y = x.to(device), y.to(device)

        xr = model(y)
        loss = F.mse_loss(xr, x)

        cnt_ = x.shape[0]
        cnt += cnt_
        valid_loss += loss.item() * cnt_

        result = {"valid/loss": valid_loss / cnt}
        bar.set_postfix_str(", ".join(f"{k}={v:.05f}" for k, v in result.items()))

    return result


def test():
    model.eval()

    test_loss, cnt = 0, 0
    bar = tqdm(loader_test, desc="[test]")
    for idx, (x, y) in enumerate(bar):
        x, y = x.to(device), y.to(device)

        xr = model(y)
        loss = F.mse_loss(xr, x)

        cnt_ = x.shape[0]
        cnt += cnt_
        test_loss += loss.item() * cnt_

        result = {"test/loss": test_loss / cnt}
        bar.set_postfix_str(", ".join(f"{k}={v:.05f}" for k, v in result.items()))

    return result


value = float("inf")
start = time.time()
for epoch in range(args.epochs):
    result_train = train_epoch()

    with torch.no_grad():
        result_valid = valid()
        result_test = test()

    result = dict(**result_train, **result_valid, **result_test)

    if args.log:
        wandb.log(result, step=epoch)

    print(
        f"Epoch: [{epoch + 1}/{args.epochs}], duration: {datetime.timedelta(seconds=int(time.time() - start))}\n"
        + ", ".join(f"{k}={v:.05f}" for k, v in result.items())
    )

    if (epoch + 1) % args.ckpt_freq == 0 and args.ckpt_delay < (epoch + 1):
        if min(value, result["valid/loss"]) == result["valid/loss"]:
            value = result["valid/loss"]
            print(f"save to {args.save_path}")
            torch.save(model.state_dict(), os.path.join(args.save_path, f"{args.run_name}.pt"))

torch.save(model.state_dict(), os.path.join(args.save_path, f"{args.run_name}_latest.pt"))

if args.log:
    wandb.finish()
