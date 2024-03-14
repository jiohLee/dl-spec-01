import os
from argparse import ArgumentParser
from functools import partial
import wandb
import time, datetime
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np

from utils import show_dict, set_seed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models import (
    cnn,
    rescnn,
    deepcubenet1d,
    resunet,
    transformer01,
    transformer02,
    transformer03,
    transformer04,
    transformer05,
    dataset,
)

set_seed(0)

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

if args.log:
    wandb.init(
        project="cs-spec",
        name=args.run_name,
        config=dict(args._get_kwargs()),
        save_code=True,
    )

show_dict(dict(args._get_kwargs()))


def build_model(model_cls, sensing_matrix_path=None) -> nn.Module:

    model = model_cls()

    if sensing_matrix_path:
        A = loadmat(sensing_matrix_path)["sensing_matrix"]
        T = torch.tensor(np.matmul(A.T, np.linalg.inv(np.matmul(A, A.T))), dtype=torch.float32)
        model.At.weight = nn.Parameter(T)

    return model


model_table = {
    "cnn-wa": partial(build_model, cnn.Model, "/root/spec/models/sensing_matrix.mat"),
    "cnn-woa": partial(build_model, cnn.Model),
    "rescnn-wa": partial(build_model, rescnn.Model, "/root/spec/models/sensing_matrix.mat"),
    "rescnn-woa": partial(build_model, rescnn.Model),
    "deepcubenet1d-wa": partial(build_model, deepcubenet1d.Model, "/root/spec/models/sensing_matrix.mat"),
    "deepcubenet1d-woa": partial(build_model, deepcubenet1d.Model),
    "resunet-wa": partial(build_model, resunet.Model, "/root/spec/models/sensing_matrix.mat"),
    "resunet-woa": partial(build_model, resunet.Model),
    "transformer01-wa": partial(transformer01.build_model, sensing_matrix_path="/root/spec/models/sensing_matrix.mat"),
    "transformer01-wa-nh1": partial(
        transformer01.build_model, n_heads=1, sensing_matrix_path="/root/spec/models/sensing_matrix.mat"
    ),
    "transformer01-woa": partial(transformer01.build_model),
    "transformer01-woa-nh1": partial(transformer01.build_model, n_heads=1),
    "transformer02-wa": partial(transformer02.build_model, sensing_matrix_path="/root/spec/models/sensing_matrix.mat"),
    "transformer02-wa-nh1": partial(
        transformer02.build_model, n_heads=1, sensing_matrix_path="/root/spec/models/sensing_matrix.mat"
    ),
    "transformer02-woa": partial(transformer02.build_model),
    "transformer02-woa-nh1": partial(transformer02.build_model, n_heads=1),
    "transformer03-wa": partial(transformer03.build_model, sensing_matrix_path="/root/spec/models/sensing_matrix.mat"),
    "transformer03-wa-nh1": partial(
        transformer03.build_model, n_heads=1, sensing_matrix_path="/root/spec/models/sensing_matrix.mat"
    ),
    "transformer03-woa": partial(transformer03.build_model),
    "transformer03-woa-nh1": partial(transformer03.build_model, n_heads=1),
    "transformer04-wa": partial(transformer04.build_model, sensing_matrix_path="/root/spec/models/sensing_matrix.mat"),
    "transformer04-wa-nh1": partial(
        transformer04.build_model, n_heads=1, sensing_matrix_path="/root/spec/models/sensing_matrix.mat"
    ),
    "transformer04-woa": partial(transformer04.build_model),
    "transformer04-woa-nh1": partial(transformer04.build_model, n_heads=1),
    "transformer05": partial(transformer05.build_model),
    "transformer05-nh1": partial(transformer05.build_model, n_heads=1),
}

dataset_table = {
    "synthetic": partial(dataset.Spectrum, root="/root/spec/datasets/spec-data-synthetic.mat"),
    "synthetic-n15db": partial(dataset.Spectrum, root="/root/spec/datasets/spec-data-synthetic-n15db.mat"),
    "synthetic-n20db": partial(dataset.Spectrum, root="/root/spec/datasets/spec-data-synthetic-n20db.mat"),
    "synthetic-n25db": partial(dataset.Spectrum, root="/root/spec/datasets/spec-data-synthetic-n25db.mat"),
    "synthetic-n30db": partial(dataset.Spectrum, root="/root/spec/datasets/spec-data-synthetic-n30db.mat"),
    "synthetic-n35db": partial(dataset.Spectrum, root="/root/spec/datasets/spec-data-synthetic-n35db.mat"),
    "synthetic-n40db": partial(dataset.Spectrum, root="/root/spec/datasets/spec-data-synthetic-n40db.mat"),
    "measured": partial(dataset.Spectrum, root="/root/spec/datasets/spec-data-measured.mat"),
}

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
            ckpt_dict = {
                "model": model.state_dict(),
                "optim": model_optim.state_dict(),
            }

            print(f"save checkpoint to {args.save_path}")
            torch.save(ckpt_dict, os.path.join(args.save_path, f"{args.run_name}.pt"))

torch.save(model.state_dict(), os.path.join(args.save_path, f"{args.run_name}_latest.pt"))

if args.log:
    wandb.finish()
