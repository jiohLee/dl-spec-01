import os
from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import show_dict
from models import (
    cnn,
    rescnn,
    resunet,
    dataset,
)

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="cnn")
parser.add_argument("--dataset_name", type=str, default="synthetic")
parser.add_argument("--run_name", type=str, default="exp01")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--weights", type=str, default="./model.pt")
parser.add_argument("--log", action="store_true", default=False)

args, _ = parser.parse_known_args()
args.save_path = os.path.join("./results", args.run_name)
os.makedirs(args.save_path, exist_ok=True)
show_dict(dict(args._get_kwargs()))


def build_model(model_cls) -> nn.Module:
    return model_cls()


model_table = {
    "cnn-wa": partial(build_model, cnn.Model),
    "cnn-wagu": partial(build_model, cnn.Model),
    "cnn-woa": partial(build_model, cnn.Model),
    "rescnn-wa": partial(build_model, rescnn.Model),
    "rescnn-wagu": partial(build_model, rescnn.Model),
    "rescnn-woa": partial(build_model, rescnn.Model),
    "resunet-wa": partial(build_model, resunet.Model),
    "resunet-wagu": partial(build_model, resunet.Model),
    "resunet-woa": partial(build_model, resunet.Model),
}

dataset_table = {
    "synthetic": partial(dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic.mat"),
    "synthetic-n40db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n40db.mat"
    ),
    "synthetic-n30db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n30db.mat"
    ),
    "synthetic-n20db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n20db.mat"
    ),
    "measured": partial(dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-measured.mat"),
    "drink-pink": partial(dataset.Spectrum, split="pink", root="/root/spec/datasets/spec-data-drink.mat"),
    "drink-gold": partial(dataset.Spectrum, split="gold", root="/root/spec/datasets/spec-data-drink.mat"),
    "drink-pyellow": partial(dataset.Spectrum, split="pyellow", root="/root/spec/datasets/spec-data-drink.mat"),
    "drink-blue": partial(dataset.Spectrum, split="blue", root="/root/spec/datasets/spec-data-drink.mat"),
    "drink-purple": partial(dataset.Spectrum, split="purple", root="/root/spec/datasets/spec-data-drink.mat"),
}

if args.log:
    api = wandb.Api()
    runs = api.runs(os.path.join("jioh0826", "cs-spec"))
    table_run = {run.name: run for run in runs}

    if args.run_name in table_run:
        wandb.init(project="cs-spec", id=table_run[args.run_name].id, resume="must")
    else:
        wandb.init(project="cs-spec", name=args.run_name, config=dict(args._get_kwargs()))

rank = os.environ.get("LOCAL_RANK", 0)
device = torch.device("cuda", rank)

model_cls = model_table[args.model_name]
dataset_cls = dataset_table[args.dataset_name]

dataset_test = dataset_cls()
loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size)

model = model_cls().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))


data = {}


def named_hook(name: str):

    def hook(module, x, y):
        data[name] = y

    return hook


model.At.register_forward_hook(named_hook("At_out"))


def test():
    model.eval()

    y_list, x_list, xr_list, At_out_list, error_list, error_reduced_list = [], [], [], [], [], []
    bar = tqdm(loader_test)
    for idx, (x, y) in enumerate(bar):
        # x: (N, 1, 350)
        # y: (N, 1, 36)
        x, y = x.to(device), y.to(device)

        xr = model(y)
        error = x - xr
        error_reduced = error.pow(2).mean(dim=-1).view(-1)

        y_list.append(y.squeeze(1))
        x_list.append(x.squeeze(1))
        xr_list.append(xr.squeeze(1))
        At_out_list.append(data["At_out"].squeeze(1))
        error_list.append(error.squeeze(1))
        error_reduced_list.append(error_reduced)

    y_list = torch.cat(y_list).cpu().numpy()
    x_list = torch.cat(x_list).cpu().numpy()
    xr_list = torch.cat(xr_list).cpu().numpy()
    At_out_list = torch.cat(At_out_list).cpu().numpy()
    error_list = torch.cat(error_list).cpu().numpy()
    error_reduced_list = torch.cat(error_reduced_list).cpu().numpy()
    psnr_list = [10 * np.log10(np.power(np.max(x) - np.min(x), 2) / mse) for x, mse in zip(x_list, error_reduced_list)]

    np.savetxt(os.path.join(args.save_path, "y.csv"), y_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "x.csv"), x_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "xr.csv"), xr_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "At_out.csv"), At_out_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "error.csv"), error_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "error_reduced.csv"), error_reduced_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "psnr.csv"), psnr_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "At.csv"), model.At.weight.data.cpu().numpy(), delimiter=",")

    avg_mse = sum(error_reduced_list) / len(error_reduced_list)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    print(f"total {len(error_reduced_list)} samples, avg MSE: {avg_mse}, avg PSNR: {avg_psnr}")

    if args.log:
        tb = wandb.Table(columns=["model_name", "dataset_name", "avg_mse", "avg_psnr"])
        tb.add_data(args.model_name, args.dataset_name, avg_mse, avg_psnr)
        wandb.log({"table/test_summary": tb})
        wandb.save(os.path.join(args.save_path, "*.csv"))


with torch.no_grad():
    test()

if args.log:
    wandb.finish()
