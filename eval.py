import os
import csv
from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
import numpy as np

from utils import show_dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import (
    cnn,
    rescnn,
    deepcubenet1d,
    resunet,
    dataset,
)

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="cnn")
parser.add_argument("--dataset_name", type=str, default="synthetic")
parser.add_argument("--run_name", type=str, default="exp01")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--weights", type=str, default="./model.pt")

args, _ = parser.parse_known_args()
args.save_path = os.path.join("./results", args.run_name)
os.makedirs(args.save_path, exist_ok=True)

show_dict(dict(args._get_kwargs()))


def build_model(model_cls) -> nn.Module:
    return model_cls()


model_table = {
    "cnn-wa": partial(build_model, cnn.Model),
    "cnn-woa": partial(build_model, cnn.Model),
    "rescnn-wa": partial(build_model, rescnn.Model),
    "rescnn-woa": partial(build_model, rescnn.Model),
    "deepcubenet1d-wa": partial(build_model, deepcubenet1d.Model),
    "deepcubenet1d-woa": partial(build_model, deepcubenet1d.Model),
    "resunet-wa": partial(build_model, resunet.Model),
    "resunet-woa": partial(build_model, resunet.Model),
}

dataset_table = {
    "synthetic": partial(dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic.mat"),
    "synthetic-n15db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n15db.mat"
    ),
    "synthetic-n20db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n20db.mat"
    ),
    "synthetic-n25db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n25db.mat"
    ),
    "synthetic-n30db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n30db.mat"
    ),
    "synthetic-n35db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n35db.mat"
    ),
    "synthetic-n40db": partial(
        dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-synthetic-n40db.mat"
    ),
    "measured": partial(dataset.Spectrum, split="test", root="/root/spec/datasets/spec-data-measured.mat"),
    "drink-pink": partial(dataset.Spectrum, split="pink", root="/root/spec/datasets/spec-data-drink.mat"),
    "drink-gold": partial(dataset.Spectrum, split="gold", root="/root/spec/datasets/spec-data-drink.mat"),
    "drink-pyellow": partial(dataset.Spectrum, split="pyellow", root="/root/spec/datasets/spec-data-drink.mat"),
    "drink-blue": partial(dataset.Spectrum, split="blue", root="/root/spec/datasets/spec-data-drink.mat"),
    "drink-purple": partial(dataset.Spectrum, split="purple", root="/root/spec/datasets/spec-data-drink.mat"),
}

rank = os.environ.get("LOCAL_RANK", 0)
device = torch.device("cuda", rank)

model_cls = model_table[args.model_name]
dataset_cls = dataset_table[args.dataset_name]

dataset_test = dataset_cls()
loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size)

model = model_cls().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device)["model"])


def test():
    model.eval()

    x_list, xr_list, error_list, error_reduced_list = [], [], [], []
    bar = tqdm(loader_test)
    for idx, (x, y) in enumerate(bar):
        # x: (N, 1, 350)
        # y: (N, 1, 36)
        x, y = x.to(device), y.to(device)

        xr = model(y)
        error = x - xr
        error_reduced = error.pow(2).mean(dim=-1).view(-1)

        x_list.append(x.squeeze(1))
        xr_list.append(xr.squeeze(1))
        error_list.append(error.squeeze(1))
        error_reduced_list.append(error_reduced)

    x_list = torch.cat(x_list).cpu().numpy()
    xr_list = torch.cat(xr_list).cpu().numpy()
    error_list = torch.cat(error_list).cpu().numpy()
    error_reduced_list = torch.cat(error_reduced_list).cpu().numpy()

    np.savetxt(os.path.join(args.save_path, "x.csv"), x_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "xr.csv"), xr_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "error.csv"), error_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "error_reduced.csv"), error_reduced_list, delimiter=",")
    np.savetxt(os.path.join(args.save_path, "At.csv"), model.At.weight.data.cpu().numpy(), delimiter=",")

    print(f"total {len(error_reduced_list)} samples, avg MSE: {sum(error_reduced_list) / len(error_reduced_list)}")


with torch.no_grad():
    test()
