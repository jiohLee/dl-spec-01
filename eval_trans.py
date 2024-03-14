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
    transformer01,
    transformer02,
    transformer03,
    transformer04,
    transformer05,
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
attn_maps = transformer01.MultiheadAttentionHook(model.transformer.decoder.layers[-1].mha1)

data = {}


def named_hook(name):

    def hook(module, x, y):
        data[name] = y

    return hook


model.At.register_forward_hook(named_hook("At_out"))


def draw_signal(x, file_name):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    print(x.shape)
    plt.plot(x.reshape(-1))
    plt.show()

    plt.savefig(file_name)
    plt.close("all")


def test():
    model.eval()

    y_list, x_list, xr_list, error_list, error_reduced_list, attn_map_list = [], [], [], [], [], []
    bar = tqdm(loader_test)
    for idx, (x, y) in enumerate(bar):
        # x: (N, 1, 350)
        # y: (N, 1, 36)
        x, y = x.to(device), y.to(device)

        xr = model(y)
        attn_map = attn_maps.data
        error = x - xr
        error_reduced = error.pow(2).mean(dim=-1).view(-1)

        draw_signal(data["At_out"][0][0].cpu().numpy(), "./draw-woa.png")

        break

        # y_list.append(y.squeeze(1))
        # x_list.append(x.squeeze(1))
        # xr_list.append(xr.squeeze(1))
        # error_list.append(error.squeeze(1))
        # error_reduced_list.append(error_reduced)
        # attn_map_list.append(attn_map)

    # y_list = torch.cat(y_list).cpu().numpy()
    # x_list = torch.cat(x_list).cpu().numpy()
    # xr_list = torch.cat(xr_list).cpu().numpy()
    # error_list = torch.cat(error_list).cpu().numpy()
    # error_reduced_list = torch.cat(error_reduced_list).cpu().numpy()
    # attn_map_list = torch.cat(attn_map_list).cpu().numpy()

    # np.savetxt(os.path.join(args.save_path, "y.csv"), y_list, delimiter=",")
    # np.savetxt(os.path.join(args.save_path, "x.csv"), x_list, delimiter=",")
    # np.savetxt(os.path.join(args.save_path, "xr.csv"), xr_list, delimiter=",")
    # np.savetxt(os.path.join(args.save_path, "error.csv"), error_list, delimiter=",")
    # np.savetxt(os.path.join(args.save_path, "error_reduced.csv"), error_reduced_list, delimiter=",")
    # # np.savetxt(os.path.join(args.save_path, "At.csv"), model.At.weight.data.cpu().numpy(), delimiter=",")
    # np.save(os.path.join(args.save_path, "attn_maps.npy"), attn_map_list)

    # print(f"total {len(error_reduced_list)} samples, avg MSE: {sum(error_reduced_list) / len(error_reduced_list)}")


with torch.no_grad():
    test()
