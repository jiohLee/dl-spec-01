import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

table_dataset = (
    "synthetic",
    # "synthetic-n15db",
    # "synthetic-n20db",
    # "synthetic-n25db",
    # "synthetic-n30db",
    # "synthetic-n35db",
    # "synthetic-n40db",
    # "measured",
    # "drink-pink",
    # "drink-gold",
    # "drink-pyellow",
    # "drink-blue",
    # "drink-purple",
)


table_model = (
    # "transformer01-wa",
    # "transformer01-wa-nh1",
    # "transformer01-woa",
    # "transformer01-woa-nh1",
    # "transformer02-wa",
    # "transformer02-wa-nh1",
    # "transformer02-woa",
    # "transformer02-woa-nh1",
    # "transformer03-wa",
    # "transformer03-wa-nh1",
    # "transformer03-woa",
    # "transformer03-woa-nh1",
    # "transformer04-wa",
    # "transformer04-wa-nh1",
    # "transformer04-woa",
    # "transformer04-woa-nh1",
    # "transformer05",
    "transformer05-nh1",
)

num_exp = os.environ["num_exp"]

table_results = {}


def save_results_table():
    data = {"model": table_model}
    for dataset_name in table_dataset:
        data[dataset_name + "_mse"] = [v["error_reduced"].mean() for v in table_results[dataset_name].values()]
        data[dataset_name + "_psnr"] = []

        for model_name in table_model:
            result = table_results[dataset_name][model_name]

            avg_psnr = sum(
                10 * np.log10(np.power(np.max(x) - np.min(x), 2) / mse)
                for x, mse in zip(result["x"], result["error_reduced"].reshape(-1))
            ) / len(result["error_reduced"].reshape(-1))
            data[dataset_name + "_psnr"].append(avg_psnr)

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f"/root/spec/results/results_trans-{num_exp}.csv")


def draw_figure_recon(x, xr, error, error_reduced):
    fig, axes = plt.subplots(2, 1, height_ratios=[3, 1])
    fig.suptitle(f"MSE: {error_reduced:.05f}")

    axes[0].plot(x, "k--")
    axes[0].plot(xr, "r", linewidth=1)
    axes[0].set_title("spectra")

    axes[1].plot(error, color=(0.5, 0.5, 0.5))
    axes[1].set_title("error graph")
    axes[1].set_ylim(ymax=1, ymin=-1)

    fig.tight_layout()
    return fig


def draw_figure_At(At):

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)

    mat = ax.matshow(At.T, aspect="auto")
    fig.colorbar(mat)
    return fig


def draw_figure_recon_attn_map(y, x, xr, attn_map):
    fig, axes = plt.subplot_mosaic("ABBB;ABBB;CBBB")
    fig.set_size_inches(10, 7.5)

    axes["A"].plot(x, "k--")
    axes["A"].plot(xr, "r", linewidth=1)
    axes["A"].set_title("spectra")

    axes["C"].plot(y, color=(0.5, 0.5, 0.5))
    axes["C"].set_title("measurements")
    axes["C"].set_ylim(ymax=1, ymin=0)

    axes["B"].matshow(attn_map.T, aspect="auto")
    axes["B"].set_title("attention map")
    fig.tight_layout()
    return fig


def save_figures():

    for dataset_name, model_name in product(table_dataset, table_model):
        result = table_results[dataset_name][model_name]
        run_name = f"{model_name}-{dataset_name}-{num_exp}"

        save_path = os.path.join("/root/spec/results", run_name)

        # print(f"save At matrix figure to {save_path}")
        # fig = draw_figure_At(result["At"])
        # fig.savefig(os.path.join(save_path, f"0000.png"))
        # plt.close("all")

        print(f"save reconstruction results figures to {save_path}")
        for i, _ in tqdm(enumerate(range(len(result["x"])), start=1)):
            if i > 2:
                break

            x = result["x"][i - 1]
            xr = result["xr"][i - 1]
            error = result["error"][i - 1]
            mse = result["error_reduced"].reshape(-1)[i - 1]

            fig = draw_figure_recon(x, xr, error, mse)
            fig.savefig(os.path.join(save_path, f"{i:04d}.png"))
            plt.close("all")

            y = result["y"][i - 1]
            attn_maps = result["attn_maps"][i - 1]

            for h in range(len(attn_maps)):
                fig = draw_figure_recon_attn_map(y, x, xr, attn_maps[h])
                fig.savefig(os.path.join(save_path, f"{i:04d}_attn_map_{(h + 1):d}.png"))
                plt.close("all")


def main():

    for i, (dataset_name, model_name) in enumerate(product(table_dataset, table_model), start=1):
        table_results[dataset_name] = table_results.get(dataset_name, {})

        run_name = f"{model_name}-{dataset_name}-{num_exp}"
        path = os.path.join("/root/spec/results", run_name)
        print(f"{i:03d}: load data from {path}")

        # At = np.loadtxt(os.path.join(path, "At.csv"), delimiter=",")
        y = np.loadtxt(os.path.join(path, "y.csv"), delimiter=",")
        x = np.loadtxt(os.path.join(path, "x.csv"), delimiter=",")
        xr = np.loadtxt(os.path.join(path, "xr.csv"), delimiter=",")
        error = np.loadtxt(os.path.join(path, "error.csv"), delimiter=",")
        error_reduced = np.loadtxt(os.path.join(path, "error_reduced.csv"), delimiter=",")
        attn_maps = np.load(os.path.join(path, "attn_maps.npy"))

        table_results[dataset_name][model_name] = {
            "y": y,
            "x": x,
            "xr": xr,
            "error": error,
            "error_reduced": error_reduced,
            # "At": At,
            "attn_maps": attn_maps,
        }

    save_figures()
    save_results_table()


if __name__ == "__main__":
    main()
