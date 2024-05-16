import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt


class Spectrum:
    def __init__(self, root, split):
        data = loadmat(root)[split][0][0]

        self.x = torch.tensor(data[0], dtype=torch.float32)
        self.y = torch.tensor(data[1], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx][None, :], self.y[idx][None, :]


def main():
    ds = Spectrum("../datasets/measured.mat", "test")

    x, y = next(iter(ds))
    print(x.shape, y.shape, len(ds))

    fig, axes = plt.subplots(2, 1, height_ratios=[1, 1])

    fig.suptitle("Measured Test Spectra 1")

    axes[0].plot(x.view(-1).numpy(), "k")
    axes[0].set_title("input spectra")

    axes[1].plot(y.view(-1).numpy(), "k")
    axes[1].set_title("measurement")

    fig.tight_layout()
    fig.savefig("test_spectra.png")
    plt.close("all")


if __name__ == "__main__":
    main()
