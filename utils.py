import os, sys
from functools import wraps
import json
import time, datetime

import numpy as np
import torch


def batch_loader(iterable, batch_size, shuffle=False):
    indices = list(range(len(iterable)))

    import random

    if shuffle:
        random.shuffle(indices)

    for batch_idx in range(0, len(iterable) // batch_size + 1):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(iterable))

        batch_idx = indices[start_idx:end_idx]
        yield [iterable[idx] for idx in batch_idx]


def tpr_at_fpr(tpr: np.ndarray, fpr: np.ndarray, points: list[float] = [1e-4, 1e-3, 1e-2, 1e-1]):
    results = {}
    for p, pt in enumerate(points):
        for i in range(len(fpr) - 1):
            if fpr[i] <= pt and pt <= fpr[i + 1]:
                results[f"{pt:.0E}"] = [tpr[i].item()]

    return results


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def rank_zero_only(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            return fn(*args, **kwargs)
        return None

    return wrapper


@rank_zero_only
def rank_zero_exec(fn, *args, **kwargs):
    return fn(*args, **kwargs)


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)


@rank_zero_only
def save_dict_json(d, path, name="args.json"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), "w") as f:
        json.dump(d, f, indent=4)


@rank_zero_only
def show_dict(d: dict):
    print("-" * 100)
    print("\n".join(f"{k}: {v}" for k, v in d.items()))
    print("-" * 100)
