from itertools import product
import time, datetime
import wandb

pjt_name = "jioh0826/cs-spec"

table_dataset = [
    "synthetic",
    "synthetic-n40db",
    "synthetic-n35db",
    "synthetic-n30db",
    "synthetic-n25db",
    "synthetic-n20db",
    "synthetic-n15db",
    "measured",
    "drink-pink",
    "drink-gold",
    "drink-pyellow",
    "drink-blue",
    "drink-purple",
]

table_model = [
    "cnn-wa",
    "cnn-wagu",
    "cnn-woa",
    "rescnn-wa",
    "rescnn-wagu",
    "rescnn-woa",
    "resunet-wa",
    "resunet-wagu",
    "resunet-woa",
]

num_exp = "01"

api = wandb.Api()
start = time.time()
runs = list(api.runs(pjt_name))

table_run = {run.name: run for run in api.runs(pjt_name)}

resume = 22
skip = {}
# skip = {22}

for idx, (dataset_name, model_name) in enumerate(product(table_dataset, table_model)):

    if idx < resume:
        continue

    if idx in skip:
        continue

    run_name = f"{model_name}-{dataset_name}-{num_exp}"
    run = table_run[run_name]
    print(run_name, run.name, run.id)
    print(f"[{idx:02d}] download from {run.name}")

    run.file(f"results/{run.name}/y.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/x.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/xr.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/error.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/error_reduced.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/psnr.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/At_out.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/At.csv").download(exist_ok=True)

    print(f"duration: {datetime.timedelta(seconds=time.time() - start)}\n" + "-" * 50)
