from itertools import product
import time, datetime
import wandb

pjt_name = "jioh0826/cs-spec"

table_dataset = [
    "synthetic",
    "synthetic-n40db",
    "synthetic-n30db",
    "synthetic-n20db",
    "synthetic-n35db",
    "synthetic-n25db",
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
table_run = {run.name: run for run in api.runs(pjt_name)}
for dataset_name, model_name in product(table_dataset, table_model):
    run = table_run[f"{model_name}-{dataset_name}-{num_exp}"]
    run.file(f"results/{run.name}/y.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/x.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/xr.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/error.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/error_reduced.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/psnr.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/At_out.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/At.csv").download(exist_ok=True)
    print(f"download from {run.name}, duration: {datetime.timedelta(seconds=time.time() - start)}")
