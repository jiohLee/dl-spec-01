import time, datetime
import wandb

pjt_name = "jioh0826/cs-spec"

api = wandb.Api()
start = time.time()
for run in api.runs("jioh0826/cs-spec"):
    run.file(f"results/{run.name}/y.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/x.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/xr.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/error.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/error_reduced.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/At_out.csv").download(exist_ok=True)
    run.file(f"results/{run.name}/At.csv").download(exist_ok=True)
    print(f"download from {run.name}, duration: {datetime.timedelta(seconds=time.time() - start)}")
