# %%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from distutils.util import strtobool

plt.style.use(["science", "no-latex"])


def config_to_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return strtobool(value)
    else:
        raise ValueError("Wrong parameter")


# %%
def extract_metrics(exp):

    proc = dict()
    proc["dataset"] = exp["DATASET"]
    proc["num_workers"] = exp["NUM_WORKERS"]
    proc["batch_size"] = exp["BATCH_SIZE"]
    proc["rep"] = exp["REP"]
    proc["library"] = exp["LIBRARY"]
    proc["remote"] = config_to_bool(exp["REMOTE"])
    proc["cutoff"] = exp["CUTOFF"]

    if config_to_bool(exp["FILTERING"]):
        mode = "filtering"
    elif config_to_bool(exp["DISTRIBUTED"]):
        mode = "distributed"
    else:
        mode = "default"
    proc["mode"] = mode

    runtimes = [data["0"]["all_runtimes"]]
    if "1" in data:
        runtimes += [data["1"]["all_runtimes"]]
    proc["avg_speed"] = compute_average_speed(runtimes, proc["batch_size"])

    return proc


def compute_average_speed(runtimes: list, batch_size: int):
    """
    runtimes: list of the batch times for each worker.
    """

    speed_list = [np.mean([batch_size / t for t in rt[1:]]) for rt in runtimes]
    return np.sum(speed_list)


# %%
PATH = Path("/Users/danski/workspace/dataloader-benchmarks/downloaded_results")
experiments = []
for file in PATH.rglob("*.json"):
    with open(file, "r") as fh:
        data = json.load(fh)
    proc = extract_metrics(data)
    experiments += [proc]
df = pd.DataFrame(experiments)


# %%

# %%


# %%
# %%
p2 = Path("/Users/danski/workspace/dataloader-benchmarks/plots")
p2.mkdir(parents=True, exist_ok=True)


datasets = ["cifar10", "random", "coco"]
modes = ["default", "distributed", "filtering"]

for ds in datasets:
    for m in modes:
        df_ = df[(df["dataset"] == ds) & (df["mode"] == m) & (df["remote"] == False)]

        try:
            title = f"{ds}_{m}_batch"

            fig, ax = plt.subplots()
            sns.boxplot(
                data=df_,
                x="batch_size",
                y="avg_speed",
                hue="library",
                dodge=True,
                ax=ax,
            )
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=3, loc="lower center")
            fig.savefig(p2 / (title + ".jpg"))

        except Exception as e:
            print(e)
            print("Failed", ds, m)
        title = f"{ds}_{m}_workers"
        fig2, ax2 = plt.subplots()
        sns.boxplot(
            data=df_,
            x="num_workers",
            y="avg_speed",
            hue="library",
            dodge=True,
            ax=ax2,
        )
        ax2.set_title(title)
        ax2.legend(bbox_to_anchor=(0.5, 1.1), ncol=3, loc="lower center")
        fig2.savefig(p2 / (title + ".jpg"))

# %%
