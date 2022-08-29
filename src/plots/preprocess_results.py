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


def compute_average_speed(runtimes: list, batch_size: int):
    """
    runtimes: list of the batch times for each worker.
    """

    speed_list = [np.mean([batch_size / t for t in rt[1:]]) for rt in runtimes]
    return np.sum(speed_list)


def add_single_runtimes(d, exp):

    accounted_time = 0
    all_runtimes = exp["0"]["all_runtimes"]
    train_loader = exp["0"]["dataloader_start_time"]["train"]
    total_time = exp["0"]["total_training_time"]

    for t, v in enumerate(all_runtimes):
        d[f"time_batch_{t}"] = v
        accounted_time += v

    d["time_train_loader"] = train_loader[1] - train_loader[0]
    accounted_time += train_loader[1] - train_loader[0]
    d["remaining_time"] = total_time - accounted_time


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

    runtimes = [exp["0"]["all_runtimes"]]
    if "1" in exp:
        runtimes += [exp["1"]["all_runtimes"]]
    proc["avg_speed"] = compute_average_speed(runtimes, proc["batch_size"])

    add_single_runtimes(proc, exp)

    return proc


# %%
PATH = Path("/Users/danski/workspace/dataloader-benchmarks/downloaded_results")
experiments = []
for file in PATH.rglob("*.json"):
    with open(file, "r") as fh:
        data = json.load(fh)
    proc = extract_metrics(data)
    experiments += [proc]
df = pd.DataFrame(experiments)

df = df[(df["cutoff"] == 10)].dropna(axis=1, how="all")


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


ds = "random"
nw = 0
rep = 1
bs = 64
m = "default"
df_2 = df.copy()
df_2 = df_2.query("dataset == @ds and num_workers == @nw and batch_size == @bs")
df_2 = df_2.query("rep == @rep and remote == 0 and mode == @m")

ylabels = []
times = []
colors = []
for i, r in df_2.iterrows():
    ylabels.append(r.library)

    colors = ["orange"]

    t_ = [r.time_train_loader]
    for j in range(r.cutoff):
        t_ += [r[f"time_batch_{j}"]]
        colors += ["powderblue"]
    t_ += [r.remaining_time]
    colors += ["coral"]
    times.append(t_)

# %%
fig, ax = plt.subplots()
from matplotlib.patches import Rectangle

h = 0.1
for i, exp in enumerate(times):
    x = 0
    y = i * h * 2
    for j, t in enumerate(exp):
        rec = Rectangle((x, y), t, h, linewidth=1, fc=colors[j], ec="k")

        if j > 0 and j < len(exp) - 1:

            rx, ry = rec.get_xy()
            cx = rx + rec.get_width() / 2.0
            cy = ry + rec.get_height() / 2.0

            print(cx)

            if rec.get_width() > 0.03:
                ax.annotate(
                    str(j - 1),
                    (cx, cy),
                    color="k",
                    weight="bold",
                    fontsize=6,
                    ha="center",
                    va="center",
                )

        x += t
        ax.add_patch(rec)


max_x = max([sum(x) for x in times])
ax.set_xlim([0, max_x * 1.05])
ax.set_ylim([0, 2 * h * (len(times))])

ax.set_yticks([(h / 2) + (h * i * 2) for i in range(len(times))])
ax.set_yticklabels(ylabels)
fig.show()

# %%
