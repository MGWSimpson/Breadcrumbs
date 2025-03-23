import json
import os
import datetime

import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

COLOR = "black"

mpl.rcParams["text.color"] = COLOR
mpl.rcParams["axes.labelcolor"] = COLOR
mpl.rcParams["xtick.color"] = COLOR
mpl.rcParams["ytick.color"] = COLOR
mpl.rcParams["figure.dpi"] = 200
sns.set(style="darkgrid")


def convert_to_pandas(human_scores, machine_scores):
    human_scores = human_scores["score"]
    machine_scores = machine_scores["score"]

    df = pd.DataFrame(
        {"score": human_scores + machine_scores, "class": [0] * len(human_scores) + [1] * len(machine_scores)}
    )
    return df


def save_json(data, save_path):
    data.end_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    with open(os.path.join(save_path, "experiments_details.json"), "w", encoding="utf-8") as f:
        json.dump(data.__dict__, f, ensure_ascii=False, indent=4)


"""
 df = pd.DataFrame(
        {"score": human_scores + machine_scores, "class": [0] * len(human_scores) + [1] * len(machine_scores)}
    )
"""

def scatter_plot_save(args, score_df): 
    

    sns.scatterplot(
        data=score_df,
        y="score",
        x=[0]*len(score_df),  # no x-axis meaning, all scores aligned
        hue="class",
        palette="Set2",
        alpha=0.7,
    )

    plt.xticks([])
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("Score Distribution Colored by True Class")
    plt.legend(title="Class", labels=["Human (0)", "Machine (1)"])
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{args.experiment_path}/scores.png", bbox_inches='tight')



def save_experiment(args, score_df, fpr, tpr, f1_score, roc_auc, tpr_at_fpr_0_01):
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log")

    annotation = f"ROC AUC: {roc_auc:.4f}\nF1 Score: {f1_score:.2f}\nTPR at 0.01% FPR:{100 * tpr_at_fpr_0_01:.2f}%"
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=annotation)
    display.plot(ax=ax, linestyle="--")
    ax.set_title(f"{args.dataset_name} (n={len(score_df)})\nMachine Text from {args.machine_text_source}")

    fig.savefig(f"{args.experiment_path}/performance.png", bbox_inches='tight')
    score_df.to_csv(f"{args.experiment_path}/score_df.csv", index=False)
    scatter_plot_save(args, score_df)
    save_json(args, args.experiment_path)
