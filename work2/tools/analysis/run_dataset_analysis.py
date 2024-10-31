# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import circlify
from pywaffle import Waffle
import os
from tools.preprocess import load_datasets

plt.style.use("default")

def load_data():
    SCRIPT_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
    REPORTS_DIR = os.path.join(SCRIPT_DIR, "../../reports")

    mushroom_train_path = f"{DATA_DIR}/raw/mushroom/*train.arff"
    mushroom_test_path = f"{DATA_DIR}/raw/mushroom/*test.arff"
    mushroom_train_dfs = load_datasets(mushroom_train_path)
    mushroom_test_dfs = load_datasets(mushroom_test_path)

    hepatitis_train_path = f"{DATA_DIR}/raw/hepatitis/*train.arff"
    hepatitis_test_path = f"{DATA_DIR}/raw/hepatitis/*test.arff"
    hepatitis_train_dfs = load_datasets(hepatitis_train_path)
    hepatitis_test_dfs = load_datasets(hepatitis_test_path)

    full_mushroom_df = pd.concat([mushroom_train_dfs[0], mushroom_test_dfs[0]])
    full_hepatitis_df = pd.concat([hepatitis_train_dfs[0], hepatitis_test_dfs[0]])

    return full_mushroom_df, full_hepatitis_df, REPORTS_DIR

def plot_dataset_partitions(full_mushroom_df, full_hepatitis_df, REPORTS_DIR):
    hepatitis_die_count = len(full_hepatitis_df[full_hepatitis_df["Class"] == "DIE"])
    hepatitis_live_count = len(full_hepatitis_df[full_hepatitis_df["Class"] == "LIVE"])
    total_hepatitis_count = len(full_hepatitis_df)

    mushroom_poisonous_count = len(full_mushroom_df[full_mushroom_df["class"] == "p"])
    mushroom_edible_count = len(full_mushroom_df[full_mushroom_df["class"] == "e"])
    total_mushroom_count = len(full_mushroom_df)

    data = [
        {
            "id": "All Data",
            "datum": total_hepatitis_count + total_mushroom_count,
            "children": [
                {
                    "id": "Mushrooms",
                    "datum": total_mushroom_count,
                    "children": [
                        {"id": "Edible", "datum": mushroom_edible_count},
                        {"id": "Poisonous", "datum": mushroom_poisonous_count},
                    ],
                },
                {
                    "id": "Hepatitis",
                    "datum": total_hepatitis_count,
                    "children": [
                        {"id": "Live", "datum": hepatitis_live_count},
                        {"id": "Die", "datum": hepatitis_die_count},
                    ],
                },
            ],
        }
    ]

    circles = circlify.circlify(
        data, show_enclosure=False, target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Partitions of the datasets by Class", fontsize=24, fontweight="bold")
    ax.axis("off")

    lim = max(
        max(
            abs(circle.x) + circle.r,
            abs(circle.y) + circle.r,
        )
        for circle in circles
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    for circle in circles:
        if circle.level != 2:
            continue
        x, y, r = circle
        ax.add_patch(plt.Circle((x, y), r, alpha=0.5, linewidth=2, color="lightblue"))

    for circle in circles:
        if circle.level != 3:
            continue
        x, y, r = circle
        label = f"{circle.ex['id']}:\n{circle.ex['datum']}"
        ax.add_patch(plt.Circle((x, y), r, alpha=0.5, linewidth=2, color="#69b3a2"))
        plt.annotate(label, (x, y), ha="center", va="center", color="white")

    for circle in circles:
        if circle.level != 2:
            continue
        x, y, r = circle
        label = f"{circle.ex['id']}:\n{circle.ex['datum']}"
        plt.annotate(
            label,
            (x, y + 3 * r / 4),
            va="center",
            ha="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.5),
        )

    plt.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/figures/dataset-partitions.png", dpi=300)

def plot_hepatitis_distribution(full_hepatitis_df, REPORTS_DIR):
    hepatitis_die_count = len(full_hepatitis_df[full_hepatitis_df["Class"] == "DIE"])
    hepatitis_live_count = len(full_hepatitis_df[full_hepatitis_df["Class"] == "LIVE"])
    
    data = {"Live": hepatitis_live_count, "Die": hepatitis_die_count}
    repartition = [f"{k} ({round(v / sum(data.values()) * 100)}%)" for k, v in data.items()]

    fig = plt.figure(
        FigureClass=Waffle,
        figsize=(12, 3),
        rows=5,
        columns=31,
        values=data,
        colors=("#69b3a2", "#404040"),
        title={
            "label": "Class distribution in the hepatitis dataset",
            "fontdict": {"fontsize": 16, "fontweight": "bold"},
        },
        labels=repartition,
        legend={
            "loc": "lower left",
            "bbox_to_anchor": (0, -0.25),
            "ncol": len(data),
            "fontsize": 12,
        },
    )
    plt.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/figures/hepatitis-class-distribution.png", dpi=300)

def plot_mushroom_distribution(full_mushroom_df, REPORTS_DIR):
    mushroom_poisonous_count = len(full_mushroom_df[full_mushroom_df["class"] == "p"])
    mushroom_edible_count = len(full_mushroom_df[full_mushroom_df["class"] == "e"])
    
    data = {"Edible": mushroom_edible_count, "Poisonous": mushroom_poisonous_count}
    repartition = [f"{k} ({round(v / sum(data.values()) * 100)}%)" for k, v in data.items()]

    fig = plt.figure(
        FigureClass=Waffle,
        figsize=(12, 3),
        rows=5,
        columns=34,
        values=data,
        colors=("#8B4513", "#FF9999"),
        title={
            "label": "Class distribution in the mushroom dataset",
            "fontdict": {"fontsize": 16, "fontweight": "bold"},
        },
        labels=repartition,
        legend={
            "loc": "lower left",
            "bbox_to_anchor": (0, -0.25),
            "ncol": len(data),
            "fontsize": 12,
        },
    )
    plt.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/figures/mushroom-class-distribution.png", dpi=300)

# Load data and generate plots
# %%
full_mushroom_df, full_hepatitis_df, REPORTS_DIR = load_data()
# %%
plot_dataset_partitions(full_mushroom_df, full_hepatitis_df, REPORTS_DIR)
# %%
plot_hepatitis_distribution(full_hepatitis_df, REPORTS_DIR)
# %%
plot_mushroom_distribution(full_mushroom_df, REPORTS_DIR)

# %%
