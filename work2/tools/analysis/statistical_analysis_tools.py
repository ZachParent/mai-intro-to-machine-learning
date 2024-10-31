import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math
from pypalettes import load_cmap

# create a color palette
alexandrite = load_cmap("Alexandrite")
emrld = load_cmap("Emrld", reverse=True)
colors = alexandrite.colors
plt.style.use("default")


def custom_boxplot(ax, data):
    ax.boxplot(
        data,
        patch_artist=True,  # Fill boxes with color
        medianprops={"color": colors[3], "linewidth": 2},  # Style median lines
        flierprops={
            "marker": "o",
            "markerfacecolor": colors[4],
            "markersize": 4,
        },  # Style outlier points
        boxprops={"facecolor": colors[2], "alpha": 0.7},  # Style boxes
        whiskerprops={"linestyle": "--"},  # Style whiskers
        capprops={"linewidth": 2},  # Style caps
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)  # Put grid behind plot elements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.7)


def get_knn_model_label(model_row):
    k_map = {1: "k1", 3: "k3", 5: "k5", 7: "k7"}
    distance_map = {
        "EuclideanDistance": "Euc",
        "ManhattanDistance": "Man",
        "ChebyshevDistance": "Cheb",
    }
    voting_map = {
        "MajorityClassVote": "Maj",
        "InverseDistanceWeightedVote": "Dis",
        "ShepardsWorkVote": "Shp",
    }
    weighting_map = {
        "EqualWeighting": "Eq",
        "InformationGainWeighting": "Inf",
        "ReliefFWeighting": "Rlf",
    }

    return f"{k_map[model_row['k']]}{distance_map[model_row['distance_func']]}{voting_map[model_row['voting_func']]}{weighting_map[model_row['weighting_func']]}"


def get_knn_reduction_model_label(model_row):
    reduction_map = {"control": "Ctrl", "GGCN": "Ggcn", "ENNTH": "Ennth", "Drop3": "Drop3"}

    return f"{reduction_map[model_row['reduction_func']]}{get_knn_model_label(model_row)}"


def get_svm_model_label(model_row):
    c_map = {1: "C1", 3: "C3", 5: "C5", 7: "C7"}
    kernel_map = {"linear": "Lin", "rbf": "Rbf", "poly": "Poly", "sigmoid": "Sig"}

    return f"{c_map[model_row['C']]}{kernel_map[model_row['kernel_type']]}"


def friedman_test(df_with_folds, fold_cols):
    results = {i: df_with_folds.loc[i, fold_cols] for i in df_with_folds.index}
    f1_df = pd.DataFrame(results)
    statistic, p_value = stats.friedmanchisquare(*[f1_df[k] for k in f1_df])
    return statistic, p_value


def linear_sample(df_with_folds, num_samples=8):
    num_samples = min(num_samples, len(df_with_folds))
    samples_indices = np.linspace(0, len(df_with_folds) - 1, num_samples, dtype=int)
    return df_with_folds.loc[samples_indices, :]


def top_samples(df_with_folds, num_samples=8):
    num_samples = min(num_samples, len(df_with_folds))
    return df_with_folds.sort_values(by="mean_f1_score", ascending=False).head(num_samples)


def get_p_values_df(
    model_results_dfs,
    models=["KNN", "SVM", "KNN-Reduction"],
    num_sample_options=[4, 8, 16],
    sample_types=["linear", "top"],
):
    data = []

    for sample_type, num_samples, (model, df) in itertools.product(
        sample_types, num_sample_options, zip(models, model_results_dfs)
    ):
        sample_func = linear_sample if sample_type == "linear" else top_samples

        p_value = friedman_test(sample_func(df, num_samples))[1]
        data.append(
            {
                "model": model,
                "num_samples": num_samples,
                "sample_type": sample_type,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(data)


# TODO: only plot one point for KNN-Reduction
def plot_p_values_vs_num_samples(
    axes, p_values_df, models, sample_types, num_sample_options, alpha=0.05
):
    for idx, model in enumerate(models):
        model_data = p_values_df[p_values_df["model"] == model]

        for sample_type in sample_types:
            data = model_data[model_data["sample_type"] == sample_type]

            # Plot the line
            axes[idx].plot(
                data["num_samples"],
                data["p_value"],
                marker="o",
                label=f"{sample_type} sampling",
                color=alexandrite(0.3 if sample_type == "linear" else 0.6),
            )

            # Scatter significant points (green circle)
            significant = data[data["p_value"] < alpha]
            axes[idx].scatter(
                significant["num_samples"],
                significant["p_value"],
                color="green",
                marker="o",
                s=100,
                zorder=3,
            )

            # Scatter non-significant points (red x)
            non_significant = data[data["p_value"] >= alpha]
            axes[idx].scatter(
                non_significant["num_samples"],
                non_significant["p_value"],
                color="red",
                marker="x",
                s=100,
                zorder=3,
            )

        axes[idx].set_title(f"{model}")
        if idx == 0:  # Only show y-label for first subplot
            axes[idx].set_ylabel("P-value")
        axes[idx].set_yscale("log")  # Log scale for y-axis
        axes[idx].set_xscale("log")  # Log scale for x-axis
        axes[idx].set_xticks(num_sample_options)
        axes[idx].set_xticklabels(num_sample_options)
        axes[idx].axhline(
            y=alpha, color="r", linestyle="--", alpha=0.5, label="Significance Threshold"
        )  # Add significance threshold line
        axes[idx].grid(True)
        axes[idx].legend()

    axes[len(axes) // 2].set_xlabel("Number of Samples")


def plot_independent_effects(df, x_cols, y_cols=["mean_f1_score"]):
    cols = len(x_cols) * len(y_cols)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    for ax, (x_col, y_col) in zip(axes.flatten(), itertools.product(x_cols, y_cols)):
        # Sort unique values if numeric
        unique_vals = df[x_col].unique()
        if np.issubdtype(unique_vals.dtype, np.number):
            unique_vals.sort()
            sorted_data = [df[df[x_col] == val][y_col].tolist() for val in unique_vals]
            custom_boxplot(ax, sorted_data)
        else:
            custom_boxplot(ax, df.groupby(x_col)[y_col].apply(list))

        # Convert categorical values to numeric positions for xticks
        positions = np.arange(len(unique_vals)) + 1
        ax.set_xticks(positions)
        ax.set_xticklabels(unique_vals, rotation=15)

        x_col_label = x_col.replace("_", " ").capitalize()
        y_col_label = y_col.replace("_", " ").capitalize() + ("(s)" if "time" in y_col else "")
        ax.set_title(f"{y_col_label} vs {x_col_label}", fontsize=16, fontweight="bold")
        ax.set_ylabel(y_col_label, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col_label, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_interactions(df, col_names):
    num_cols = len(col_names)
    fig, axes = plt.subplots(num_cols, num_cols, figsize=(5 * num_cols, 4 * num_cols))

    # Make axes 2D if it's 1D
    if num_cols == 1:
        axes = np.array([[axes]])
    elif num_cols == 2:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, ((col_name1, col_name2), ax) in enumerate(
        zip(itertools.product(col_names, repeat=2), axes.flatten())
    ):
        # Skip plots above diagonal
        row = i // len(col_names)
        col = i % len(col_names)
        if col > row:
            ax.set_visible(False)
            continue

        if col_name1 == col_name2:
            # Diagonal plots: show distribution for single variable
            custom_boxplot(ax, df.groupby(col_name1)["mean_f1_score"].apply(list))
        else:
            # Off-diagonal plots: show interaction between variables
            pivot_table = df.pivot_table(
                values="mean_f1_score", index=col_name1, columns=col_name2, aggfunc="mean"
            )
            sns.heatmap(
                pivot_table,
                ax=ax,
                cmap=emrld,
                annot=True,
                fmt=".3f",
                cbar=False,
            )

        # Remove all ticks initially
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Add row labels and y-ticks only on the leftmost plots
    for i in range(num_cols):
        axes[i, 0].set_ylabel(col_names[i], fontsize=20, rotation=90)
        if i > 0:
            unique_vals = (
                sorted(df[col_names[i]].unique())
                if pd.api.types.is_numeric_dtype(df[col_names[i]])
                else df[col_names[i]].unique()
            )
            axes[i, 0].set_yticks(np.arange(len(unique_vals)) + 0.5)
            axes[i, 0].set_yticklabels(unique_vals, rotation=15)

    # Add column labels and x-ticks only on the bottom plots
    for j in range(num_cols):
        unique_vals = (
            sorted(df[col_names[j]].unique())
            if pd.api.types.is_numeric_dtype(df[col_names[j]])
            else df[col_names[j]].unique()
        )
        if j == num_cols - 1:
            axes[-1, j].set_xticks(np.arange(len(unique_vals)))
        else:
            axes[-1, j].set_xticks(np.arange(len(unique_vals)) + 0.5)
        axes[-1, j].set_xticklabels(unique_vals, rotation=15)
        axes[-1, j].set_xlabel(col_names[j], fontsize=20, labelpad=20)

    plt.tight_layout()
    return fig


def get_ranked_folds(results_df, fold_cols):
    rankings = results_df.copy()
    # Calculate rankings for each fold
    for fold in fold_cols:
        fold_ranks = rankings[fold].rank(ascending=False, method="average")
        rankings[fold] = fold_ranks

    return rankings


def plot_ranked_folds(ax, ranked_folds_df, fold_cols):
    # Create boxplot with custom style
    custom_boxplot(ax, np.array([ranked_folds_df.loc[:, col] for col in fold_cols]))

    ax.set_xlabel("Models", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ranked_folds_df["model_label"], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Rank across folds", fontsize=12, fontweight="bold")

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)  # Put grid behind plot elements

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def nemenyi_test(df_with_f1_per_fold, fold_cols):
    results = {i: df_with_f1_per_fold.loc[i, fold_cols] for i in df_with_f1_per_fold.index}
    f1_df = pd.DataFrame(results)
    return sp.posthoc_nemenyi_friedman(f1_df)


def get_top_values(df, col_names, num_to_select=2, models_to_consider=10):
    top_values = {}

    for col_name in col_names:
        best_scoring_models = df.sort_values(by="mean_f1_score", ascending=False).head(
            models_to_consider
        )
        top_values[col_name] = (
            best_scoring_models[col_name].value_counts().nlargest(num_to_select).index.tolist()
        )

    return top_values


def get_models_with_top_values(df, top_values):
    result = df[
        df["k"].isin(top_values["k"])
        & df["distance_func"].isin(top_values["distance_func"])
        & df["voting_func"].isin(top_values["voting_func"])
        & df["weighting_func"].isin(top_values["weighting_func"])
    ].sort_values(by="mean_f1_score", ascending=False)
    result.reset_index(drop=True, inplace=True)
    return result


def analyze_parameters(df, nemenyi_results, alpha=0.05):
    # 1. Main Effects Analysis
    print("=== Main Effects ===")
    for param in ["k", "weighting_func", "voting_func", "distance_func"]:
        means = df.groupby(param)["mean_f1_score"].mean()
        print(f"\n{param} effects:")
        print(means)

    # 2. Interaction Analysis
    print("\n=== Parameter Interactions ===")
    interactions = (
        df.groupby(["k", "weighting_func", "voting_func", "distance_func"])["mean_f1_score"]
        .mean()
        .unstack()
    )
    print(interactions)

    # 3. Find Best Combinations
    best_combos = df.nlargest(3, "mean_f1_score")
    print("\n=== Top 3 Parameter Combinations ===")
    print(best_combos[["k", "weighting_func", "voting_func", "distance_func", "mean_f1_score"]])

    # 4. Statistical Significance Summary
    print("\n=== Significant Differences ===")
    alpha = 0.05
    significant_pairs = []
    for i in nemenyi_results.index:
        for j in nemenyi_results.columns:
            if i < j and nemenyi_results.loc[i, j] < alpha:
                significant_pairs.append((i, j, nemenyi_results.loc[i, j]))

    for pair in sorted(significant_pairs, key=lambda x: x[2]):
        print(f"{pair[0]} vs {pair[1]}: p={pair[2]:.4f}")


def get_significant_pairs(nemenyi_results, models_with_top_values, alpha=0.05):

    significant_pairs = []
    for i in nemenyi_results.index:
        for j in nemenyi_results.columns:
            if i < j and nemenyi_results.loc[i, j] < alpha:
                significant_pairs.append((i, j, nemenyi_results.loc[i, j]))

    significant_pairs_df = models_with_top_values.iloc[
        list(set(np.array([[pair[0], pair[1]] for pair in significant_pairs]).flatten()))
    ]
    return significant_pairs_df