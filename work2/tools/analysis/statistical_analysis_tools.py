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
        medianprops={"color": alexandrite(0.2), "linewidth": 4},  # Style median lines
        flierprops={
            "marker": "o",
            "markerfacecolor": alexandrite(0.4),
            "markeredgewidth": 1,
            "markersize": 4,
        },  # Style outlier points
        boxprops={"facecolor": alexandrite(0.3), "alpha": 0.7},  # Style boxes
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
    reduction_map = {"control": "Ctrl", "GCNN": "Gcnn", "ENNTH": "Ennth", "Drop3": "Drop3"}

    return f"{reduction_map[model_row['reduction_func']]}{get_knn_model_label(model_row)}"


def get_svm_reduction_model_label(model_row):
    reduction_map = {"control": "Ctrl", "GCNN": "Gcnn", "ENNTH": "Ennth", "Drop3": "Drop3"}

    return f"{reduction_map[model_row['reduction_func']]}{get_svm_model_label(model_row)}"


def get_svm_model_label(model_row):
    c_map = {0.05: "C0.05", 0.5: "C0.5", 5: "C5", 50: "C50"}
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


def top_samples(df_with_folds, num_samples=8, by="mean_f1"):
    num_samples = min(num_samples, len(df_with_folds))
    return df_with_folds.sort_values(by=by, ascending=False).head(num_samples)


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

        axes[idx].set_title(f"{model}", fontsize=16, fontweight="bold")
        if idx == 0:  # Only show y-label for first subplot
            axes[idx].set_ylabel("P-value", fontsize=14, fontweight="bold")
        axes[idx].set_yscale("log")  # Log scale for y-axis
        axes[idx].set_xscale("log")  # Log scale for x-axis
        axes[idx].set_xticks(num_sample_options)
        axes[idx].set_xticklabels(num_sample_options)
        axes[idx].axhline(
            y=alpha, color="r", linestyle="--", alpha=0.5, label="Significance Threshold"
        )  # Add significance threshold line
        axes[idx].grid(True)
        axes[idx].legend()

    axes[len(axes) // 2].set_xlabel("Number of Samples", fontsize=14, fontweight="bold")


def plot_independent_effects(df, x_cols, y_cols=["mean_f1"]):
    cols = len(x_cols) * len(y_cols)
    fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 5))

    for ax, (x_col, y_col) in zip(axes.flatten(), itertools.product(x_cols, y_cols)):
        # Sort unique values if numeric
        unique_vals = df[x_col].unique()
        unique_vals.sort()
        sorted_data = [df[df[x_col] == val][y_col].tolist() for val in unique_vals]
        custom_boxplot(ax, sorted_data)

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
            unique_vals = df[col_name1].unique()
            if np.issubdtype(unique_vals.dtype, np.number):
                unique_vals.sort()
            sorted_data = [df[df[col_name1] == val]["mean_f1"].tolist() for val in unique_vals]
            print('custom boxplot of', col_name1)
            custom_boxplot(ax, sorted_data)
        else:
            # Off-diagonal plots: show interaction between variables
            pivot_table = df.pivot_table(
                values="mean_f1", index=col_name1, columns=col_name2, aggfunc="mean"
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
            axes[-1, j].set_xticks(np.arange(len(unique_vals)) + 1)
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
        best_scoring_models = df.sort_values(by="mean_f1", ascending=False).head(
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
    ].sort_values(by="mean_f1", ascending=False)
    result.reset_index(drop=True, inplace=True)
    return result


def expand_data_per_fold(df, x_col="model", metrics=["f1", "train_time", "test_time"]):
    """Reshape data to have one row per fold"""
    expanded_data = []

    for _, row in df.iterrows():
        model = row[x_col]
        for i in range(10):
            fold_data = {x_col: model, "fold": i}
            for metric in metrics:
                fold_data[metric] = row[f"{metric}_{i}"]
            expanded_data.append(fold_data)

    return pd.DataFrame(expanded_data)


def plot_metrics_comparison(axs, df_by_fold, x_col="model"):
    """Create comparison plots for each metric"""
    titles = ["F1 Score", "Train Time", "Test Time"]
    metrics = ["f1", "train_time", "test_time"]

    for i, (title, metric) in enumerate(zip(titles, metrics)):
        data = [
            df_by_fold[df_by_fold[x_col] == x_col_op][metric].values
            for x_col_op in df_by_fold[x_col].unique()
        ]
        custom_boxplot(axs[i], data)
        axs[i].set_xticklabels(df_by_fold[x_col].unique())
        axs[i].set_title(title)
        axs[i].set_xlabel("Model", fontsize=12, fontweight="bold")
        if i == 0:
            axs[i].set_ylabel("F1 Score", fontsize=12, fontweight="bold")
        else:
            axs[i].set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")


def analyze_parameters(df, nemenyi_results, x_cols, alpha=0.05):
    # 1. Main Effects Analysis
    print("=== Main Effects ===")
    for param in x_cols:
        means = df.groupby(param)["mean_f1"].mean()
        print(f"\n{param} effects:")
        print(means)

    # 2. Interaction Analysis
    print("\n=== Parameter Interactions ===")
    interactions = df.groupby(x_cols)["mean_f1"].mean().unstack()
    print(interactions)

    # 3. Find Best Combinations
    best_combos = df.nlargest(3, "mean_f1")
    print("\n=== Top 3 Parameter Combinations ===")
    print(best_combos[x_cols + ["mean_f1"]])

    # 4. Statistical Significance Summary
    print("\n=== Significant Differences ===")
    significant_pairs = []
    for i in nemenyi_results.index:
        for j in nemenyi_results.columns:
            if i < j and nemenyi_results.loc[i, j] < alpha:
                significant_pairs.append((i, j, nemenyi_results.loc[i, j]))

    for pair in sorted(significant_pairs, key=lambda x: x[2]):
        print(f"{pair[0]} vs {pair[1]}: p={pair[2]:.4f}")


def get_significant_pairs(nemenyi_results, alpha=0.05):

    significant_pairs = []
    for i in range(len(nemenyi_results)):
        for j in range(i + 1, len(nemenyi_results)):
            if nemenyi_results.iloc[i, j] < alpha:
                significant_pairs.append((i, j, nemenyi_results.iloc[i, j]))
    return significant_pairs


def get_df_pairs(df, pairs):
    return df.iloc[list(set(np.array([[pair[0], pair[1]] for pair in pairs]).flatten()))]


def plot_reduction_results_scatter(axs, metric_cols_map, metric_pairs, reduction_results):

    for (metric_a, metric_b), ax in zip(
        metric_pairs,
        axs,
    ):
        for idx in reduction_results.index:
            metric_a_values = reduction_results.loc[idx, metric_cols_map[metric_a]].values
            metric_b_values = reduction_results.loc[idx, metric_cols_map[metric_b]].values
            ax.scatter(
                metric_a_values,
                metric_b_values,
                color=colors[idx],
                alpha=0.6,
                label=reduction_results.loc[idx, "reduction_func"],
            )
        ax.set_title(f"{metric_b} vs {metric_a}", fontsize=16, fontweight="bold")
        ax.set_xlabel(metric_a, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric_b, fontsize=14, fontweight="bold")
        ax.legend()
