import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import itertools
import logging
from pypalettes import load_cmap
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

alexandrite = load_cmap("Alexandrite")
emrld = load_cmap("Emrld", reverse=True)
colors = alexandrite.colors
plt.style.use("default")

logger = logging.getLogger(__name__)


def plot_interactions_grid(df, col_names, metrics, save_path=None):
    """
    Plots interaction plots for the specified columns and metrics.

    Parameters:
    - df: DataFrame containing the data.
    - col_names: List of column names to include in the grid (e.g., ['fuzzyness', 'n_clusters']).
    - metrics: List of metrics to analyze (e.g., ['ari', 'purity', 'dbi', 'f_measure']).

    Returns:
    - fig: The generated matplotlib figure.
    """
    num_cols = len(col_names)
    num_metrics = len(metrics)

    # Create subplots grid
    fig, axes = plt.subplots(num_cols, num_cols, figsize=(6 * num_cols, 4 * num_cols))

    # If axes is a single subplot, make it iterable
    if num_cols == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)

    # Iterate over the combinations of columns
    for i, ((col_name1, col_name2), ax) in enumerate(
        zip(itertools.product(col_names, repeat=2), axes.flatten())
    ):
        if col_name1 == col_name2:
            # Diagonal: Plot distribution of each metric for the given column
            sns.boxplot(data=df, x=col_name1, y=metrics[0], ax=ax)
            ax.set_title(
                f"Distribution of {metrics[0].upper()} by {col_name1.capitalize()}", fontsize=10
            )
        else:
            # Off-diagonal: Plot interaction heatmap between two columns for each metric
            pivot_table = df.pivot_table(
                values=metrics[0],  # Use the first metric for simplicity
                index=col_name1,
                columns=col_name2,
                aggfunc="mean",
            )
            sns.heatmap(pivot_table, ax=ax, cmap="coolwarm", annot=True, fmt=".3f", cbar=False)
            ax.set_title(
                f"Interaction of {col_name1.capitalize()} and {col_name2.capitalize()} on {metrics[0].upper()}",
                fontsize=10,
            )

        # Remove ticks to make plots cleaner
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Add row labels for columns
    for i in range(num_cols):
        if axes[i, 0] is not None:  # Ensure this is a valid subplot
            axes[i, 0].set_ylabel(col_names[i], fontsize=14, rotation=90)
        if axes[-1, i] is not None:  # Ensure this is a valid subplot
            axes[-1, i].set_xlabel(col_names[i], fontsize=14, rotation=0)

    plt.tight_layout()
    return fig


def plot_interactions(data, model_name, paramsGrid, save_path=None):
    """
    Plots interaction plots for various metrics based on a model and parameter grid.

    Parameters:
    - data (df): metrics pandas df
    - model_name (str): Name of the model to filter for in the dataset (e.g., 'fuzzy_cmeans').
    - paramsGrid (list of str): List of parameter names to include in the interaction plot (e.g., ['fuzzyness']).

    Returns:
    - None: Displays the interaction plot.



    """

    # Filter data for the selected model
    data = data[data["model"] == model_name]

    # Ensure that the parameters in paramsGrid exist in the dataset
    for param in paramsGrid:
        if param not in data.columns:
            raise ValueError(f"Parameter '{param}' not found in the dataset columns.")

    # Set up the plotting style
    sns.set(style="whitegrid")

    # Create interaction plots for each parameter in paramsGrid
    metrics = ["ari", "purity", "dbi", "f_measure"]  # List of metrics to visualize
    for param in paramsGrid:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 5 * len(metrics)), sharex=True)
        if len(metrics) == 1:  # If only one metric, make axes a list for uniform indexing
            axes = [axes]

        # Plot each metric
        for ax, metric in zip(axes, metrics):
            sns.lineplot(x=param, y=metric, data=data, marker="o", ax=ax)
            ax.set_title(f"{metric.upper()} vs {param.capitalize()} for {model_name}")
            ax.set_xlabel(param.capitalize())
            ax.set_ylabel(metric.upper())

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_pairplot(data, vars, save_path=None):
    # vars = ['ari', 'purity', 'dbi', 'f_measure']
    sns.pairplot(data, vars=vars, hue="model")

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_clusters(path, features, save_path=None):
    """
    Example of use

        for path in glob.glob(f'{CLUSTERED_DATA_DIR}/{dataset_name}/{model_name}/*'):
            if 'n_clusters=10' in path:  # change the number of clusters or any other param
                plot_clusters(path=path, features=['0','1']) # for synthetic dataset


    """
    data = pd.read_csv(path)
    config = " ".join(path.split("/")[-1][:-4].split(","))
    labels = data["cluster"].to_list()
    x = str(features[0])
    y = str(features[1])

    plt.figure(figsize=(8, 6))
    plt.scatter(data[x], data[y], c=labels, cmap="viridis", s=50, alpha=0.7)
    plt.colorbar()
    plt.title(f"Clusters visualized using {config}")
    plt.xlabel(x)
    plt.ylabel(y)

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_model_comparisons(data, metric, title, save_path=None):
    """
    Plot comparisons of models across datasets based on a given metric.

    Args:
        data (pd.DataFrame): The dataset containing performance metrics.
        metric (str): The metric to visualize (e.g., 'ari', 'purity').
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="dataset", y=metric, hue="model", errorbar="sd", palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.legend(title="Model", fontsize=10, loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_combined_heatmaps(data, metrics, datasets, models, save_path=None):
    """
    Generate a grid of heatmaps (e.g., 2x2) for multiple metrics.

    Args:
        data (pd.DataFrame): The dataset containing performance metrics.
        metrics (list): List of metrics to plot heatmaps for.
        datasets (list): List of dataset names.
        models (list): List of model names.
    """
    n_metrics = len(metrics)
    n_cols = 2  # Number of columns for the heatmap grid
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows), constrained_layout=True)

    # Flatten axes to easily iterate over them, even for 2D array
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # Pivot table for the heatmap
        pivot_table = data.pivot_table(
            index="model", columns="dataset", values=metric, aggfunc="mean"
        ).reindex(
            index=models, columns=datasets
        )  # Ensure consistent order

        # Plot the heatmap on the respective subplot
        sns.heatmap(
            pivot_table,
            ax=axes[i],
            annot=True,
            cmap="coolwarm",
            cbar=True,
            linewidths=0.5,
            fmt=".2f",
        )
        axes[i].set_title(f"Heatmap for {metric.capitalize()}", fontsize=14)
        axes[i].set_xlabel("Dataset", fontsize=12)
        axes[i].set_ylabel("Model", fontsize=12)

    # Hide any unused subplots (in case metrics < n_rows * n_cols)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_radar_chart(data, dataset_name, metrics, models, save_path=None):
    """
    Generate radar chart to visualize multiple metrics for each model.

    Args:
        data (pd.DataFrame): The dataset containing performance metrics.
        dataset_name (str): The name of the dataset to filter by.
        metrics (list): List of metrics to include in the radar chart.
        models (list): List of models to compare.
    """
    # Filter data for the selected dataset
    subset = data[data["dataset"] == dataset_name]

    # Normalize metrics to make them comparable (optional but recommended)
    normalized_data = subset.copy()
    for metric in metrics:
        max_value = subset[metric].max()
        min_value = subset[metric].min()
        normalized_data[metric] = (subset[metric] - min_value) / (max_value - min_value)

    # Prepare data for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    plt.figure(figsize=(8, 8))

    for model in models:
        model_data = normalized_data[normalized_data["model"] == model]
        if not model_data.empty:
            # Extract metric values
            values = model_data[metrics].mean().tolist()
            values += values[:1]  # Complete the circle

            # Plot data
            plt.polar(angles, values, label=model)

    # Draw one axis per metric
    plt.xticks(angles[:-1], metrics, fontsize=12)

    # Add legend and title
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.title(f"Radar Chart for {dataset_name}", fontsize=15)

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_all_interactions(data, model_name, params, metric, save_dir=None):
    """
    Create interaction plots for all pairwise combinations of parameters and a given metric.

    Args:
        data (pd.DataFrame): The dataset containing parameters and metrics.
        model_name (str): The name of the model to analyze.
        params (list): A list of parameter column names.
        metric (str): The metric to analyze.
        save_dir (str or None): Directory to save the plots. If None, plots are shown interactively.
    """
    # Filter the data for the selected model
    subset = data[data["model"] == model_name]

    # Ensure the specified columns exist in the dataset
    if not set(params + [metric]).issubset(data.columns):
        raise ValueError(
            "One or more specified parameters or metric are not present in the dataset."
        )

    # Generate all pairwise combinations of parameter
    print(params, type(params))
    # params.remove(['n_clusters'])
    param_combinations = list(itertools.permutations(params, 2))

    # Iterate over each pair of parameters
    for param_x, param_y in param_combinations:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x=param_x, y=metric, hue=param_y, marker="o", palette="viridis")

        # Add labels and title
        plt.title(
            f"Interaction Plot: {model_name} ({param_x} vs {metric} by {param_y})", fontsize=14
        )
        plt.xlabel(param_x, fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(title=param_y, fontsize=10)
        plt.grid(True)

        # Save or display the plot
        if save_dir:
            save_path = f"{save_dir}/{model_name}_{param_x}_vs_{param_y}_interaction.png"
            plt.savefig(save_path, format="png", dpi=300)
            print(f"Saved interaction plot: {save_path}")
        else:
            plt.show()


def custom_boxplot(ax, data):
    ax.boxplot(
        data,
        patch_artist=True,  # Fill boxes with color
        medianprops={"color": "blue", "linewidth": 2},  # Style median lines
        flierprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markeredgewidth": 1,
            "markersize": 4,
        },  # Style outlier points
        boxprops={"facecolor": "lightblue", "alpha": 0.7},  # Style boxes
        whiskerprops={"linestyle": "--"},  # Style whiskers
        capprops={"linewidth": 2},  # Style caps
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)  # Put grid behind plot elements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_interactions_with_gridspec(df, col_names, datasets, model_name, save_path=None):
    num_cols = len(col_names)
    total_datasets = len(datasets)

    # Create the main figure and grid for datasets
    fig = plt.figure(figsize=(10 * total_datasets, 10))
    outer_grid = GridSpec(1, total_datasets, figure=fig, wspace=0.05)  # Reduced from 0.1

    # Prepare a placeholder for the colorbar data
    heatmap_min = max(0.00001, df[df["model"] == model_name]["f_measure"].min())
    heatmap_max = df[df["model"] == model_name]["f_measure"].max()
    norm = LogNorm(heatmap_min, heatmap_max, clip=True)

    for dataset_idx, dataset_name in enumerate(datasets):
        # Create a sub-grid for each dataset within the main grid
        inner_grid = GridSpecFromSubplotSpec(
            num_cols,
            num_cols,
            subplot_spec=outer_grid[dataset_idx],
            wspace=0.1,  # Add tight spacing between columns
            hspace=0.05,  # Add tight spacing between rows
        )

        filtered_df = df[(df["dataset"] == dataset_name) & (df["model"] == model_name)]

        for i, ((col_name1, col_name2), inner_idx) in enumerate(
            zip(itertools.product(col_names, repeat=2), range(num_cols * num_cols))
        ):
            row = inner_idx // num_cols
            col = inner_idx % num_cols

            ax = fig.add_subplot(inner_grid[row, col])

            # Skip plots above the diagonal
            if col > row:
                ax.axis("off")
                continue

            unique_vals_1 = filtered_df[col_name1].unique()
            unique_vals_2 = filtered_df[col_name2].unique()
            if np.issubdtype(unique_vals_1.dtype, np.number):
                unique_vals_1.sort()
            if np.issubdtype(unique_vals_2.dtype, np.number):
                unique_vals_2.sort()

            if col_name1 == col_name2:
                # Diagonal-like plots: Single-variable distribution (boxplot)

                sorted_data = [
                    filtered_df[filtered_df[col_name1] == val]["f_measure"].tolist()
                    for val in unique_vals_1
                ]

                custom_boxplot(ax, sorted_data)
                # ax.set_title(f"{col_name1}", fontsize=10)
            else:
                # Off-diagonal: Interaction heatmap
                pivot_table = filtered_df.pivot_table(
                    values="f_measure", index=col_name1, columns=col_name2, aggfunc="mean"
                )
                sns.heatmap(
                    pivot_table,
                    ax=ax,
                    cmap="viridis",
                    annot=False,
                    cbar=False,
                    vmin=heatmap_min,
                    vmax=heatmap_max,
                    norm=norm,
                )

                # ax.set_yticks(range(len(unique_vals_1)))
                # ax.set_xticks(range(len(unique_vals_2)))
                ax.set_xticklabels(unique_vals_2)
                ax.set_yticklabels(unique_vals_1)
            ax.set_xlabel(col_name2, fontsize=14, fontweight="bold")
            ax.set_ylabel(col_name1, fontsize=14, fontweight="bold")
            if col != 0:
                ax.set_yticks([])
                ax.set_ylabel("")

            if row != len(col_names) - 1:
                ax.set_xticks([])
                ax.set_xlabel("")

            if row == 0 and col == 0:
                ax.set_ylabel("F1 Score", fontsize=14, fontweight="bold")

        # Add a title for the entire dataset grid
        inner_plot = fig.add_subplot(outer_grid[dataset_idx])
        inner_plot.axis("off")
        inner_plot.set_title(dataset_name, fontsize=30, fontweight="bold", y=0.96)

    # Adjust the overall layout with tighter spacing
    fig.subplots_adjust(
        top=0.90,  # More room at top for title
        right=0.92,  # Slightly more room for colorbar
        wspace=0.05,  # Tighter spacing between dataset panels
        hspace=0.05,  # Tighter vertical spacing
        left=0.05,  # Less left margin
        bottom=0.05,  # Less bottom margin
    )

    # Adjust colorbar position for new spacing
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.75])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="vertical",
        format=ticker.FuncFormatter(lambda x, pos: f"{x:.3f}"),
    )
    cbar.set_label("F1 Score", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig
