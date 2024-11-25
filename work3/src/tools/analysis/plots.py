import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import itertools
from config import DATA_DIR


def plot_metrics_comparison(metrics_dict):
    for metric_name, metric_values in metrics_dict.items():
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(metric_values.keys()), y=list(metric_values.values()))
        plt.title(f"Comparison of {metric_name}")
        plt.xlabel("Algorithm")
        plt.ylabel(metric_name)
        plt.show()


def plot_clustering_results(X, true_labels, predicted_labels, metrics):
    # Scatter plot of the data points, colored by predicted clusters
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=predicted_labels, palette="viridis")
    plt.title("Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Bar plot of evaluation metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title("Evaluation Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.show()


def plot_interactions_grid(df, col_names, metrics):
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
    for i, ((col_name1, col_name2), ax) in enumerate(zip(itertools.product(col_names, repeat=2), axes.flatten())):
        if col_name1 == col_name2:
            # Diagonal: Plot distribution of each metric for the given column
            sns.boxplot(data=df, x=col_name1, y=metrics[0], ax=ax)
            ax.set_title(f"Distribution of {metrics[0].upper()} by {col_name1.capitalize()}", fontsize=10)
        else:
            # Off-diagonal: Plot interaction heatmap between two columns for each metric
            pivot_table = df.pivot_table(
                values=metrics[0],  # Use the first metric for simplicity
                index=col_name1,
                columns=col_name2,
                aggfunc='mean'
            )
            sns.heatmap(
                pivot_table, 
                ax=ax, 
                cmap='coolwarm', 
                annot=True, 
                fmt='.3f', 
                cbar=False
            )
            ax.set_title(f"Interaction of {col_name1.capitalize()} and {col_name2.capitalize()} on {metrics[0].upper()}", fontsize=10)
        
        # Remove ticks to make plots cleaner
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Add row labels for columns
    for i in range(num_cols):
        if axes[i, 0] is not None:  # Ensure this is a valid subplot
            axes[i, 0].set_ylabel(col_names[i], fontsize=14, rotation=90)
        if axes[-1, i] is not None:  # Ensure this is a valid subplot
            axes[-1, i].set_xlabel(col_names[i], fontsize=14, rotation=0)
    
    plt.tight_layout()
    return fig


def plot_interactions(csv_path, model_name, paramsGrid):
    """
    Plots interaction plots for various metrics based on a model and parameter grid.

    Parameters:
    - csv_path (str): Path to the metrics CSV file.
    - model_name (str): Name of the model to filter for in the dataset (e.g., 'fuzzy_cmeans').
    - paramsGrid (list of str): List of parameter names to include in the interaction plot (e.g., ['fuzzyness']).

    Returns:
    - None: Displays the interaction plot.


    Example usage:

    csv_path = '3_metrics.csv'
    model_name = 'fuzzy_cmeans'
    paramsGrid = FuzzyCMeansParamsGrid.keys()  # Example parameter grid

    plot_interactions(csv_path, model_name, paramsGrid)

    """
    # Load the data
    data = pd.read_csv(f'{DATA_DIR}/{csv_path}')
    
    # Filter data for the selected model
    data = data[data['model'] == model_name]
    
    # Ensure that the parameters in paramsGrid exist in the dataset
    for param in paramsGrid:
        if param not in data.columns:
            raise ValueError(f"Parameter '{param}' not found in the dataset columns.")
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # Create interaction plots for each parameter in paramsGrid
    metrics = ['ari', 'purity', 'dbi', 'f_measure']  # List of metrics to visualize
    for param in paramsGrid:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 5 * len(metrics)), sharex=True)
        if len(metrics) == 1:  # If only one metric, make axes a list for uniform indexing
            axes = [axes]
        
        # Plot each metric
        for ax, metric in zip(axes, metrics):
            sns.lineplot(
                x=param,
                y=metric,
                data=data,
                marker="o",
                ax=ax
            )
            ax.set_title(f'{metric.upper()} vs {param.capitalize()} for {model_name}')
            ax.set_xlabel(param.capitalize())
            ax.set_ylabel(metric.upper())
        
        plt.tight_layout()
        plt.show()



def plot_clusters(path, features):
    """
    Example of use 

        for path in glob.glob(f'{CLUSTERED_DATA_DIR}/{dataset_name}/{model_name}/*'):
        if 'n_clusters=10' in path:  # change the number of clusters or any other param
            plot_clusters(path=path, features=['0','1']) # for synthetic dataset


    """
    data = pd.read_csv(path)
    config = " ".join(path.split('/')[-1][:-4].split(','))
    labels = data['cluster'].to_list()
    x = str(features[0])
    y = str(features[1])


    plt.figure(figsize=(8, 6))
    plt.scatter(data[x], data[y], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar()
    plt.title(f'Clusters visualized using {config}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()



def plot_model_comparisons(data, metric, title):
    """
    Plot comparisons of models across datasets based on a given metric.
    
    Args:
        data (pd.DataFrame): The dataset containing performance metrics.
        metric (str): The metric to visualize (e.g., 'ari', 'purity').
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data, 
        x='dataset', 
        y=metric, 
        hue='model', 
        ci='sd', 
        palette='viridis'
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.legend(title='Model', fontsize=10, loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

