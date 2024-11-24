import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


# TODO: add interconnection plot: check work 2 code for plots page 27 in report
