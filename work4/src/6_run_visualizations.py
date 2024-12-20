import argparse
import pandas as pd
import os
import logging
from tools.dimensionality_reduction import PCA
from tools.analysis.plots import *
from tools.config import VISUALIZATIONS_DIR, CLUSTERED_DATA_DIR
from pathlib import Path
from umap import UMAP

parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose", "-v", action="store_true", help="Whether to print verbose output"
)

logger = logging.getLogger(__name__)

VISUALIZATION_METHODS = {"pca": PCA, "umap": UMAP}


def get_metrics_from_row(row: pd.Series) -> pd.Series:
    return row.loc[["ari", "purity", "dbi", "f_measure"]]


def get_config_from_filepath(filepath: Path) -> dict:
    dataset_name = filepath.parent.parent.parent.name
    reduction_method = filepath.parent.parent.name
    clustering_model = filepath.parent.name
    params_str = filepath.stem.split(",")
    params = {param.split("=")[0]: param.split("=")[1] for param in params_str}
    return {
        "dataset": dataset_name,
        "reduction_method": reduction_method,
        "clustering_model": clustering_model,
        "params": params,
    }


def plot_visualization(ax, two_d_data: np.ndarray, labels: np.ndarray):
    ax.scatter(two_d_data[:, 0], two_d_data[:, 1], c=labels, cmap="viridis")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    clustered_paths = sorted(list(CLUSTERED_DATA_DIR.glob("**/*.csv")))
    clustered_paths = [path for path in clustered_paths if path.name != "runtime.csv"]
    logger.info(f"Found {len(clustered_paths)} clustered data files")

    clustered_data_dfs = {}
    for path in clustered_paths:
        clustered_data_dfs[path] = pd.read_csv(path)

    # Create output directories
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    logger.info("Generating visualizations...")
    for path, df in clustered_data_dfs.items():
        config = get_config_from_filepath(path)
        for (
            visualization_method_name,
            visualization_method,
        ) in VISUALIZATION_METHODS.items():
            logger.info(
                f"Generating {visualization_method_name} visualization for {path}"
            )
            viz = visualization_method(
                n_components=2
            )  # Explicitly set n_components for PCA
            two_d_data = viz.fit_transform(df.iloc[:, :-1])

            fig, ax = plt.subplots()
            plot_visualization(ax, two_d_data, df["cluster"].values)
            ax.set_title(
                f"{config['reduction_method']} {config['clustering_model']} {visualization_method_name}"
            )

            output_dir = (
                VISUALIZATIONS_DIR
                / config["dataset"]
                / config["reduction_method"]
                / config["clustering_model"]
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = (
                output_dir
                / f"{visualization_method_name}_{','.join([f'{k}={v}' for k, v in config['params'].items()])}.png"
            )
            fig.savefig(output_path)
            plt.close(fig)
            logger.info(f"Saved {output_path}")


if __name__ == "__main__":
    main()
