import argparse
import pandas as pd
import os
import logging
from tools.dimensionality_reduction import REDUCTION_PARAMS_GRID_MAP
from tools.clustering import CLUSTERING_PARAMS_MAP
from tools.analysis.plots import *
from tools.analysis.tables import (
    generate_best_models_table,
    generate_top_models_by_dataset,
    generate_model_best_configs_table,
)
from tools.config import METRICS_DATA_PATH, METRICS_PLOTS_DIR, METRICS_TABLES_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose", "-v", action="store_true", help="Whether to print verbose output"
)

logger = logging.getLogger(__name__)


def compute_analysis(metrics_data: pd.Series, metrics_data_config: dict):
    print(metrics_data_config)
    print(metrics_data)
    pass


def get_metrics_from_row(row: pd.Series) -> pd.Series:
    return row.loc[["ari", "purity", "dbi", "f_measure"]]


def get_config_from_row(row: pd.Series) -> dict:
    reduction_params_keys = REDUCTION_PARAMS_GRID_MAP[row["reduction_method"]].keys()
    clustering_params_keys = CLUSTERING_PARAMS_MAP[row["dataset"]][
        row["clustering_model"]
    ].keys()
    return {
        "dataset_name": row["dataset"],
        "clustering_model": row["clustering_model"],
        "reduction_method": row["reduction_method"],
        "params": {
            key: row[key] for key in reduction_params_keys + clustering_params_keys
        },
    }


metrics = ["ari", "chi", "dbi", "f_measure", "clustering_runtime", "reduction_runtime"]
models = [
    "global_kmeans",
    "optics",
]
datasets = ["mushroom", "vowel"]


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    metrics_data = pd.read_csv(METRICS_DATA_PATH)

    # Create output directories
    os.makedirs(METRICS_PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_TABLES_DIR, exist_ok=True)

    # Generate tables
    logger.info("Generating LaTeX tables...")

    # Overall best models table
    generate_best_models_table(
        metrics_data, f"{METRICS_TABLES_DIR}/best_models_overall.tex"
    )

    # Per-dataset top models tables
    for dataset_name in datasets:
        logger.info(f"Generating table for {dataset_name}...")
        generate_top_models_by_dataset(
            metrics_data,
            dataset_name,
            f"{METRICS_TABLES_DIR}/top_models_{dataset_name}.tex",
        )

    # Per-model best configurations tables
    for model_name in models:
        logger.info(f"Generating best configs table for {model_name}...")
        generate_model_best_configs_table(
            metrics_data,
            model_name,
            f"{METRICS_TABLES_DIR}/best_configs_{model_name}.tex",
        )

    # Generate plots
    logger.info("Generating plots...")

    plot_pairplot(
        data=metrics_data, vars=metrics, save_path=f"{METRICS_PLOTS_DIR}/pairplot.png"
    )

    for metric in metrics:
        plot_model_comparisons(
            data=metrics_data,
            metric=metric,
            title=f"Comparison of {metric.capitalize()} Across Models and Datasets",
            save_path=f"{METRICS_PLOTS_DIR}/model_comparison_{metric}.png",
        )

    plot_combined_heatmaps(
        metrics_data,
        metrics,
        datasets,
        models,
        save_path=f"{METRICS_PLOTS_DIR}/heatmaps.png",
    )

    for dataset_name in datasets:
        plot_radar_chart(
            metrics_data,
            dataset_name,
            metrics,
            models,
            save_path=f"{METRICS_PLOTS_DIR}/radar_chart_{dataset_name}.png",
        )

    for model_name, value in CLUSTERING_PARAMS_MAP.items():
        params = list(value[datasets[0]].keys())
        logger.info(
            f"Plotting interactions (GridSpec) of {model_name} between {params}..."
        )
        plot_interactions_with_gridspec(
            metrics_data,
            params,
            datasets,
            model_name,
            save_path=f"{METRICS_PLOTS_DIR}/interactions_{model_name}.png",
        )


if __name__ == "__main__":
    main()
