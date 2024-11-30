import argparse
import pandas as pd
import os
import logging
from tools.clustering import PARAMS_GRID_MAP
from tools.analysis.plots import *
from tools.config import METRICS_DATA_PATH, PLOTS_DIR
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print verbose output")

logger = logging.getLogger(__name__)


def compute_analysis(metrics_data: pd.Series, metrics_data_config: dict):
    print(metrics_data_config)
    print(metrics_data)
    pass


def get_metrics_from_row(row: pd.Series) -> pd.Series:
    return row.loc[["ari", "purity", "dbi", "f_measure"]]


def get_config_from_row(row: pd.Series) -> dict:
    params_keys = PARAMS_GRID_MAP[row["model"]].keys()
    return {
        "dataset_name": row["dataset"],
        "model_name": row["model"],
        "params": {key: row[key] for key in params_keys},
    }

metrics = ['ari', 'chi', 'dbi', 'f_measure', 'runtime']
models = ['fuzzy_cmeans', 'kmeans', 'gmeans', 'global_kmeans', 'optics', 'spectral_clustering'] # add more models if needed
datasets = ['mushroom','hepatitis','vowel']


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    metrics_data = pd.read_csv(METRICS_DATA_PATH)

    # for _, row in metrics_data.iterrows():
    #     metrics_data_config = get_config_from_row(row)
    #     metrics_data = get_metrics_from_row(row)
    #     logger.info(f"Running analysis for config {metrics_data_config}")

    #     compute_analysis(metrics_data, metrics_data_config)

    # save plots for further analysis
    plot_pairplot(
        data=metrics_data,
        vars=metrics,
        save_path=f'{PLOTS_DIR}/pairplot.png'
    )

    for metric in metrics:
        plot_model_comparisons(
            data=metrics_data,
            metric=metric,
            title=f'Comparison of {metric.capitalize()} Across Models and Datasets',
            save_path=f'{PLOTS_DIR}/model_comparison_{metric}.png'
        )


    plot_combined_heatmaps(
        metrics_data, metrics,
        datasets, models,
        save_path=f'{PLOTS_DIR}/heatmaps.png'
    )

    for dataset_name in datasets:
        plot_radar_chart(
            metrics_data, dataset_name,
            metrics, models,
            save_path=f'{PLOTS_DIR}/radar_chart_{dataset_name}.png'
        )


    for model_name, value in PARAMS_GRID_MAP.items():
        params = list(value.keys())
        logger.info(f'Plotting interactions of {model_name} between {params}...')
        plot_all_interactions(metrics_data, model_name, params, 'f_measure', save_dir=PLOTS_DIR)


    for model_name, value in PARAMS_GRID_MAP.items():
        params = list(value.keys())
        logger.info(f'Plotting interactions (GridSpec) of {model_name} between {params}...')
        plot_interactions_with_gridspec(
            metrics_data, params, datasets, model_name,
            save_path=f'{PLOTS_DIR}/interactions_gripspec_{model_name}.png'
        )


    # for path in glob.glob(f'{CLUSTERED_DATA_DIR}/{dataset_name}/{model_name}/*'):
    #     if 'n_clusters=10' in path:  # change the number of clusters or any other param
    #         plot_clusters(path=path, features=['0','1']) # for synthetic dataset



if __name__ == "__main__":
    main()