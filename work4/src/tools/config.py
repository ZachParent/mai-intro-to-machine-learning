from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "0_raw"
PREPROCESSED_DATA_DIR = DATA_DIR / "1_preprocessed"
REDUCED_DATA_DIR = DATA_DIR / "2_reduced"
CLUSTERED_DATA_DIR = DATA_DIR / "3_clustered"
METRICS_DATA_PATH = DATA_DIR / "4_metrics.csv"
METRICS_PLOTS_DIR = DATA_DIR / "5_metrics_plots"
METRICS_TABLES_DIR = DATA_DIR / "5_metrics_tables"
VISUALIZATIONS_DIR = DATA_DIR / "6_visualizations"
NON_REDUCED_DATA_NAME = "non_reduced"

N_CLUSTERS = [2, 3, 5, 10, 11, 12]
RANDOM_STATE = [1, 2, 3, 4, 5]

REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR = REPORT_DIR / "tables"
