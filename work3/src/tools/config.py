from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "0_raw"
PREPROCESSED_DATA_DIR = DATA_DIR / "1_preprocessed"
CLUSTERED_DATA_DIR = DATA_DIR / "2_clustered"
METRICS_DATA_PATH = DATA_DIR / "3_metrics.csv"

N_CLUSTERS = [2, 3, 5, 10, 11, 12]
RANDOM_STATE = [1, 2, 3, 4, 5]

REPORT_DIR = PROJECT_ROOT / "report"
PLOTS_DIR = REPORT_DIR / "figures"
TABLES_DIR = os.path.join(REPORT_DIR, "tables")
