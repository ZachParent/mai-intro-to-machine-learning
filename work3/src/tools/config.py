from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "0_raw"
PREPROCESSED_DATA_DIR = DATA_DIR / "1_preprocessed"
CLUSTERED_DATA_DIR = DATA_DIR / "2_clustered"
METRICS_DATA_PATH = DATA_DIR / "3_metrics.csv"
