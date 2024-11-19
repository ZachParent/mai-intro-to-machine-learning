from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "0_raw"
PROCESSED_DATA_DIR = DATA_DIR / "1_processed"
CLUSTERED_DATA_DIR = DATA_DIR / "2_clustered"
METRICS_DIR = PROJECT_ROOT / "3_metrics"
