from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SQL_DIR = BASE_DIR / "src" / "sql"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_DIR
NORMALIZED_DATA_PATH = DATA_DIR / "normalized"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
FINAL_DATA_PATH = DATA_DIR / "final_data.csv"
MODEL_SAVE_PATH = MODELS_DIR / "best_model.joblib"

PROCESS_DATA_SCRIPT = SQL_DIR / "process_data.sql"

TARGET_COLUMN = "delta"
TOP_N_FEATURES = 10

RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2
RF_PARAMS = {
    'random_state': RANDOM_STATE,
    'n_estimators': 100
}
