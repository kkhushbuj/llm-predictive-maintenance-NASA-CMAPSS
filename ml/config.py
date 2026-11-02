from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "CMAPSS"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = MODELS_DIR / "results"

for d in [PROCESSED_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SUBSET = "FD001"

COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors with ~zero variance in FD001 (single operating condition) — carry no
# degradation signal and are dropped before training.
CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

FEATURE_COLUMNS = [c for c in COLUMNS[2:] if c not in CONSTANT_SENSORS]

SEQUENCE_LENGTH = 30
RUL_CAP = 125  # piecewise-linear RUL cap (Heimes 2008), standard for C-MAPSS LSTM benchmarks

RANDOM_SEED = 42
