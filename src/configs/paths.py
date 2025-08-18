from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FEATS_DIR = DATA_DIR / "features"
TARGETS_DIR = DATA_DIR / "targets"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
CONFIGS_DIR = ROOT / "configs"

PATHS = {
    "root_dir": ROOT,
    "data_dir": ROOT / "data",
    "splits_dir": ROOT / "data" / "splits",
    # "logs_dir": ROOT / "logs",
    "models_dir": ROOT / "models",
    "runs_dir": ROOT / "runs",
    "feats_dir": ROOT / "data" / "features",
    "targets_dir": ROOT / "data" / "targets",
    "configs_dir": ROOT / "configs"
}
