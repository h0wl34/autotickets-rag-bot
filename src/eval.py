import argparse
import numpy as np
from pathlib import Path
from src.utils.model_utils import ModelBundle
from src.utils.data_loader import load_features, load_target, subset_by_idx, load_split
from src.utils.io_utils import get_logger, save_json
from src.utils.config_helper import resolve_config
from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True, help="Path to the experiment directory (model + config)")
    parser.add_argument("--target", type=str, required=False, help="Target to evaluate")
    parser.add_argument("--split", type=str, default="val", help="Which split to use (val/test)")
    parser.add_argument("--threshold", type=float, required=False, help="threshold for bin classification")
    return parser.parse_args()

def main():
    args = parse_args()

    run_path = Path(args.run_path)
    cfg = resolve_config(run_path / "config.yaml")
    logger = get_logger("eval", out_dir=run_path)

    # Load model bundle
    bundle = ModelBundle.load(run_path)
    model = bundle.model

    # Load data
    X = load_features(cfg["features"], output_format="dense")
    y = load_target(cfg["target"])

    threshold = args.threshold
    if (threshold is not None and len(np.unique(y)) > 2):
        logger.warning("Target variable has more than 2 unique values, ignoring --threshold")
        threshold = None

    eval_idx = load_split(args.split)
    X_eval = subset_by_idx(X, eval_idx)
    y_eval = y[eval_idx]

    logger.info(f"Started evaluation on {args.split} split, using {X_eval.shape} samples")

    # Predict and compute metrics
    if threshold is not None:
        probs = model.predict_proba(X_eval)
        y_pred = (probs[:, 1] >= threshold).astype(int)  
    else:
        y_pred = model.predict(X_eval)

    metrics = classification_report(y_eval, y_pred, zero_division=0)
    logger.info(f"Evaluation metrics on {args.split}:")
    logger.info(f"\n{metrics}")

    save_json(metrics, run_path / "metrics.json")

if __name__ == "__main__":
    main()
