from src.utils.io_utils import ensure_dir, get_logger, save_json, save_yaml
from src.training.trainers import train_on_split, train_kfold
from src.utils.data_loader import load_features, load_split, load_stratify_vector, load_target, subset_by_idx
import numpy as np
from datetime import datetime
from pathlib import Path
from src.configs.paths import RUNS_DIR

from src.utils.model_utils import ModelBundle

class TableTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["name"]
        self.features = config["features"]
        self.target = config["target"]
        self.params = config["model"]["params"]
        self.max_samples = config.get("max_samples", None)
        self.random_state = config["seed"]
        self.class_weight: str | list | None = config.get("class_weight", None)
        
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{ts}_{self.model_name}_{self.target}"
        self.run_dir = Path(RUNS_DIR) / run_name
        ensure_dir(self.run_dir)

    def save_single_run(self, metrics: dict, model):
        """Save model bundle, metrics, and config for a single run."""                
        bundle = ModelBundle(model=model, model_type=self.model_name, params=self.params)
        bundle.save(self.run_dir)

        save_json({"metrics": metrics}, self.run_dir / "metrics.json")
        save_yaml(self.config, self.run_dir / "config.yaml")

        return self.run_dir

    def run_train(self, use_full_train: bool=False):
        logger = get_logger("train", out_dir=self.run_dir)
        logger.info(f"Starting training for {self.model_name}")
        logger.info(f"Target: {self.target}")
        logger.info(f"Features: {self.features}")
        logger.info(f"Params: {self.params}")

        train_idx = load_split("train")
        val_idx = load_split("val")
        
        # Combine train + val
        if use_full_train:
            train_idx = np.concatenate([train_idx, val_idx])
            val_idx = None  # no validation
            logger.info(f"Using full (train + val) set: {len(train_idx)} samples")
            
        X = load_features(self.features, output_format="dense")
        y = load_target(self.target)
            
        if self.max_samples and len(train_idx) > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            subset_idx = rng.choice(train_idx, size=self.max_samples, replace=False)
            
            if not use_full_train:
                ratio = len(val_idx)/len(train_idx)
                val_idx = rng.choice(val_idx, size=int(self.max_samples * ratio), replace=False)
            
            train_idx = subset_idx
            logger.info(f"Subsampled to {self.max_samples} samples")

        X_train, y_train = subset_by_idx(X, train_idx), y[train_idx]
        
        if val_idx is not None:
            X_val = subset_by_idx(X, val_idx)
            y_val = y[val_idx]
        else:
            X_val = y_val = None

        if (use_full_train):
            logger.info(f"Training on {X_train.shape}, skipped validation")
        else:
            logger.info(f"Training on {X_train.shape}, validating on {X_val.shape}")

        model, metrics = train_on_split(
            model_name=self.model_name,
            params=self.params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            class_weight=self.class_weight
        )

        run_path = self.save_single_run(metrics, model)
        logger.info(f"Hold-out training completed. Run saved to {run_path}")
    
    
    def run_kfold(self):
        '''fancy train run with k-fold validation'''
        logger = get_logger("kfold-train", out_dir=self.run_dir)
        logger.info(f"Starting K-Fold training for {self.model_name}")
        logger.info(f"Target: {self.target}")
        logger.info(f"Features: {self.features}")
        logger.info(f"Params: {self.params}")

        n_splits=self.config.get("validation", {}).get("n_splits", 5)
        stratify_by=self.config.get("validation", {}).get("stratify_by")
        
        train_idx = load_split("train")
        val_idx = load_split("val")
        train_val_idx = np.concatenate([train_idx, val_idx])
        
        # Cut train + val size
        if self.max_samples and len(train_idx) > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            subset_idx = rng.choice(train_val_idx, size=self.max_samples, replace=False)
            train_val_idx = subset_idx
            logger.info(f"Subsampled to {self.max_samples} samples")
        
        X = load_features(self.features, output_format="dense")[train_val_idx]
        y = load_target(self.target)[train_val_idx]
        
        stratify = load_stratify_vector(stratify_by)[train_val_idx]

        # Train K folds
        fold_models, fold_metrics = train_kfold(
            model_name=self.model_name,
            params=self.params,
            X=X,
            y=y,
            n_splits=n_splits,
            stratify=stratify,
            random_state=self.random_state,
            class_weight=self.class_weight
        )

        # Save each fold model and metrics
        for i, (model, metrics) in enumerate(zip(fold_models, fold_metrics)):
            fold_path = self.run_dir / f"fold_{i}"
            ensure_dir(fold_path)
            bundle = ModelBundle(model=model, model_type=self.model_name, params=self.params)
            bundle.save(fold_path)
            save_json({"metrics": metrics}, fold_path / "metrics.json")

        # Save aggregated metrics
        aggregated_metrics = {
            "mean": {
                k: float(np.mean([m[k]["f1-score"] for m in fold_metrics if k in m and "f1-score" in m[k]]))
                for k in fold_metrics[0]
                if k not in ["accuracy", "macro avg", "weighted avg"]  # optional: handle separately
            },
            "std": {
                k: float(np.std([m[k]["f1-score"] for m in fold_metrics if k in m and "f1-score" in m[k]]))
                for k in fold_metrics[0]
                if k not in ["accuracy", "macro avg", "weighted avg"]
            },
            "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics])),
        }
        save_json(aggregated_metrics, self.run_dir / "aggregated_metrics.json")
        save_yaml(self.config, self.run_dir / "config.yaml")

        logger.info(f"K-Fold training completed. Run saved to {self.run_dir}")
