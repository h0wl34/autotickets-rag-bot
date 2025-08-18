import argparse
from pathlib import Path
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
import time
from datetime import datetime

from src.utils.config_helper import resolve_config
from src.utils.io_utils import ensure_dir, get_logger, save_yaml, save_json
from src.utils.data_loader import load_features, load_target, load_split, load_stratify_vector
from src.utils.model_utils import build_model
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--study-name", type=str)
    p.add_argument("--direction", choices=["maximize","minimize"], default="maximize")
    p.add_argument("--metric", default="weighted_f1", help="weighted_f1|macro_f1|accuracy|binary_f1")
    p.add_argument("--n_splits", type=int, default=None)  # если None — возьмём из config
    p.add_argument("--use_holdout", action="store_true", help="вместо KFold использовать train/val holdout")
    p.add_argument("--save_every", type=int, default=10, help="сохранять каждый n-ый trial")
    return p.parse_args()

def get_score_from_report(report_dict, metric):
    if metric == "accuracy":
        return float(report_dict["accuracy"])
    elif metric == "weighted_f1":
        return float(report_dict["weighted avg"]["f1-score"])
    elif metric == "macro_f1":
        return float(report_dict["macro avg"]["f1-score"])
    elif metric == "binary_f1":
        return float(report_dict["1"]["f1-score"])
    else:
        raise ValueError(f"Unknown metric: {metric}")

def crossval_score(model_name, base_params, X, y, n_splits, stratify, class_weight=None, random_state=42, metric="weighted_f1"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, stratify if stratify is not None else y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        params = dict(base_params)  # копия на всякий
        model = build_model(model_name, params)

        sample_weight = None
        if class_weight is not None:
            sample_weight = compute_sample_weight(class_weight=class_weight, y=y_tr)

        # бустинги — с валидацией для early stopping, остальные — просто fit/predict
        if model_name in ["catboost", "lightgbm", "xgboost"]:
            model.fit(
                X_tr, y_tr,
                sample_weight=sample_weight,
                eval_set=(X_va, y_va),
                verbose=params.get("verbose", 0)
            )
        else:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)

        y_pred = model.predict(X_va)
        report = classification_report(y_va, y_pred, output_dict=True)
        scores.append(get_score_from_report(report, metric))

    return float(np.mean(scores))

def holdout_score(model_name, base_params, X_train, y_train, X_val, y_val, class_weight=None, metric="weighted_f1"):
    model = build_model(model_name, dict(base_params))
    sample_weight = None
    if class_weight is not None:
        sample_weight = compute_sample_weight(class_weight=class_weight, y=y_train)

    if model_name in ["catboost", "lightgbm", "xgboost"]:
        model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=(X_val, y_val), verbose=base_params.get("verbose", 100))
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    return get_score_from_report(report, metric)

def sample_params(trial, cfg):
    # 1) если в конфиге есть search_space — берём оттуда
    ss = cfg.get("search_space", None)
    if ss:
        params = {}
        for name, spec in ss.items():
            t = spec["type"]
            if t == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif t == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif t == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown param type: {t}")
        return params

    # 2) дефолтные search spaces по имени модели (fallback)
    m = cfg["model"]["name"]
    if m == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 1e-1, log=True),
        }
    elif m == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": trial.suggest_int("max_iter", 200, 2000),
        }
    elif m == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 1e-1, log=True),
        }
    elif m == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-6, 1e-1, log=True),
        }
    else:
        # ничего не подбираем
        return cfg["model"].get("params", {})

def main():
    args = parse_args()
    cfg = resolve_config(args.config)

    study_name = args.study_name if args.study_name else f'{cfg["model"]["name"]}_{cfg["target"]}'
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("runs") / f"{ts}_hps_{study_name}"
    ensure_dir(out_dir)
    logger = get_logger("hp_search", out_dir)

    # загрузка фич и таргета один раз
    X = load_features(cfg["features"], output_format="dense")
    y = load_target(cfg["target"])

    # выбор сплитов
    train_idx = load_split("train")
    val_idx   = load_split("val")
    
    logger.info("Started hyperparameter search")
    logger.info(f"Using {'holdout validation' if args.use_holdout else 'cross-validation'}, ({len(train_idx)} train + {len(val_idx)} val) samples ")

    max_samples = cfg.get("max_samples")
    if max_samples is not None and len(train_idx) > max_samples:
        rng = np.random.RandomState(cfg.get("seed", 42))
        subset_idx = rng.choice(train_idx, size=max_samples, replace=False)

        ratio = len(val_idx)/len(train_idx)
        val_idx = rng.choice(val_idx, size=int(max_samples * ratio), replace=False)

        train_idx = subset_idx
        logger.info(f"Subsampled to {max_samples} samples")

    # подготовка стратификации
    stratify_by = cfg.get("validation", {}).get("stratify_by")
    stratify_vec = load_stratify_vector(stratify_by) if stratify_by else y

    # параметры CV
    n_splits = args.n_splits or cfg.get("validation", {}).get("n_splits", 5)

    # веса классов (если заданы в конфиге)
    class_weight = cfg.get("class_weight", None)
    
    trials_results = []

    def objective(trial):
        params = sample_params(trial, cfg)
        model_name = cfg["model"]["name"]
        start_time = time.time()

        if args.use_holdout:
            score = holdout_score(
                model_name, params,
                X[train_idx], y[train_idx],
                X[val_idx], y[val_idx],
                class_weight=class_weight,
                metric=args.metric
            )
        else:
            idx = np.concatenate([train_idx, val_idx])  # CV на train+val
            score = crossval_score(
                model_name, params,
                X[idx], y[idx],
                n_splits=n_splits,
                stratify=(stratify_vec[idx] if stratify_vec is not None else None),
                class_weight=class_weight,
                random_state=cfg.get("seed", 42),
                metric=args.metric
            )
        trial_time = time.time() - start_time
        
        trial.set_user_attr("params_used", params)
        
        logger.info(f"Trial {trial.number}: score={score:.6f}, duration={trial_time:.1f}s, params={params}")
        
        # сохраняем промежуточные результаты каждые N трейлов
        trials_results.append({"number": trial.number, "value": score, "params": params})
        if (trial.number + 1) % args.save_every == 0:
            with open(out_dir / "trials_intermediate.json", "w") as f:
                save_json(trials_results, f)
        
        return score

    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler(n_startup_trials=5)
    study = optuna.create_study(direction=args.direction, study_name=study_name, pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=args.trials)

    logger.info(f"Best value: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")

    # сохраняем результаты (один аккуратный артефакт на весь поиск)
    ensure_dir(out_dir)

    # итоговый конфиг с лучшими гиперами
    best_cfg = dict(cfg)
    best_cfg["model"] = dict(cfg["model"])
    best_cfg["model"]["params"] = dict(cfg["model"].get("params", {}))
    best_cfg["model"]["params"].update(study.best_params)
    save_yaml(best_cfg, out_dir / "best_params_config.yaml")
    save_json({
        "best_value": study.best_value,
        "best_params": study.best_params,
        "direction": args.direction,
        "metric": args.metric
    }, out_dir / "best_metrics.json")

    # история трейалов
    trials = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials if t.state.name == "COMPLETE"
    ]
    save_json(trials, out_dir / "trials.json")

if __name__ == "__main__":
    main()
