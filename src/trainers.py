from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
# from .utils.metrics_utils import compute_metrics
from sklearn.metrics import classification_report
from .utils.model_utils import build_model


def train_on_split(model_name, params, X_train, y_train, X_val=None, y_val=None, class_weight: str | list | None =None):
    """
    Train a model on a train/val split, with optional class weighting.
    """
    model = build_model(model_name, params)
    
    # Compute sample weights if needed
    sample_weight = None
    if class_weight is not None:
        sample_weight = compute_sample_weight(class_weight=class_weight, y=y_train)
    
    # Fit model
    if model_name in ["catboost", "lightgbm", "xgboost"]:
        eval_set = (X_val, y_val) if (X_val is not None and y_val is not None) else None
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=params.get("verbose", 200)
        )
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)

    metrics = {}
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        metrics = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    return model, metrics

def train_kfold(model_name, params, X, y, n_splits=5, stratify=None, random_state=42, class_weight: str | list | None =None):
    '''Used for cross-validation training'''

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    fold_models = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, stratify if stratify is not None else y)):
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model, report = train_on_split(model_name, params, X_train, y_train, X_val, y_val, class_weight)
        fold_models.append(model)
        fold_metrics.append(report)

    return fold_models, fold_metrics
