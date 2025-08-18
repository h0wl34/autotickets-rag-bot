from copy import deepcopy
from src.utils.io_utils import load_yaml
from src.configs import paths

BASE_CONFIG = load_yaml(paths.CONFIGS_DIR / "base.yaml")
# PATHS = load_yaml("configs/paths.yaml")
MODELS_CONFIG = load_yaml(paths.CONFIGS_DIR / "models.yaml")
TARGETS_CONFIG = load_yaml(paths.CONFIGS_DIR / "targets.yaml")
FEATURES_CONFIG = load_yaml(paths.CONFIGS_DIR / "features.yaml")

def merge_dicts(base, override):
    for k, v in override.items():
        if isinstance(v, dict):
            base[k] = merge_dicts(base.get(k, {}), v)
        elif v is not None:
            base[k] = v
    return base

def get_features(features: list[str] | None):
    """Return validated feature list from config or override"""
    if features is None:
        features = BASE_CONFIG["features"]
    
    for f in features:
        if f not in FEATURES_CONFIG["features"]:
            raise ValueError(f"Unknown feature: {f}")

    return features


def get_model_params(model_name: str, override: dict = None):
    """Return model params: defaults + overrides"""
    params = {}
    
    if override:
        params = (override)
        return params 
    
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")
    
    params = deepcopy(MODELS_CONFIG[model_name])
    return params


def get_target(target_name: str | None):    
    if target_name is None:
        return BASE_CONFIG["target"]
    if target_name not in  TARGETS_CONFIG["targets"]:
        raise ValueError(f"Unknown target: {target_name}")
    
    return target_name


def resolve_config(exp_path=None, overrides=None):
    """
    Load defaults + experiment file + CLI overrides
    """
    cfg = deepcopy(BASE_CONFIG)
    if exp_path:
        exp_cfg = load_yaml(exp_path)
        cfg.update(exp_cfg)

    if overrides:
        cfg = merge_dicts(cfg, overrides)

    model_name = cfg["model"]["name"]
    model_override = cfg["model"].get("params")
    cfg["model"]["params"] = get_model_params(model_name, model_override)

    cfg["features"] = get_features(cfg.get("features"))
    cfg["target"] = get_target(cfg.get("target")) 

    return cfg
