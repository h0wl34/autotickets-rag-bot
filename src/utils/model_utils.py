from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import joblib
from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .io_utils import ensure_dir
from typing import Self


@dataclass
class ModelBundle:
    model: object
    model_type: str
    params: Dict

    def save(self, out_dir: Path):
        ensure_dir(out_dir)
        
        if self.model_type.lower().startswith("catboost"):
            # родной формат CatBoost для совместимости
            self.model.save_model(str(out_dir / "model.cbm"))
        else:
            joblib.dump(self.model, out_dir / "model.pkl")
            
        joblib.dump(self.params, out_dir / "params.joblib")

    @staticmethod
    def load(model_dir: Path) -> Self:
        """
        Load a saved ModelBundle from directory
        
        Args:
            model_dir: Path to directory containing saved model files
            
        Returns:
            ModelBundle instance with loaded model and config
        """
        # Check model type by looking at file extensions
        if (model_dir / "model.cbm").exists():
            model_type = "catboost"
        elif (model_dir / "model.pkl").exists():
            model_type = "xgboost"  # or other non-catboost models
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
        
        # Load parameters
        params = joblib.load(model_dir / "params.joblib")

        # Load model based on type
        if model_type == "catboost":
            model = CatBoostClassifier()
            model.load_model(str(model_dir / "model.cbm"))
        else:
            model = joblib.load(model_dir / "model.pkl")
        
        return ModelBundle(
            model=model,
            model_type=model_type,
            params=params
        )

def build_model(model_type: str, params: Dict):
    mt = model_type.lower()
    if mt in {"catboost", "catboostclassifier", "cat"}:
        return CatBoostClassifier(**params)
    # if mt in {"lgbm", "lightgbm", "lightgbmclassifier"}:
    #     return LGBMClassifier(**params)
    if mt in {"xgboost", "xgb", "xgbclassifier"}:
        return XGBClassifier(**params)
    # if mt in {"lightgbm", "lgbm", "lgbmclassifier"}:
    #     return LGBMClassifier(**params)
    if mt in {"logreg", "logisticregression"}:
        return LogisticRegression(**params)

    raise ValueError(f"Unsupported model_type: {model_type}")

