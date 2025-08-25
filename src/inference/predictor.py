# src/inference/predictor.py
import torch
import pandas as pd

from src.preprocessing.preprocess import Preprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.models.multihead_model import MultiHeadModel
from src.utils.io_utils import get_logger


class Predictor:
    def __init__(self, cfg, device="cpu"):
        self.logger = get_logger('Predictor')
        self.device = device if device else self._find_best_device()
        self.cfg = cfg
        
        self.model = MultiHeadModel(**cfg["model"])
        self.model.load_state_dict(torch.load(cfg['model']['path'], map_location=device))
        self.model.to(device).eval()
        
        self.preproc = Preprocessor()
        self.fe = FeatureEngineer(cfg)

    def predict(self, sample: dict):
        # sample: {"QUESTION": ..., "TITLE": ..., "S_NAME": ..., ...}
        df = pd.DataFrame([sample])
        df = self.preproc.clean_dataframe(df, [self.cfg["text_col"]])
        
        X_cat, X_time, X_text = self.fe.transform(df)
        cats = torch.tensor(X_cat, dtype=torch.float32).to(self.device)
        times = torch.tensor(X_time, dtype=torch.float32).to(self.device)
        texts = torch.tensor(X_text, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(cats, times, texts)
        return {k: v.cpu().numpy() for k, v in outputs.items()}
    
    def _find_best_device(self):
        if torch.cuda.is_available():
            self.logger.info("Using CUDA device")
            return "cuda"
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                self.logger.info("Using DirectML device")
                return device
            except ImportError:
                self.logger.warning("DirectML not available, falling back to CPU")
            self.logger.info("Using CPU device")
            return "cpu"
