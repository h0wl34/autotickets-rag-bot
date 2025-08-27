# src/inference/predictor.py
import torch
import pandas as pd

from src.preprocessing.preprocess import Preprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.models.multihead_model import MultiHeadModel
from src.utils.io_utils import get_logger, load_yaml


class Predictor:
    def __init__(self, cfg: dict, device: str=None):
        self.logger = get_logger('Predictor')
        self.device = device if device else self._find_best_device()
        self.cfg = cfg
        
        self.model = MultiHeadModel(load_yaml(cfg["model"]["cfg_path"]))
        checkpoint = torch.load(cfg['model']['path'], map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model_state")
        model_state = model_state if model_state is not None else checkpoint

        self.model.load_state_dict(model_state)
        self.model.to(device).eval()

    def predict(self, feats_dict: dict):
        features = {k: v.to(self.device) for k, v in feats_dict.items()}
        
        with torch.no_grad():
            outputs = self.model(feats_dict)
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
