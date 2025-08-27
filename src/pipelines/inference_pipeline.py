import pandas as pd

from src.inference.predictor import Predictor
from src.preprocessing.preprocess import Preprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.inference.postprocessor import Postprocessor

class InferencePipeline:
    def __init__(self, cfg):
        self.preproc = Preprocessor(cfg['preprocessing'])
        self.fe = FeatureEngineer(cfg['preprocessing'])
        self.predictor = Predictor(cfg, device=cfg.get('inference').get('device', 'cpu'))
        self.postproc = Postprocessor(cfg['postprocessing'])

    def run(self, sample: pd.DataFrame):
        cleaned_sample = self.preproc.transform_dataframe(sample)
        feats_dict = self.fe.transform(cleaned_sample)
        raw_preds = self.predictor.predict(feats_dict)
        final_preds = self.postproc.process(raw_preds)
        
        return final_preds
