from src.inference.predictor import Predictor
from src.preprocessing.preprocess import Preprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.inference.postprocessor import Postprocessor

class InferencePipeline:
    def __init__(self, cfg):
        self.predictor = Predictor(cfg, device=cfg.get('inference').get('device', 'cpu'))
        self.postproc = Postprocessor(cfg['postprocessing'])

    def run(self, sample):
        raw_preds = self.predictor.predict(sample)
        final_preds = self.postproc.process(raw_preds)
        return final_preds
