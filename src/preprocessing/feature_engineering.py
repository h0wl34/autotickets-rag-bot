# src/preprocessing/feature_engineering.py
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import OrdinalEncoder
from sentence_transformers import SentenceTransformer

class FeatureEngineer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cat_encoders: dict[str, OrdinalEncoder] = {col: joblib.load(path) for col, path in cfg["cat_encoders"].items()}
        self.text_embedder = SentenceTransformer(cfg['embedding_model_path'])

    def encode_cats(self, df: pd.DataFrame, cat_cols: list[str]):
        cat_arrays = []
        for col in cat_cols:
            if col not in df.columns:
                n = len(df)
                cat_arrays.append(np.full((n,1), -1, dtype=np.float32))
            else:    
                enc = self.cat_encoders[col]
                known_count = len(enc.categories_[0])
                arr = enc.transform(df[[col]].fillna("__MISSING__")) # unseen classes mapped to -1
                arr = np.where(arr == -1, known_count, arr) # map unseen values to the last dim for cat embeddings
                cat_arrays.append(arr.astype(np.float32))
        
        return np.concatenate(cat_arrays, axis=1)
    
    # def encode_avariya(series):
    #     return series.map({'Да':1, 'Нет':0}).fillna(0).astype(int)
    
    # def encode_priority(series):
    #     return series.fillna(2).astype(int)  

    def extract_time_features(self, df: pd.DataFrame, time_cols: list[str]):
        time_feats = []
        for col in time_cols:
            datetime_col = pd.to_datetime(df[col], errors='coerce')
            hour = datetime_col.dt.hour
            day_of_week = datetime_col.dt.dayofweek
            day_of_month = datetime_col.dt.day
            feats = np.stack([
                np.sin(hour * 2*np.pi/24), # hours sin
                np.cos(hour * 2*np.pi/24), # hours cos
                np.sin(day_of_week * 2*np.pi/7), # days sin
                np.cos(day_of_week * 2*np.pi/7), # days cos
                (day_of_week >= 5).astype(int), # is_weekend
                np.sin(day_of_month * 2*np.pi/31), # day of month sin
                np.cos(day_of_month * 2*np.pi/31) # day of month cos
            ], axis=1)
            time_feats.append(feats.astype(np.float32))
            
        return np.concatenate(time_feats, axis=1)

    def encode_texts(self, df: pd.DataFrame, text_col: str):
        texts = df[text_col].tolist()
        embeddings = self.text_embedder.encode(texts, convert_to_tensor=False, convert_to_numpy=True)  
        return embeddings.astype(np.float32)

    def _extract_sensitive_flags(self, text: str) -> dict:
        flags = {f"HAS_{k}": 0 for k in self.cfg["sensitive_patterns"].keys()}
        
        if not isinstance(text, str) or text.strip() == "" or text == "[NO_TEXT]":
            flags["HAS_TEXT"] = 0
            return flags
        
        for placeholder in self.cfg["sensitive_patterns"].keys():
            if placeholder in text:
                flags[f"HAS_{placeholder}"] = 1
        
        flags["HAS_TEXT"] = 1
        return flags
    
    def resolve_sensitive_flags(self, df: pd.DataFrame, text_col: str):
        return df[text_col].apply(self._extract_sensitive_flags).apply(pd.Series).to_numpy(dtype=np.float32)

    def transform(self, df: pd.DataFrame):
        oe_cats = self.encode_cats(df, self.cfg['categorical'])
        bin_cats = self.resolve_sensitive_flags(df, self.cfg['text_col'])    
        times = self.extract_time_features(df, self.cfg['time'])
        text_emb = self.encode_texts(df, self.cfg['text_col'])
        
        return {
            "oe_cats": torch.from_numpy(oe_cats).long(),
            "bin_cats": torch.from_numpy(bin_cats).long(),
            "times": torch.from_numpy(times).float(),
            "text": torch.from_numpy(text_emb).float()
        }
