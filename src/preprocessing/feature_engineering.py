# src/preprocessing/feature_engineering.py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sentence_transformers import SentenceTransformer

class FeatureEngineer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cat_encoders: dict[str, OrdinalEncoder] = {col: joblib.load(path) for col, path in cfg["cat_encoders"].items()}
        # self.label_encoders: dcit[str, LabelEncoder] = {col: joblib.load(path) for col, path in cfg["label_encoders"].items()}
        self.text_embedder = SentenceTransformer(cfg['embedding_model_path'])

    def encode_cats(self, df: pd.DataFrame, cols: list[str]):
        for col in cols:
            enc = self.cat_encoders[col]
            df[f'{col}_enc'] = enc.transform(df[[col]])    
        return df
    
    # def encode_avariya(series):
    #     return series.map({'Да':1, 'Нет':0}).fillna(0).astype(int)
    
    # def encode_priority(series):
    #     return series.fillna(2).astype(int)  

    def extract_time_features(self, df: pd.DataFrame, time_cols = ['OPEN_TIME_']):
        for col in time_cols:#['OPEN_TIME_', 'RESOLVE_TIME_', 'CLOSE_TIME_', 'ATC_NEXT_BREACH_']:
            df[col] = pd.to_datetime(df[col], errors='coerce')

            # Encoding cyclical features
            hour = df[col].dt.hour
            df[f'{col}_hour_sin'] = np.sin(hour * (2. * np.pi / 24.))
            df[f'{col}_hour_cos'] = np.cos(hour * (2. * np.pi / 24.))

            day_of_week = df[col].dt.dayofweek
            df[f'{col}_day_of_week_sin'] = np.sin(day_of_week * (2. * np.pi / 7.))
            df[f'{col}_day_of_week_cos'] = np.cos(day_of_week * (2. * np.pi / 7.))

            df['is_weekend'] = (day_of_week >= 5).astype(int)

            day_of_month = df[col].dt.day
            df[f'{col}_day_of_month_sin'] = np.sin(day_of_month * (2. * np.pi / 31.))
            df[f'{col}_day_of_month_cos'] = np.cos(day_of_month * (2. * np.pi / 31.))
        
        return df

    def encode_texts(self, df: pd.DataFrame, col):
        texts = df[col].tolist()
        embeddings = self.text_embedder.encode(texts, convert_to_tensor=False, convert_to_numpy=True)  
        df[f'{col}_emb'] = embeddings.tolist()
        return df

    def _extract_sensitive_flags(self, text: str) -> dict:
        """Возвращает булевы фичи для классификаторов"""
        flags = {f"HAS_{k}": 0 for k in self.cfg["sensitive_patterns"].keys()}
        
        if not isinstance(text, str) or text.strip() == "" or text == "[NO_TEXT]":
            flags["HAS_TEXT"] = 0
            return flags
        
        for placeholder in self.cfg["sensitive_patterns"].keys():
            if placeholder in text:
                flags[f"HAS_{placeholder}"] = 1
        
        # Флаг наличия текста
        flags["HAS_TEXT"] = 1
        
        return flags
    
    def resolve_sensitive_flags(self, df: pd.DataFrame, col: str):
        flags_df = df[col].apply(self._extract_sensitive_flags).apply(pd.Series)
        df = pd.concat([df, flags_df], axis=1)
        return df

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        
        df = self.encode_cats(df, self.cfg["categorical"])
        df = self.extract_time_features(df, self.cfg["time"])
        df = self.encode_texts(df, self.cfg["text_col"])
        df = self.resolve_sensitive_flags(df, self.cfg["text_col"])    
        return df
        
