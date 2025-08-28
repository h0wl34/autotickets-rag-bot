import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import OrdinalEncoder
from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import OrdinalEncoder
from sentence_transformers import SentenceTransformer

class FeatureEngineer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cat_encoders: dict[str, OrdinalEncoder] = {
            col: joblib.load(path) for col, path in cfg["cat_encoders"].items()
        }
        self.text_embedder = SentenceTransformer(cfg['embedding_model_path'])

        # --- compute dimensions ---
        self.categorical_cols = cfg.get('categorical_cols', [])
        self.time_cols = cfg.get('time_cols', [])
        self.emb_text_col = cfg.get('emb_text_col')

        self.cat_dims = {col: 1 for col in self.categorical_cols} # 1 dim per OE categorical feature (just indices)
        self.total_cat_dim = sum(self.cat_dims.values()) 
        self.bin_cat_dim = len(cfg.get("SENSITIVE_PATTERNS", {})) + 1 # +1 for [HAS_TEXT]
        self.time_dim = 7 * len(self.time_cols) # sin/cos for hour, day of week, day of month + is_weekend
        self.text_dim = self.text_embedder.encode("dummy", convert_to_tensor=False).shape[0]

    def _encode_cats(self, df: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
        cat_arrays = []
        for col in cat_cols:
            enc = self.cat_encoders[col]
            known_count = len(enc.categories_[0])
            arr = enc.transform(df[[col]].fillna("__MISSING__"))  # unseen classes mapped to -1
            arr = np.where(arr == -1, known_count, arr)  # map unseen values to last dim
            cat_arrays.append(arr.astype(np.float32))
        return np.concatenate(cat_arrays, axis=1)

    def _extract_time_features(self, df: pd.DataFrame, time_cols: list[str]) -> np.ndarray:
        time_feats = []
        for col in time_cols:
            if col not in df.columns:
                time_feats.append(np.zeros((len(df), 7), dtype=np.float32))
                continue

            datetime_col = pd.to_datetime(df[col], errors='coerce')
            hour = datetime_col.dt.hour
            day_of_week = datetime_col.dt.dayofweek
            day_of_month = datetime_col.dt.day
            feats = np.stack([
                np.sin(hour * 2*np.pi/24),
                np.cos(hour * 2*np.pi/24),
                np.sin(day_of_week * 2*np.pi/7),
                np.cos(day_of_week * 2*np.pi/7),
                (day_of_week >= 5).astype(int),
                np.sin(day_of_month * 2*np.pi/31),
                np.cos(day_of_month * 2*np.pi/31)
            ], axis=1)
            time_feats.append(feats.astype(np.float32))
        return np.concatenate(time_feats, axis=1) if time_feats else np.zeros((len(df), 0), dtype=np.float32)

    def _encode_texts(self, df: pd.DataFrame, text_col: str) -> np.ndarray:
        texts = df[text_col].tolist()
        embeddings = self.text_embedder.encode(texts, convert_to_tensor=False, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def _resolve_sensitive_flags(self, df: pd.DataFrame, text_col: str) -> np.ndarray:
        def _flags(text):
            flags = {f"HAS_{k}": 0 for k in self.cfg.get("SENSITIVE_PATTERNS", {}).keys()}
            if not isinstance(text, str) or text in ["", "[NO_TEXT]"]:
                flags["HAS_TEXT"] = 0
                return list(flags.values())
            for placeholder in self.cfg.get("SENSITIVE_PATTERNS", {}):
                if f"[{placeholder}]" in text:
                    flags[f"HAS_{placeholder}"] = 1
            flags["HAS_TEXT"] = 1
            return list(flags.values())

        return np.stack(df[text_col].apply(_flags).values).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> dict:
        n = len(df)

        # --- OE categorical features ---
        if all(col in df for col in self.categorical_cols):
            oe_cats = self._encode_cats(df, self.categorical_cols)
        else:
            unknown_indices = [len(self.cat_encoders[col].categories_[0]) for col in self.categorical_cols]
            oe_cats = np.tile(unknown_indices, (n, 1)).astype(np.float32)

        # --- binary sensitive flags ---
        if self.emb_text_col in df:
            bin_cats = self._resolve_sensitive_flags(df, self.emb_text_col)
        else:
            bin_cats = np.zeros((n, self.bin_cat_dim), dtype=np.float32)

        # --- time features ---
        if all(col in df for col in self.time_cols):
            times = self._extract_time_features(df, self.time_cols)
        else:
            times = np.zeros((n, self.time_dim), dtype=np.float32)

        # --- text embeddings ---
        if self.emb_text_col in df:
            text_emb = self._encode_texts(df, self.emb_text_col)
        else:
            text_emb = np.zeros((n, self.text_dim), dtype=np.float32)

        return {
            "oe_cats": torch.from_numpy(oe_cats).long(),
            "bin_cats": torch.from_numpy(bin_cats).float(),
            "times": torch.from_numpy(times).float(),
            "text": torch.from_numpy(text_emb).float()
        }

    def get_valid_classes(self) -> dict[str, list]:
        return {col: enc.categories_[0].tolist() for col, enc in self.cat_encoders.items()}
