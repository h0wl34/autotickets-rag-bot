import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix, issparse
from src.configs.paths import PATHS
from src.utils.config_helper import TARGETS_CONFIG


def load_features(feature_files: list[str] | str, output_format: str = "sparse"):
    """Универсальная загрузка с выбором формата."""
    if isinstance(feature_files, str):
        feature_files = [feature_files]

    sparse, dense = load_features_separately(feature_files)
    
    if output_format == "sparse":
        return combine_to_sparse(sparse, dense)
    elif output_format == "dense":
        return combine_to_dense(sparse, dense)
    else:
        raise ValueError("Формат должен быть 'sparse' или 'dense'")

    
def load_target(target_name: str) -> np.ndarray:
    target_col = TARGETS_CONFIG["targets"].get(target_name)
    if target_col is None:
        raise ValueError(f"Unknown target: {target_name}")
    
    df = pd.read_csv(PATHS["targets_dir"] / "targets.csv")
    return df[target_col].values


def load_split(split_name: str) -> np.ndarray:
    '''Loads a split as a np array of indices'''
    return np.load(PATHS["splits_dir"] / f"{split_name}.npy")


def load_stratify_vector(target_name: str) -> np.ndarray:
    target_col = TARGETS_CONFIG["targets"].get(target_name)
    if target_col is None:
        raise ValueError(f"Unknown target: {target_name}")

    df = pd.read_csv(PATHS["targets_dir"] / "targets.csv")
    return df[target_col].values


def combine_to_sparse(sparse: list[csr_matrix], dense: list[np.ndarray]) -> csr_matrix:
    """Объединяет sparse и dense фичи в единую sparse-матрицу."""
    if dense:
        # Конвертируем dense в sparse и объединяем
        dense_as_sparse = [csr_matrix(x) for x in dense]
        combined = hstack(sparse + dense_as_sparse)
    else:
        combined = hstack(sparse)
    return combined.tocsr()  # На выходе sparse


def combine_to_dense(sparse: list[csr_matrix], dense: list[np.ndarray]) -> np.ndarray:
    """Объединяет все фичи в единую dense-матрицу."""
    if sparse:
        # Конвертируем sparse OHE в dense
        sparse_as_dense = [x.toarray() for x in sparse]
        combined = np.hstack(sparse_as_dense + dense)
    else:
        combined = np.hstack(dense)
    return combined  # На выходе dense


def load_features_separately(feature_files: list[str]) -> tuple[list[csr_matrix], list[np.ndarray]]:
    """Загружает фичи, разделяя их на sparse (OHE) и dense (эмбеддинги)."""
    sparse_matrices = []
    dense_matrices = []
    
    for fname in feature_files:
        fpath = Path(PATHS["feats_dir"]) / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Feature file not found: {fpath}")

        if fname.endswith(".npz"):
            arr = load_npz_feature(fpath)
        else:
            arr = np.load(fpath)

        if issparse(arr):
            sparse_matrices.append(arr.tocsr())
        else:
            dense_matrices.append(arr)

    print(f"Loaded {len(sparse_matrices)} sparse and {len(dense_matrices)} dense feature arrays.")
    return sparse_matrices, dense_matrices

def load_npz_feature(fpath):
    arr = np.load(fpath, allow_pickle=True)
    # If multiple arrays stored in npz, take first
    first = arr[arr.files[0]]
    # If object array wrapping sparse
    if isinstance(first, np.ndarray) and first.dtype == object:
        first = first.item()
    # If it's already a scipy sparse matrix, fine
    return first

def subset_by_idx(X, idx: np.ndarray): 
    '''Subset X by row indices idx, handling dense and sparse arrays efficiently.'''
    return X[idx] if issparse(X) else X[idx, :]