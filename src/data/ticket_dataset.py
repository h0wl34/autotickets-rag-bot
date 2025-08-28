import torch
from torch.utils.data import Dataset

from src.utils.data_loader import load_split, load_features, load_target


class TicketDataset(Dataset):
    """
    Generic dataset for tickets with multiple feature modalities.
    Features are expected to be loaded and aligned with the same row indices.
    """

    def __init__(self, feature_files: dict, heads_config: dict, split_name: str):
        self.idx = load_split(split_name)
        self.features = {}
        
        for feat_name, file_path in feature_files.items():
            X = load_features(file_path, output_format="dense")
            # cat features -> LongTensor
            if feat_name in ['oe_cats', 'bin_cats']:
                self.features[feat_name] = torch.tensor(X[self.idx], dtype=torch.long)
            else:
                self.features[feat_name] = torch.tensor(X[self.idx], dtype=torch.float32)

        self.targets = {}
        for name, cfg in heads_config.items():
            t = torch.tensor(load_target(name)[self.idx])
            if cfg["type"] == "classification":
                # cross-entropy expects LongTensor labels
                self.targets[name] = t.long()
            else:  # binary/regression -> float
                self.targets[name] = t.float()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i) -> dict[str, dict[str, torch.Tensor]]:
        x = {k: v[i] for k, v in self.features.items()}
        y = {k: v[i] for k, v in self.targets.items()}
        return {"inputs": x, "targets": y}
    
