import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from src.utils.io_utils import get_logger, ensure_dir, save_json, save_yaml
from sklearn.metrics import classification_report
from src.models.multihead_model import MultiHeadModel


class DLTrainer:
    def __init__(self, model, train_dataset, val_dataset, cfg, out_dir="../../runs/dl"):
        self.model: MultiHeadModel = model
        self.cfg: dict = cfg
        
        # логгер и директория
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{ts}_{cfg['model']['name']}"
        self.run_dir = Path(out_dir) / run_name
        ensure_dir(self.run_dir)
        
        self.logger = get_logger("DLTrainer", self.run_dir)
        save_yaml(self.cfg, self.run_dir / "config.yaml")

        self.device = self._find_best_device()
        self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["training"].get("batch_size", 32),
            shuffle=True,
            num_workers=cfg["training"].get("num_workers", 2)
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["training"].get("batch_size", 32),
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 2)
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["training"].get("lr", 1e-4)
        )
        self.epochs = cfg["training"].get("epochs", 10)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            inputs, targets = self._move_batch_to_device(batch)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)  # dict: {"subcategory": ..., "priority": ..., "avariya": ...}
            loss = self._compute_loss(outputs, targets)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        preds, gts = {h: [] for h in self.model.heads}, {h: [] for h in self.model.heads}
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = self._move_batch_to_device(batch)
                outputs = self.model(inputs)
                loss = 0

                loss += self._compute_loss(outputs, targets)
                total_loss += loss.item()

        # можно вернуть отчёт для каждой головы
        reports = {h: classification_report(gts[h], preds[h], output_dict=True, zero_division=0) for h in self.model.heads}
        return total_loss / len(self.val_loader), reports


    def fit(self):
        best_val = -np.inf
        best_state = None

        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"Starting epoch {epoch}")
            
            # train_loss = self.train_epoch()
            val_loss, reports = self.validate()
            
            # for now using weighted F1 for classification
            cls_heads = []
            for name, cfg in self.cfg["model"]["heads"].items():
                if cfg.get("type") in ["classification", "binary"]:
                    cls_heads.append(name)
            score = np.mean([reports[h]["macro avg"]["f1-score"] for h in cls_heads]) if cls_heads else 0.0

            self.logger.info(f"Epoch {epoch}: "
                            #  f"train_loss={train_loss:.4f}, "
                             f"val_loss={val_loss:.4f}, "
                             f"val_f1={score:.4f}")

            if score > best_val:
                best_val = score
                best_state = self.model.state_dict().copy()
                
                metrics = {
                    # "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": score
                }
                
                checkpoint = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics
                }
                
                torch.save(checkpoint, self.run_dir / "best_model_with_metrics.pt")
                save_json(metrics, self.run_dir / "best_model_metrics.json")
                self.logger.info(f"Saved new best model to {self.run_dir / 'best_model.pt'}")

        # финальное сохранение
        self.logger.info(f"Best val f1: {best_val:.4f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        save_yaml(self.cfg, self.run_dir / "best_model_config.yaml")

    def _compute_loss(self, outputs, targets):
        loss = 0
        for head_name, logits in outputs.items():
            target = targets[head_name].to(self.device)
            head_type = self.model.heads_config[head_name]["type"]

            if head_type == "classification":
                loss += nn.CrossEntropyLoss()(logits, target)
            elif head_type == "binary":
                loss += nn.BCEWithLogitsLoss()(logits, target)
            elif head_type == "regression":
                loss += nn.MSELoss()(logits, target)

        return loss

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

    def _move_batch_to_device(self, batch) -> tuple[dict, dict]:
        """Move entire batch to device"""
        device_batch = {"inputs": {}, "targets": {}}
        
        # Move inputs
        for key, value in batch["inputs"].items():
            if torch.is_tensor(value):
                device_batch["inputs"][key] = value.to(self.device)
            else:
                device_batch["inputs"][key] = value
        
        # Move targets
        for key, value in batch["targets"].items():
            if torch.is_tensor(value):
                device_batch["targets"][key] = value.to(self.device)
            else:
                device_batch["targets"][key] = value
        
        return device_batch["inputs"], device_batch["targets"]
            