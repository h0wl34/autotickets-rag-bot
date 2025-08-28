import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pprint
from transformers import get_cosine_schedule_with_warmup

from src.utils.io_utils import get_logger, ensure_dir, save_json, save_yaml
from sklearn.metrics import classification_report
from src.models.multihead_model import MultiHeadModel
from src.training.losses import focal_loss

class DLTrainer:
    def __init__(self, model, train_dataset, cfg, out_dir, checkpoint: dict = None, val_dataset=None):
        self.model: MultiHeadModel = model
        self.cfg: dict = cfg
        
        # логгер и директория
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{ts}_{cfg['model']['name']}"
        self.run_dir = Path(out_dir) / run_name
        ensure_dir(self.run_dir)
        
        self.logger = get_logger("DLTrainer", self.run_dir, tee_stdout=True)
        save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")

        self.device = self._find_best_device()
        self.model.to(self.device)  # move first
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["training"].get("lr", 1e-4)
        ) 
        
        if checkpoint is not None:
            model_state = checkpoint.get("model_state")
            if model_state is None:
                self.logger.info("No model state found, skipping")
            else:
                self.model.load_state_dict(checkpoint["model_state"])
                self.logger.info("Loaded model state")
                
            if "optimizer_state" not in checkpoint:
                self.logger.info("No optimizer state found, skipping")
            else:    
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                # Ensure optimizer params are on the right device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.device)
                self.logger.info("Loaded optimizer state")
                
            if "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                self.logger.info("Loaded scheduler state")
            else:
                self.logger.info("No scheduler state found, skipping")

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
        ) if val_dataset is not None else None
             
        self.epochs = cfg["training"].get("epochs", 10)
        
        num_training_steps = len(self.train_loader) * self.epochs
        num_warmup_steps = int(0.2 * num_training_steps)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training batches", leave=False):
            inputs, targets = self._move_batch_to_device(batch)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)  # dict: {"subcategory": ..., "priority": ..., "avariya": ...}
            loss = self._compute_loss(outputs, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        preds, gts = {h: [] for h in self.model.heads}, {h: [] for h in self.model.heads}
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation batches", leave=False):
                inputs, targets = self._move_batch_to_device(batch)
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
                total_loss += loss.item()

                for head_name, logits in outputs.items():
                    head_type = self.model.heads_config[head_name]["type"]

                    if head_type == "classification":
                        pred = torch.argmax(logits, dim=1)
                        preds[head_name].extend(pred.cpu().numpy())
                        gts[head_name].extend(targets[head_name].cpu().numpy())

                    elif head_type == "binary":
                        pred = (torch.sigmoid(logits) > 0.5).long()
                        preds[head_name].extend(pred.cpu().numpy())
                        gts[head_name].extend(targets[head_name].cpu().numpy())

                    elif head_type == "regression":
                        preds[head_name].extend(logits.cpu().numpy().flatten())
                        gts[head_name].extend(targets[head_name].cpu().numpy().flatten())
        
        reports = {}
        for h in self.model.heads:
            if len(gts[h]) > 0:  # avoid empty report
                if self.model.heads_config[h]["type"] in ["classification", "binary"]:
                    reports[h] = classification_report(
                        gts[h], preds[h], output_dict=True, zero_division=0
                    )
                else:
                    reports[h] = {}  # or some regression metric
        return total_loss / len(self.val_loader), reports


    def fit(self):
        best_val = -np.inf
        best_state = None

        cls_heads = [name for name, cfg in self.cfg["model"]["heads"].items()
                    if cfg.get("type") in ["classification", "binary"]]

        self.logger.info(f"Starting training, {self.epochs} epochs")
        self.logger.info(f"Features configuration:\n{pprint.pformat(self.cfg['features'], indent=2)}")
        self.logger.info(f"Feature dropout:\n{pprint.pformat(self.cfg['training']['feature_dropout'], indent=2)}")

        for epoch in tqdm(range(1, self.epochs + 1), desc="Epochs", leave=True):
            self.logger.info(f"Starting epoch {epoch}")
            train_loss = self.train_epoch()

            val_loss, reports, score = None, None, None

            if self.val_loader is not None:
                val_loss, reports = self.validate()
                score = np.mean([reports[h]["macro avg"]["f1-score"] for h in cls_heads]) if cls_heads else 0.0
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                                f"val_loss={val_loss:.4f}, val_f1={score:.4f}")
            else:
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} (no validation)")

            # Save best model if validation exists
            if score is not None and score > best_val:
                best_val = score
                best_state = self.model.state_dict().copy()
                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": score
                }
                checkpoint = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics
                }
                torch.save(checkpoint, self.run_dir / "best_model_with_metrics.pt")
                save_json(metrics, self.run_dir / "best_model_metrics.json")
                self.logger.info(f"Saved new best model to {self.run_dir / 'best_model_with_metrics.pt' }")
            else: 
                checkpoint = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "epoch": epoch
                }
                checkpoint_path = self.run_dir/'intermediate_checkpoint.pt'
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"Intermediate checkpoint saved to {checkpoint_path}")
        
        final_model_path = self.run_dir / "final_model.pt"
        # Final save
        if best_state is not None:
            self.model.load_state_dict(best_state)
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")

        if best_val != -np.inf:
            self.logger.info(f"Best val f1: {best_val:.4f}")        
        
    def _compute_loss(self, outputs, targets):
        total_loss = 0.0

        for head_name, head_cfg in self.model.heads_config.items():
            pred = outputs[head_name]
            target = targets[head_name].to(self.device)
            loss_weight = head_cfg.get("loss_weight", 1)

            if head_cfg["type"] == "binary":
                # Compute pos_weight safely
                if "pos_weight" in head_cfg and head_cfg["pos_weight"] is not None:
                    pos_weight = head_cfg["pos_weight"]

                    # Avoid extremely large pos_weight
                    pos_weight = min(pos_weight, 5.0)  # cap at 5
                    pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
                else:
                    pos_weight = None

                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                pred = pred.view(-1)
                target = target.view(-1).float()
                loss = loss_fn(pred, target)
                # loss = focal_loss(pred, target, alpha=0.25, gamma=2)

            elif head_cfg["type"] == "classification":
                weights = None
                if "class_weights" in head_cfg and head_cfg["class_weights"] is not None:
                    weights = torch.tensor(head_cfg["class_weights"], dtype=torch.float32).to(self.device)
                #     loss_fn = nn.CrossEntropyLoss(weight=weights)
                # else:
                #     loss_fn = nn.CrossEntropyLoss()

                # loss = loss_fn(pred, target.long())
                loss = focal_loss(pred, target.long(), alpha=weights, gamma=2)

            else:
                raise ValueError(f"Unsupported head type: {head_cfg['type']}")

            total_loss += loss * loss_weight

        return total_loss

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
            