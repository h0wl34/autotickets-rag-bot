import argparse
from torch.utils.data import ConcatDataset
import torch

from src.utils.io_utils import get_logger, load_yaml
from src.training.dl_trainer import DLTrainer
from src.models.multihead_model import MultiHeadModel
from src.data.ticket_dataset import TicketDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Путь к файлу конфигурации")
    p.add_argument("--checkpoint", type=str, default=None, help="Путь к директории с чекпоинтом")
    p.add_argument("--full_dataset", action="store_true", help="Использовать полный датасет (train + val)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    logger = get_logger("DLTrainer", tee_stdout=False)

    feature_files={k: v["path"] for k, v in cfg["features"].items()}
    heads_config=cfg["model"]["heads"]

    train_dataset = TicketDataset(
        feature_files=feature_files,
        heads_config=heads_config,
        split_name="train"
    )
    val_dataset = TicketDataset(
        feature_files=feature_files,
        heads_config=heads_config,
        split_name="test"
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    model = MultiHeadModel(
        feature_config=cfg["features"],
        heads_config=heads_config,
        hidden_dims=cfg["model"]["hidden_dims"]
    )
    
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) 
        logger.info("Loading checkpoint...")
        model_state = checkpoint.get("model_state")
        opt_state = checkpoint.get("optimizer_state")
        model_state = model_state if model_state is not None else checkpoint
        checkpoint = {"model_state": model_state, "optimizer_state": opt_state}
    
    if (args.full_dataset):
        full_dataset = ConcatDataset([train_dataset, val_dataset])
        trainer = DLTrainer(
            model=model,
            train_dataset=full_dataset,
            val_dataset=None,
            cfg=cfg,
            out_dir="./runs/dl",
            checkpoint=checkpoint
            
        )
        logger.info("Using full dataset for training")
    else:
        trainer = DLTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            cfg=cfg,
            out_dir="./runs/dl",
            checkpoint=checkpoint
        )

    trainer.fit()

if __name__ == '__main__':
    main()
    
    
    