import argparse

from src.utils.io_utils import load_yaml
from src.pipelines.dl_trainer import DLTrainer
from src.models.multihead_model import MultiHeadModel
from src.datasets.ticket_dataset import TicketDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Путь к файлу конфигурации")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    feature_files={k: v["path"] for k, v in cfg["features"].items()}
    heads_config=cfg["model"]["heads"]

    train_dataset = TicketDataset(
        feature_files=feature_files,
        heads_config=heads_config,
        split_name="test_small"
    )
    val_dataset = TicketDataset(
        feature_files=feature_files,
        heads_config=heads_config,
        split_name="val_small"
    )
    
    model = MultiHeadModel(
        feature_config=cfg["features"],
        heads_config=heads_config,
        hidden_dims=cfg["model"]["hidden_dims"]
    )

    trainer = DLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg
    )

    trainer.fit()

if __name__ == '__main__':
    main()
    
    
    