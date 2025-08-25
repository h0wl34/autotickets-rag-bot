import argparse
import torch
import torch_directml
from pathlib import Path
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
import numpy as np


from src.datasets.ticket_dataset import TicketDataset
from src.models.multihead_model import MultiHeadModel
from src.utils.io_utils import get_logger, save_json, load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", type=str, required=True,
                        help="Path to the experiment directory (model + config)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary classification (default=0.5)")
    return parser.parse_args()
    

def main():
    args = parse_args()

    run_path = Path(args.state_path).parent
    cfg = load_yaml(run_path / "config.yaml")
    logger = get_logger("eval", out_dir=run_path)

    # Load model
    model = MultiHeadModel(
        feature_config=cfg["features"],
        heads_config=cfg["model"]["heads"],
        hidden_dims=cfg["model"]["hidden_dims"]
    )
    
    device = torch_directml.device()
    state = torch.load(args.state_path, map_location="cpu", weights_only=False) # first load to cpu
    model.load_state_dict(state["model_state"] if "model_state" in state else state) # instantiate
    model.to(device) # only then move to gpu
    model.eval()

    # Load dataset
    feature_files = {k: v["path"] for k, v in cfg["features"].items()}
    dataset = TicketDataset(
        feature_files=feature_files,
        heads_config=cfg["model"]["heads"],
        split_name=args.split
    )

    # Run inference
    all_inputs = {k: v.to(device) for k, v in dataset.features.items()}
    with torch.no_grad():
        outputs = model(all_inputs)
        outputs = {k: v.cpu() for k, v in outputs.items()}

    metrics = {}
    for head, cfg_head in cfg["model"]["heads"].items():
        y_true = dataset.targets[head].numpy()
        y_pred_raw = outputs[head].numpy()

        if cfg_head["type"] == "classification":
            y_pred = np.argmax(y_pred_raw, axis=1)
            metrics[head] = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            logger.info(f"Classification metrics for {head}: \n{classification_report(y_true, y_pred, zero_division=0)}")

        elif cfg_head["type"] == "binary":
            probs = torch.sigmoid(torch.from_numpy(y_pred_raw)).numpy()
            y_pred = (probs >= args.threshold).astype(int)
            metrics[head] = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            logger.info(f"Binary classification metrics for {head}: \n{classification_report(y_true, y_pred, zero_division=0)}")

        elif cfg_head["type"] == "regression":
            mse = mean_squared_error(y_true, y_pred_raw)
            mae = mean_absolute_error(y_true, y_pred_raw)
            r2 = r2_score(y_true, y_pred_raw)
            metrics[head] = {"mse": mse, "mae": mae, "r2": r2}
            logger.info(f"Regression metrics for {head}: \n{metrics[head]}")

    logger.info(f"Evaluation completed")
    save_json(metrics, run_path / f"eval_metrics_{args.split}.json")


if __name__ == "__main__":
    main()
