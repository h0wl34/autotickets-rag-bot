import argparse
from src.utils.config_helper import resolve_config
from src.pipelines.train_pipeline import TrainingPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Путь к файлу конфигурации")
    p.add_argument("--target", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--features", nargs="+", default=None)
    p.add_argument("--kfold", type=bool, default=False)
    return p.parse_args()

def main():
    args = parse_args()
        
    overrides = {}
    if args.target:
        overrides["target"] = args.target
    if args.model:
        overrides["model"] = {"name": args.model}
    if args.features:
        overrides["features"] = args.features
    
    cfg = resolve_config(args.config, overrides=overrides)
    
    pipeline = TrainingPipeline(cfg)
    if (args.kfold):
        pipeline.run_kfold()
    else:
        pipeline.run_train(cfg["full_train"])

if __name__ == "__main__":
    main()