import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
import sys
import logging


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)
        
        
def save_json(obj: dict[str, Any] | list, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_logger(name: str = "train", out_dir: Path | None = None, tee_stdout: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler if logs_dir is given
        if out_dir is not None:
            ensure_dir(out_dir)
            fh = logging.FileHandler(Path(out_dir) / f"{name}.log", mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        
        if tee_stdout:
            sys.stdout = TeeStream(logger, logging.INFO, sys.__stdout__)
            sys.stderr = TeeStream(logger, logging.ERROR, sys.__stderr__)

    return logger


class TeeStream:
    """Duplicate stdout/stderr to both console and logger."""
    def __init__(self, logger, level=logging.INFO, stream=sys.stdout):
        self.logger = logger
        self.level = level
        self.stream = stream  # real stdout/stderr

    def write(self, msg):
        msg = msg.strip()
        if msg:
            # Send to logger
            self.logger.log(self.level, msg)
            # Echo to original stream
            self.stream.write(msg + "\n")
            self.stream.flush()

    def flush(self):
        self.stream.flush()
    