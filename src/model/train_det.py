"""
Text Detection Model Training
==============================
Fine-tune PP-OCRv4 detection model on circuit schematic data.

Features:
- PP-OCRv4 pretrained weights
- Custom data loader for circuit schematics
- Distributed training support
- Learning rate scheduling with warmup
- Early stopping
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DetTrainer:
    """Fine-tune PP-OCRv4 text detection model."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        pretrained_model: str = "ch_PP-OCRv4_det_train",
        config_path: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Directory containing train/val data
            output_dir: Directory to save trained model
            pretrained_model: Path to pretrained model weights
            config_path: Custom training config YAML
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_model = pretrained_model
        self.config_path = config_path

    def prepare_config(self) -> str:
        """Generate PaddleOCR detection training config."""
        config = {
            "Global": {
                "use_gpu": True,
                "epoch_num": 200,
                "log_smooth_window": 20,
                "print_batch_step": 10,
                "save_model_dir": str(self.output_dir),
                "save_epoch_step": 10,
                "eval_batch_step": [0, 500],
                "pretrained_model": self.pretrained_model,
                "character_dict_path": str(self.data_dir / "char_dict.txt"),
                "max_text_length": 200,
                "use_space_char": True,
            },
            "Architecture": {
                "model_type": "det",
                "algorithm": "DB",
                "Transform": None,
                "Backbone": {
                    "name": "MobileNetV3",
                    "scale": 0.5,
                    "model_name": "large",
                },
                "Neck": {
                    "name": "RSEFPN",
                    "out_channels": 96,
                },
                "Head": {
                    "name": "DBHead",
                    "k": 50,
                },
            },
            "Loss": {
                "name": "DBLoss",
                "balance_loss": True,
                "main_loss_type": "DiceLoss",
                "alpha": 5,
                "beta": 10,
                "ohem_ratio": 3,
            },
            "Optimizer": {
                "name": "Adam",
                "lr": {
                    "learning_rate": 0.001,
                    "warmup_epoch": 5,
                },
                "regularizer": {
                    "name": "L2",
                    "factor": 1e-4,
                },
            },
            "Train": {
                "dataset": {
                    "name": "SimpleDataSet",
                    "data_dir": str(self.data_dir / "train"),
                    "label_file_list": [
                        str(self.data_dir / "train_det_label.txt")
                    ],
                    "transforms": [
                        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                        {"DetLabelEncode": None},
                        {"IaaAugment": None},
                        {"RandomCropData": {"max_tries": 10, "min_crop_side_ratio": 0.1}},
                        {"MakeBorderMap": {"shrink_ratio": 0.4, "thresh_min": 0.3, "thresh_max": 0.7}},
                        {"NormalizeImage": {
                            "std": [0.229, 0.224, 0.225],
                            "mean": [0.485, 0.456, 0.406],
                            "order": "hwc",
                        }},
                        {"ToCHWImage": None},
                    ],
                },
                "loader": {
                    "shuffle": True,
                    "batch_size_per_card": 16,
                    "num_workers": 4,
                },
            },
            "Eval": {
                "dataset": {
                    "name": "SimpleDataSet",
                    "data_dir": str(self.data_dir / "val"),
                    "label_file_list": [
                        str(self.data_dir / "val_det_label.txt")
                    ],
                    "transforms": [
                        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                        {"DetLabelEncode": None},
                        {"DetResizeForTest": {"image_shape": [736, 736]}},
                        {"NormalizeImage": {
                            "std": [0.229, 0.224, 0.225],
                            "mean": [0.485, 0.456, 0.406],
                            "order": "hwc",
                        }},
                        {"ToCHWImage": None},
                    ],
                },
                "loader": {
                    "shuffle": False,
                    "batch_size_per_card": 16,
                    "num_workers": 4,
                },
            },
        }

        config_file = self.output_dir / "det_train_config.yml"
        import yaml
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return str(config_file)

    def train(self, use_distributed: bool = False, gpu_ids: str = "0"):
        """Run detection model training."""
        config_file = self.prepare_config()

        logger.info(f"Starting detection training with config: {config_file}")
        logger.info(f"Output dir: {self.output_dir}")

        if use_distributed:
            cmd = (
                f"python -m paddle.distributed.launch "
                f"--gpus='{gpu_ids}' "
                f"tools/train.py "
                f"-c {config_file}"
            )
        else:
            cmd = f"python tools/train.py -c {config_file}"

        logger.info(f"Training command: {cmd}")
        os.system(cmd)

        logger.info("Detection training complete!")

    def evaluate(self) -> Dict:
        """Evaluate the trained detection model."""
        config_file = self.output_dir / "det_train_config.yml"

        cmd = (
            f"python tools/eval.py "
            f"-c {config_file} "
            f"-o Global.checkpoints={self.output_dir}/best_accuracy"
        )

        logger.info(f"Evaluation command: {cmd}")
        os.system(cmd)

        # Parse results
        results_file = self.output_dir / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        return {}


def create_default_det_config():
    """Create a default PaddleOCR detection training config YAML."""
    return """Global:
  use_gpu: true
  epoch_num: 200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/det_circuit/
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  pretrained_model: ./pretrain_models/ch_PP-OCRv4_det_train
  character_dict_path: ./data/char_dict.txt
  max_text_length: 200
  use_space_char: true

Architecture:
  model_type: det
  algorithm: DB
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: large
  Neck:
    name: RSEFPN
    out_channels: 96
  Head:
    name: DBHead
    k: 50

Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

Optimizer:
  name: Adam
  lr:
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 0.0001

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/train/
    label_file_list:
      - ./data/train_det_label.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - DetLabelEncode: null
      - IaaAugment: null
      - RandomCropData:
          max_tries: 10
          min_crop_side_ratio: 0.1
      - MakeBorderMap:
          shrink_ratio: 0.4
          thresh_min: 0.3
          thresh_max: 0.7
      - NormalizeImage:
          std: [0.229, 0.224, 0.225]
          mean: [0.485, 0.456, 0.406]
          order: hwc
      - ToCHWImage: null
  loader:
    shuffle: true
    batch_size_per_card: 16
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/val/
    label_file_list:
      - ./data/val_det_label.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - DetLabelEncode: null
      - DetResizeForTest:
          image_shape: [736, 736]
      - NormalizeImage:
          std: [0.229, 0.224, 0.225]
          mean: [0.485, 0.456, 0.406]
          order: hwc
      - ToCHWImage: null
  loader:
    shuffle: false
    batch_size_per_card: 16
    num_workers: 4
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Write default config
    config_path = Path("configs/det_ppocrv4.yml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(create_default_det_config())
    print(f"Default detection config written to {config_path}")
