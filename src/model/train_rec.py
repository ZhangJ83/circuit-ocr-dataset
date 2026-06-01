"""
Text Recognition Model Training
================================
Fine-tune PP-OCRv4 recognition model on circuit schematic data.

Special focus on electronic symbols: Ω, μ, ±, °, etc.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RecTrainer:
    """Fine-tune PP-OCRv4 text recognition model."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        pretrained_model: str = "ch_PP-OCRv4_rec_train",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_model = pretrained_model

    def train(self, use_distributed: bool = False, gpu_ids: str = "0"):
        """Run recognition model training."""
        config_file = self._prepare_config()

        logger.info(f"Starting recognition training: {config_file}")

        if use_distributed:
            cmd = (
                f"python -m paddle.distributed.launch "
                f"--gpus='{gpu_ids}' "
                f"tools/train.py -c {config_file}"
            )
        else:
            cmd = f"python tools/train.py -c {config_file}"

        logger.info(f"Command: {cmd}")
        os.system(cmd)

    def _prepare_config(self) -> str:
        """Generate recognition training config."""
        config_file = self.output_dir / "rec_train_config.yml"

        char_dict = self.data_dir / "char_dict.txt"
        dict_size = sum(1 for _ in open(char_dict, encoding="utf-8")) if char_dict.exists() else 6623

        config = f"""Global:
  use_gpu: true
  epoch_num: 200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {self.output_dir}/
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  pretrained_model: {self.pretrained_model}
  character_dict_path: {char_dict}
  max_text_length: 100
  use_space_char: true
  infer_mode: false

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: true
          Head:
            fc_decay: 0.00001
      - SARHead:
          enc_dim: 512
          max_text_length: 100

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - SARLoss:

Optimizer:
  name: Adam
  lr:
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 0.00001

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {self.data_dir}/train/
    label_file_list:
      - {self.data_dir}/train_rec_label.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug: null
      - CTCLabelEncode: null
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: true
    batch_size_per_card: 128
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {self.data_dir}/val/
    label_file_list:
      - {self.data_dir}/val_rec_label.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - CTCLabelEncode: null
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: false
    batch_size_per_card: 128
    num_workers: 4
"""
        config_file.write_text(config)
        return str(config_file)


def create_default_rec_config():
    """Create a default recognition training config."""
    return """Global:
  use_gpu: true
  epoch_num: 200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/rec_circuit/
  save_epoch_step: 10
  eval_batch_step: [0, 500]
  pretrained_model: ./pretrain_models/ch_PP-OCRv4_rec_train
  character_dict_path: ./data/char_dict.txt
  max_text_length: 100
  use_space_char: true

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: true
          Head:
            fc_decay: 0.00001
      - SARHead:
          enc_dim: 512
          max_text_length: 100

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss: null
    - SARLoss: null

Optimizer:
  name: Adam
  lr:
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 0.00001

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/train/
    label_file_list:
      - ./data/train_rec_label.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug: null
      - CTCLabelEncode: null
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: true
    batch_size_per_card: 128
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/val/
    label_file_list:
      - ./data/val_rec_label.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - CTCLabelEncode: null
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: false
    batch_size_per_card: 128
    num_workers: 4
"""
