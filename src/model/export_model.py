"""
Model Export
============
Export trained models to inference format.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export trained PaddleOCR models for inference."""

    def __init__(self, model_dir: str, output_dir: str):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_det(self, model_name: str = "det_circuit"):
        """Export detection model to inference format."""
        cmd = (
            f"python tools/export_model.py "
            f"-c {self.model_dir}/det_train_config.yml "
            f"-o Global.pretrained_model={self.model_dir}/best_accuracy "
            f"Global.save_inference_dir={self.output_dir}/{model_name}_infer"
        )
        logger.info(f"Exporting detection model: {cmd}")
        os.system(cmd)

    def export_rec(self, model_name: str = "rec_circuit"):
        """Export recognition model to inference format."""
        cmd = (
            f"python tools/export_model.py "
            f"-c {self.model_dir}/rec_train_config.yml "
            f"-o Global.pretrained_model={self.model_dir}/best_accuracy "
            f"Global.save_inference_dir={self.output_dir}/{model_name}_infer"
        )
        logger.info(f"Exporting recognition model: {cmd}")
        os.system(cmd)

    def export_all(self):
        """Export all models."""
        self.export_det()
        self.export_rec()
        logger.info(f"All models exported to {self.output_dir}")
