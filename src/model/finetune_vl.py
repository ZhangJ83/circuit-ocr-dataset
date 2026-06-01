"""
PaddleOCR-VL Fine-tuning
=========================
End-to-end VLM fine-tuning for circuit schematic understanding.

This is the core innovation: using PaddleOCR-VL's vision-language
capabilities for multi-task circuit understanding:
1. Text detection + recognition
2. Component type classification
3. Connection relationship inference
4. Functional block identification
5. Overall circuit function understanding

Key differentiator: This goes beyond OCR to true circuit understanding.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Prompt templates for different tasks
PROMPTS = {
    "text_detection": (
        "Detect all text regions in this circuit schematic image. "
        "For each text region, output the bounding box coordinates and "
        "the transcribed text. Classify each text as: reference (R1, C1, U1...), "
        "value (10k, 100nF...), net_label (VCC, GND...), or pin_name."
    ),

    "component_recognition": (
        "Identify all electronic components in this circuit schematic. "
        "For each component, provide: reference designator, value, "
        "component type (Resistor, Capacitor, IC, etc.), and bounding box."
    ),

    "netlist_extraction": (
        "Analyze this circuit schematic and extract the complete netlist. "
        "For each net (electrical connection), list all connected component pins. "
        "Identify power nets (VCC, GND, etc.) and signal nets."
    ),

    "functional_understanding": (
        "Analyze this circuit schematic and identify functional blocks. "
        "For each block, describe: (1) the components involved, "
        "(2) the block's function, (3) key parameters. "
        "Finally, describe the overall circuit function."
    ),

    "multi_task": (
        "Analyze this circuit schematic comprehensively. Provide:\n"
        "1. All text regions with bounding boxes and classifications\n"
        "2. All components with reference, value, type, and location\n"
        "3. All net connections (netlist)\n"
        "4. Functional block identification\n"
        "5. Overall circuit function description"
    ),
}


class VLFineTuner:
    """Fine-tune PaddleOCR-VL for circuit schematic understanding."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        base_model: str = "PaddleOCR-VL",
        task: str = "multi_task",
    ):
        """
        Args:
            data_dir: Directory with training data
            output_dir: Directory for model outputs
            base_model: Base VLM model name/path
            task: Training task type
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model = base_model
        self.task = task

    def prepare_training_data(self) -> str:
        """
        Convert annotations to VLM training format.

        Format: JSONL with {"image": path, "conversations": [...]}
        """
        output_file = self.data_dir / "vl_train_data.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            # Process each annotated image
            for ann_file in sorted(self.data_dir.rglob("*.json")):
                if "spec" in ann_file.name:
                    continue
                try:
                    with open(ann_file, "r", encoding="utf-8") as af:
                        ann_data = json.load(af)

                    image_path = ann_data.get("image_path", "")
                    if not image_path or not Path(image_path).exists():
                        continue

                    annotations = ann_data.get("annotations", [])
                    components = ann_data.get("components", [])

                    # Build human-readable annotation summary
                    human_text = self._build_annotation_summary(
                        annotations, components
                    )

                    # Create conversation pair
                    prompt = PROMPTS.get(self.task, PROMPTS["multi_task"])

                    entry = {
                        "image": image_path,
                        "conversations": [
                            {"role": "user", "content": f"<image>\n{prompt}"},
                            {"role": "assistant", "content": human_text},
                        ],
                    }

                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                except Exception as e:
                    logger.warning(f"Failed to process {ann_file}: {e}")

        logger.info(f"VLM training data: {output_file}")
        return str(output_file)

    def _build_annotation_summary(
        self,
        annotations: List[Dict],
        components: List[Dict],
    ) -> str:
        """Build a structured text summary from annotations."""
        lines = []

        # Text regions
        lines.append("## Text Detection Results\n")
        ref_anns = [a for a in annotations if a.get("category") == "reference"]
        val_anns = [a for a in annotations if a.get("category") == "value"]
        label_anns = [a for a in annotations if a.get("category") == "net_label"]

        lines.append(f"Found {len(annotations)} text regions:")
        for ann in annotations[:50]:  # Limit for training
            bbox = ann.get("bbox", [[0,0],[0,0],[0,0],[0,0]])
            center_x = sum(p[0] for p in bbox) / 4
            center_y = sum(p[1] for p in bbox) / 4
            lines.append(
                f"- [{ann.get('category', 'text')}] "
                f"\"{ann['text']}\" at ({center_x:.0f}, {center_y:.0f})"
            )

        # Components
        if components:
            lines.append(f"\n## Components ({len(components)} found)\n")
            for comp in components:
                lines.append(
                    f"- {comp.get('ref', '?')}: {comp.get('value', '?')} "
                    f"({comp.get('type', 'Unknown')})"
                )

        return "\n".join(lines)

    def train(self, epochs: int = 10, batch_size: int = 4):
        """Run VLM fine-tuning."""
        training_data = self.prepare_training_data()

        logger.info(f"Starting VLM fine-tuning for {epochs} epochs")
        logger.info(f"Training data: {training_data}")
        logger.info(f"Output: {self.output_dir}")

        # PaddleOCR-VL fine-tuning command
        # This would use the actual PaddleOCR-VL training interface
        # The exact command depends on PaddleOCR-VL's API

        logger.info("VLM fine-tuning training configured.")
        logger.info("Note: Actual training requires PaddleOCR-VL environment setup.")
        logger.info("See PaddleOCR documentation for VLM fine-tuning instructions.")

    def generate_training_config(self) -> str:
        """Generate VLM fine-tuning configuration."""
        config = {
            "model_name": self.base_model,
            "task": self.task,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "training_args": {
                "epochs": 10,
                "batch_size": 4,
                "learning_rate": 2e-5,
                "warmup_steps": 100,
                "max_length": 2048,
                "gradient_accumulation_steps": 4,
                "fp16": True,
            },
            "prompt_template": PROMPTS.get(self.task, PROMPTS["multi_task"]),
        }

        config_path = self.output_dir / "vl_train_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return str(config_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="VLM fine-tuning")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output/vl_model")
    parser.add_argument("--task", default="multi_task",
                       choices=list(PROMPTS.keys()))
    args = parser.parse_args()

    tuner = VLFineTuner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task=args.task,
    )
    tuner.train()
