"""
Dataset Builder
===============
Orchestrate the complete dataset construction pipeline:
1. Scrape GitHub for KiCad projects
2. Parse schematic files
3. Render to PNG images
4. Generate annotations
5. Generate synthetic data
6. Apply degradation augmentation
7. Split into train/val/test sets
8. Export in PaddleOCR training format

This is the top-level orchestrator that ties all data pipeline modules together.
"""

import json
import logging
import random
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .github_scraper import GitHubKiCadScraper
from .kicad_parser import KiCadParser
from .renderer import SchematicRenderer
from .annotation_generator import AnnotationGenerator
from .synthetic_generator import SyntheticSchematicGenerator
from .degradation import SchematicDegradation

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about the constructed dataset."""
    total_schematics: int = 0
    total_images: int = 0
    total_annotations: int = 0
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0
    synthetic_images: int = 0
    degraded_images: int = 0
    component_types: Dict[str, int] = field(default_factory=dict)
    text_categories: Dict[str, int] = field(default_factory=dict)
    char_count: int = 0
    unique_chars: int = 0


class DatasetBuilder:
    """
    Build the complete circuit schematic OCR dataset.

    Orchestrates: scraping → parsing → rendering → annotation → augmentation → split
    """

    def __init__(
        self,
        project_dir: str,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            project_dir: Root directory for the project
            config: Optional configuration overrides
        """
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"

        # Default config
        self.config = {
            "dpi": 300,
            "max_repos": 200,
            "min_file_size": 1000,
            "synthetic_count": 300,
            "degradation_variants": 3,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "circuit_types": ["mixed", "digital", "analog", "power"],
            "complexities": ["simple", "medium", "complex"],
            "github_token": None,
            "kicad_cli_path": None,
        }
        if config:
            self.config.update(config)

        # Initialize modules
        self.parser = KiCadParser()
        self.renderer = SchematicRenderer(
            output_dir=str(self.data_dir / "rendered"),
            dpi=self.config["dpi"],
            kicad_cli_path=self.config.get("kicad_cli_path"),
        )
        self.annotation_gen = AnnotationGenerator(
            mm_to_px=self.config["dpi"] / 25.4,
        )
        self.synth_gen = SyntheticSchematicGenerator(
            output_dir=str(self.data_dir / "synthetic"),
        )
        self.stats = DatasetStats()

    def build(
        self,
        skip_scraping: bool = False,
        skip_synthetic: bool = False,
        skip_degradation: bool = False,
    ) -> DatasetStats:
        """
        Execute the complete dataset building pipeline.

        Args:
            skip_scraping: Skip GitHub scraping (use existing data)
            skip_synthetic: Skip synthetic data generation
            skip_degradation: Skip degradation augmentation

        Returns:
            DatasetStats with dataset statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Dataset Construction Pipeline")
        logger.info("=" * 60)

        # Step 1: Scrape GitHub for KiCad projects
        kicad_files = []
        if not skip_scraping:
            kicad_files = self._step_scrape()
        else:
            kicad_files = self._find_existing_files()

        # Step 2: Parse + Render + Annotate real schematics
        real_data = self._step_process_real(kicad_files)

        # Step 3: Generate synthetic data
        synthetic_data = []
        if not skip_synthetic:
            synthetic_data = self._step_generate_synthetic()

        # Step 4: Apply degradation augmentation
        if not skip_degradation:
            self._step_apply_degradation(real_data + synthetic_data)

        # Step 5: Split dataset
        self._step_split_dataset(real_data, synthetic_data)

        # Step 6: Export in PaddleOCR format
        self._step_export_paddleocr_format()

        # Step 7: Generate character dictionary
        self._step_generate_char_dict()

        # Step 8: Save statistics
        self._save_stats()

        logger.info("=" * 60)
        logger.info("Dataset Construction Complete!")
        logger.info(f"  Total images: {self.stats.total_images}")
        logger.info(f"  Total annotations: {self.stats.total_annotations}")
        logger.info(f"  Train: {self.stats.train_images}")
        logger.info(f"  Val: {self.stats.val_images}")
        logger.info(f"  Test: {self.stats.test_images}")
        logger.info("=" * 60)

        return self.stats

    def _step_scrape(self) -> List[str]:
        """Step 1: Scrape GitHub for KiCad projects."""
        logger.info("Step 1: Scraping GitHub for KiCad projects...")

        scraper = GitHubKiCadScraper(
            output_dir=str(self.data_dir / "raw"),
            github_token=self.config.get("github_token"),
            max_repos=self.config["max_repos"],
            min_file_size=self.config["min_file_size"],
        )

        projects = scraper.run()
        kicad_files = []
        for p in projects:
            kicad_files.extend(p.sch_files)

        logger.info(f"Found {len(kicad_files)} schematic files from {len(projects)} repos")
        return kicad_files

    def _find_existing_files(self) -> List[str]:
        """Find existing .kicad_sch files in data/raw."""
        raw_dir = self.data_dir / "raw"
        if not raw_dir.exists():
            logger.warning("No data/raw directory found")
            return []

        files = list(raw_dir.rglob("*.kicad_sch"))
        logger.info(f"Found {len(files)} existing schematic files")
        return [str(f) for f in files]

    def _step_process_real(self, kicad_files: List[str]) -> List[Dict]:
        """Step 2: Parse, render, and annotate real schematics."""
        logger.info(f"Step 2: Processing {len(kicad_files)} real schematics...")

        processed = []
        for i, sch_path in enumerate(kicad_files):
            try:
                # Parse
                data = self.parser.parse(sch_path)

                # Skip very small circuits
                if len(data.components) < 3:
                    continue

                # Render
                png_path = self.renderer.render_schematic(sch_path)
                if not png_path:
                    continue

                # Generate annotations
                from PIL import Image
                img = Image.open(png_path)
                annotations = self.annotation_gen.generate_annotations(
                    data, img.width, img.height
                )

                if not annotations:
                    continue

                # Save annotation file
                ann_path = str(
                    self.data_dir / "annotations" / f"{Path(sch_path).stem}.json"
                )
                ann_data = {
                    "image_path": png_path,
                    "image_width": img.width,
                    "image_height": img.height,
                    "schematic_path": sch_path,
                    "annotations": [
                        {
                            "text": a.text,
                            "bbox": a.bbox,
                            "category": a.category,
                            "component_ref": a.component_ref,
                        }
                        for a in annotations
                    ],
                    "components": [
                        {
                            "ref": c.reference,
                            "value": c.value,
                            "type": c.properties.get("_type", "Unknown"),
                        }
                        for c in data.components
                    ],
                }
                Path(ann_path).parent.mkdir(parents=True, exist_ok=True)
                with open(ann_path, "w", encoding="utf-8") as f:
                    json.dump(ann_data, f, indent=2, ensure_ascii=False)

                processed.append({
                    "image_path": png_path,
                    "annotation_path": ann_path,
                    "source": "real",
                    "num_components": len(data.components),
                    "num_annotations": len(annotations),
                })

                self.stats.total_schematics += 1
                self.stats.total_annotations += len(annotations)

                if (i + 1) % 50 == 0:
                    logger.info(f"  Processed {i+1}/{len(kicad_files)}")

            except Exception as e:
                logger.warning(f"Failed to process {sch_path}: {e}")

        logger.info(f"Processed {len(processed)} real schematics")
        return processed

    def _step_generate_synthetic(self) -> List[Dict]:
        """Step 3: Generate synthetic schematics."""
        logger.info("Step 3: Generating synthetic schematics...")

        all_results = []
        count_per_type = self.config["synthetic_count"] // len(self.config["circuit_types"])

        for ctype in self.config["circuit_types"]:
            for complexity in self.config["complexities"]:
                n = max(1, count_per_type // len(self.config["complexities"]))
                results = self.synth_gen.batch_generate(
                    count=n,
                    circuit_type=ctype,
                    complexity=complexity,
                    base_name=f"synth_{ctype}_{complexity}",
                )

                for r in results:
                    if not r.get("success"):
                        continue

                    # Parse the generated .kicad_sch to get annotations
                    try:
                        data = self.parser.parse(r["sch_path"])
                        from PIL import Image
                        img = Image.open(r["png_path"])
                        annotations = self.annotation_gen.generate_annotations(
                            data, img.width, img.height
                        )

                        # Save annotation
                        ann_path = r["png_path"].replace(".png", ".json")
                        ann_data = {
                            "image_path": r["png_path"],
                            "image_width": img.width,
                            "image_height": img.height,
                            "annotations": [
                                {
                                    "text": a.text,
                                    "bbox": a.bbox,
                                    "category": a.category,
                                    "component_ref": a.component_ref,
                                }
                                for a in annotations
                            ],
                        }
                        with open(ann_path, "w", encoding="utf-8") as f:
                            json.dump(ann_data, f, indent=2, ensure_ascii=False)

                        all_results.append({
                            "image_path": r["png_path"],
                            "annotation_path": ann_path,
                            "source": "synthetic",
                            "num_components": r["num_components"],
                            "num_annotations": len(annotations),
                        })

                        self.stats.synthetic_images += 1
                        self.stats.total_annotations += len(annotations)

                    except Exception as e:
                        logger.warning(f"Failed to annotate synthetic: {e}")

        logger.info(f"Generated {len(all_results)} synthetic schematics")
        return all_results

    def _step_apply_degradation(self, all_data: List[Dict]):
        """Step 4: Apply degradation augmentation."""
        logger.info("Step 4: Applying degradation augmentation...")

        degraded_dir = self.data_dir / "degraded"
        degraded_dir.mkdir(parents=True, exist_ok=True)

        for item in all_data:
            image_path = item.get("image_path")
            if not image_path or not Path(image_path).exists():
                continue

            try:
                variants = SchematicDegradation.generate_degraded_variants(
                    image_path,
                    str(degraded_dir),
                    num_variants=self.config["degradation_variants"],
                )
                self.stats.degraded_images += len(variants)
            except Exception as e:
                logger.warning(f"Degradation failed for {image_path}: {e}")

        logger.info(f"Generated {self.stats.degraded_images} degraded images")

    def _step_split_dataset(self, real_data: List[Dict], synthetic_data: List[Dict]):
        """Step 5: Split into train/val/test sets."""
        logger.info("Step 5: Splitting dataset...")

        all_data = real_data + synthetic_data
        random.shuffle(all_data)

        n = len(all_data)
        n_train = int(n * self.config["train_ratio"])
        n_val = int(n * self.config["val_ratio"])

        splits = {
            "train": all_data[:n_train],
            "val": all_data[n_train:n_train + n_val],
            "test": all_data[n_train + n_val:],
        }

        for split_name, items in splits.items():
            split_dir = self.data_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for item in items:
                src = item.get("image_path")
                if src and Path(src).exists():
                    dst = split_dir / Path(src).name
                    if not dst.exists():
                        shutil.copy2(src, dst)

                    # Copy annotation
                    ann_src = item.get("annotation_path")
                    if ann_src and Path(ann_src).exists():
                        ann_dst = split_dir / Path(ann_src).name
                        if not ann_dst.exists():
                            shutil.copy2(ann_src, ann_dst)

        self.stats.train_images = len(splits["train"])
        self.stats.val_images = len(splits["val"])
        self.stats.test_images = len(splits["test"])
        self.stats.total_images = (
            self.stats.train_images + self.stats.val_images + self.stats.test_images
        )

        logger.info(
            f"Split: train={self.stats.train_images}, "
            f"val={self.stats.val_images}, test={self.stats.test_images}"
        )

    def _step_export_paddleocr_format(self):
        """Step 6: Export in PaddleOCR training format."""
        logger.info("Step 6: Exporting PaddleOCR format...")

        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            det_label_path = self.data_dir / f"{split}_det_label.txt"
            rec_label_path = self.data_dir / f"{split}_rec_label.txt"

            det_lines = []
            rec_lines = []

            for ann_file in split_dir.glob("*.json"):
                try:
                    with open(ann_file, "r", encoding="utf-8") as f:
                        ann_data = json.load(f)

                    image_path = ann_data.get("image_path", "")
                    annotations = ann_data.get("annotations", [])

                    # Detection format
                    det_entries = []
                    for ann in annotations:
                        det_entries.append({
                            "points": ann["bbox"],
                            "transcription": ann["text"],
                        })
                    det_lines.append(
                        f"{image_path}\t{json.dumps(det_entries, ensure_ascii=False)}"
                    )

                    # Recognition format (individual crops would need separate images)
                    for ann in annotations:
                        rec_lines.append(f"{image_path}\t{ann['text']}")

                except Exception as e:
                    logger.warning(f"Failed to export {ann_file}: {e}")

            with open(det_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(det_lines))
            with open(rec_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(rec_lines))

            logger.info(
                f"  {split}: {len(det_lines)} det entries, "
                f"{len(rec_lines)} rec entries"
            )

    def _step_generate_char_dict(self):
        """Step 7: Generate character dictionary."""
        logger.info("Step 7: Generating character dictionary...")

        all_annotations = []
        for split in ["train", "val"]:
            for ann_file in (self.data_dir / split).glob("*.json"):
                try:
                    with open(ann_file, "r", encoding="utf-8") as f:
                        ann_data = json.load(f)
                    for ann in ann_data.get("annotations", []):
                        from .annotation_generator import TextAnnotation
                        all_annotations.append(TextAnnotation(
                            text=ann["text"],
                            bbox=ann["bbox"],
                            category=ann["category"],
                        ))
                except Exception:
                    pass

        char_dict_path = self.data_dir / "char_dict.txt"
        chars = self.annotation_gen.generate_char_dict(
            all_annotations, str(char_dict_path)
        )

        self.stats.char_count = sum(
            len(ann.text) for ann in all_annotations
        )
        self.stats.unique_chars = len(chars)

        logger.info(f"  {self.stats.unique_chars} unique characters in dictionary")

    def _save_stats(self):
        """Save dataset statistics."""
        stats_path = self.data_dir / "dataset_stats.json"
        stats_dict = {
            "total_schematics": self.stats.total_schematics,
            "total_images": self.stats.total_images,
            "total_annotations": self.stats.total_annotations,
            "train_images": self.stats.train_images,
            "val_images": self.stats.val_images,
            "test_images": self.stats.test_images,
            "synthetic_images": self.stats.synthetic_images,
            "degraded_images": self.stats.degraded_images,
            "component_types": self.stats.component_types,
            "text_categories": self.stats.text_categories,
            "char_count": self.stats.char_count,
            "unique_chars": self.stats.unique_chars,
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=2)

        logger.info(f"Statistics saved to {stats_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build circuit OCR dataset")
    parser.add_argument("--project-dir", default=".", help="Project root directory")
    parser.add_argument("--skip-scraping", action="store_true",
                       help="Skip GitHub scraping")
    parser.add_argument("--skip-synthetic", action="store_true",
                       help="Skip synthetic generation")
    parser.add_argument("--skip-degradation", action="store_true",
                       help="Skip degradation augmentation")
    parser.add_argument("--max-repos", type=int, default=200)
    parser.add_argument("--synthetic-count", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--github-token", default=None)
    args = parser.parse_args()

    config = {
        "dpi": args.dpi,
        "max_repos": args.max_repos,
        "synthetic_count": args.synthetic_count,
        "github_token": args.github_token,
    }

    builder = DatasetBuilder(project_dir=args.project_dir, config=config)
    stats = builder.build(
        skip_scraping=args.skip_scraping,
        skip_synthetic=args.skip_synthetic,
        skip_degradation=args.skip_degradation,
    )

    print(f"\nDataset built successfully!")
    print(f"  Total images: {stats.total_images}")
    print(f"  Total annotations: {stats.total_annotations}")
