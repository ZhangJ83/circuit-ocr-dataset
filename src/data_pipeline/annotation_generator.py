"""
Annotation Generator
====================
Generate training annotations from KiCad parsed data.

Produces:
1. PaddleOCR detection format: image_path \t [{"points": bbox, "transcription": text}]
2. PaddleOCR recognition format: crop_image_path \t text
3. Multi-level structured annotations (text, component, net, connection)
4. Character dictionary for electronic symbols (Ω, μ, ±, °, etc.)
5. Visualization tool for quality inspection

ZERO manual labeling cost — all annotations extracted programmatically from .kicad_sch source.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .kicad_parser import SchematicData, ComponentInstance, Point

logger = logging.getLogger(__name__)


@dataclass
class TextAnnotation:
    """A single text annotation with bbox and metadata."""
    text: str
    bbox: List[List[int]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] quadrilateral
    category: str           # reference, value, net_label, pin, text
    component_ref: str = "" # Associated component reference


class AnnotationGenerator:
    """Generate training annotations from KiCad schematic data."""

    # Special characters in electronics
    ELECTRONICS_CHARS = set("Ωμμ±°℃∞→←↑↓αβγδεθλπσφωΔΣΠ")
    VALUE_CHARS = set("kKmMnNpPuUfFhHzZΩmunpfR")
    DIGIT_CHARS = set("0123456789.")
    BASIC_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    def __init__(
        self,
        mm_to_px: float = 300.0 / 25.4,
        text_padding_px: int = 4,
    ):
        """
        Args:
            mm_to_px: Conversion factor from mm to pixels
            text_padding_px: Padding around text for bbox
        """
        self.mm_to_px = mm_to_px
        self.text_padding_px = text_padding_px

    def generate_annotations(
        self,
        data: SchematicData,
        image_width: int,
        image_height: int,
    ) -> List[TextAnnotation]:
        """
        Generate all text annotations from parsed schematic data.

        Args:
            data: Parsed schematic data
            image_width: Output image width in pixels
            image_height: Output image height in pixels

        Returns:
            List of TextAnnotation objects
        """
        annotations = []

        # Component properties (Reference, Value, Footprint, etc.)
        for comp in data.components:
            # Reference text
            if comp.reference:
                bbox = self._compute_text_bbox(
                    comp.reference,
                    comp.position.x,
                    comp.position.y,
                    image_width, image_height,
                    offset_y=-5,  # Above component
                )
                if bbox:
                    annotations.append(TextAnnotation(
                        text=comp.reference,
                        bbox=bbox,
                        category="reference",
                        component_ref=comp.reference,
                    ))

            # Value text
            if comp.value:
                # Value is typically below the component
                bbox = self._compute_text_bbox(
                    comp.value,
                    comp.position.x,
                    comp.position.y,
                    image_width, image_height,
                    offset_y=5,  # Below component
                )
                if bbox:
                    annotations.append(TextAnnotation(
                        text=comp.value,
                        bbox=bbox,
                        category="value",
                        component_ref=comp.reference,
                    ))

            # Pin names/numbers for ICs
            if len(comp.pins) > 4:
                for pin in comp.pins:
                    if pin.absolute_pos and pin.pin_name and pin.pin_name != "~":
                        bbox = self._compute_text_bbox(
                            pin.pin_name,
                            pin.absolute_pos.x,
                            pin.absolute_pos.y,
                            image_width, image_height,
                        )
                        if bbox:
                            annotations.append(TextAnnotation(
                                text=pin.pin_name,
                                bbox=bbox,
                                category="pin",
                                component_ref=comp.reference,
                            ))

        # Net labels
        for label in data.labels:
            bbox = self._compute_text_bbox(
                label.name,
                label.position.x,
                label.position.y,
                image_width, image_height,
            )
            if bbox:
                annotations.append(TextAnnotation(
                    text=label.name,
                    bbox=bbox,
                    category="net_label",
                ))

        # Text annotations
        for text in data.texts:
            bbox = self._compute_text_bbox(
                text.text,
                text.position.x,
                text.position.y,
                image_width, image_height,
            )
            if bbox:
                annotations.append(TextAnnotation(
                    text=text.text,
                    bbox=bbox,
                    category="text",
                ))

        # Filter out empty, hidden, or invalid text
        annotations = [
            a for a in annotations
            if a.text.strip() and a.text != "~" and len(a.text) > 0
        ]

        logger.info(
            f"Generated {len(annotations)} annotations: "
            f"{sum(1 for a in annotations if a.category == 'reference')} refs, "
            f"{sum(1 for a in annotations if a.category == 'value')} vals, "
            f"{sum(1 for a in annotations if a.category == 'net_label')} labels"
        )

        return annotations

    def _compute_text_bbox(
        self,
        text: str,
        x_mm: float,
        y_mm: float,
        img_width: int,
        img_height: int,
        offset_x: float = 0,
        offset_y: float = 0,
        font_size_px: int = 16,
    ) -> Optional[List[List[int]]]:
        """
        Compute a bounding box for a text string at given position.

        Args:
            text: The text content
            x_mm, y_mm: Position in mm
            img_width, img_height: Image dimensions
            offset_x, offset_y: Additional offset in mm

        Returns:
            Quadrilateral bbox [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] or None
        """
        # Convert mm to pixels
        cx = int((x_mm + offset_x) * self.mm_to_px)
        cy = int((y_mm + offset_y) * self.mm_to_px)

        # Estimate text width
        char_width = max(font_size_px * 0.6, 8)
        text_width = int(len(text) * char_width)
        text_height = int(font_size_px * 1.3)

        # Add padding
        half_w = text_width // 2 + self.text_padding_px
        half_h = text_height // 2 + self.text_padding_px

        # Compute corners
        x1, y1 = cx - half_w, cy - half_h
        x2, y2 = cx + half_w, cy - half_h
        x3, y3 = cx + half_w, cy + half_h
        x4, y4 = cx - half_w, cy + half_h

        # Clamp to image bounds
        if x1 < 0 or y1 < 0 or x2 >= img_width or y3 >= img_height:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width - 1, x2)
            y2 = max(0, y2)
            x3 = min(img_width - 1, x3)
            y3 = min(img_height - 1, y3)
            x4 = max(0, x4)
            y4 = min(img_height - 1, y4)

        # Validate bbox has positive area
        if x2 <= x1 or y3 <= y1:
            return None

        return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    def to_paddleocr_det_format(
        self,
        annotations: List[TextAnnotation],
        image_path: str,
    ) -> str:
        """
        Convert to PaddleOCR detection training format.

        Format: image_path\t[{"points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
                              "transcription": "text"}]
        """
        label_entries = []
        for ann in annotations:
            label_entries.append({
                "points": ann.bbox,
                "transcription": ann.text,
            })

        return f"{image_path}\t{json.dumps(label_entries, ensure_ascii=False)}"

    def to_paddleocr_rec_format(
        self,
        annotations: List[TextAnnotation],
        image_dir: str,
    ) -> List[str]:
        """
        Convert to PaddleOCR recognition training format.

        Format: crop_image_path\ttext
        """
        lines = []
        for i, ann in enumerate(annotations):
            crop_path = f"{image_dir}/crop_{i:06d}.png"
            lines.append(f"{crop_path}\t{ann.text}")
        return lines

    def to_structured_json(
        self,
        data: SchematicData,
        annotations: List[TextAnnotation],
    ) -> Dict:
        """
        Generate the full structured output JSON.

        Includes texts, components, nets, connections, and functional blocks.
        """
        # Components
        components = []
        for comp in data.components:
            if not comp.reference:
                continue
            # Find bbox from annotations
            bbox = None
            for ann in annotations:
                if ann.component_ref == comp.reference and ann.category == "reference":
                    bbox = ann.bbox
                    break

            components.append({
                "ref": comp.reference,
                "value": comp.value,
                "type": comp.properties.get("_type", "Unknown"),
                "bbox": bbox,
            })

        # Nets
        try:
            nets_dict = KiCadParser().extract_netlist(data)
        except Exception:
            nets_dict = {}

        nets = []
        for name, pins in nets_dict.items():
            nets.append({
                "name": name,
                "connected_to": pins,
            })

        # Texts
        texts = []
        for ann in annotations:
            texts.append({
                "text": ann.text,
                "bbox": ann.bbox,
                "category": ann.category,
            })

        return {
            "texts": texts,
            "components": components,
            "nets": nets,
            "connections": [],  # Populated by post-processor
            "functional_blocks": [],  # Populated by VLM
            "overall_function": "",
        }

    def generate_char_dict(
        self,
        all_annotations: List[TextAnnotation],
        output_path: str,
        base_chars: Optional[str] = None,
    ):
        """
        Generate a character dictionary file for OCR training.

        Includes standard alphanumeric + electronic special characters.

        Args:
            all_annotations: All annotations to extract characters from
            output_path: Path to write char dict file
            base_chars: Optional base character set
        """
        chars = set()

        if base_chars:
            chars.update(base_chars)
        else:
            # Standard alphanumeric
            chars.update("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            chars.update("abcdefghijklmnopqrstuvwxyz")
            chars.update("0123456789")
            chars.update(".-_+/=():;[]{}|\\<>,!?#@&*~`\"'")

        # Extract characters from annotations
        for ann in all_annotations:
            chars.update(ann.text)

        # Add electronics-specific characters
        chars.update("Ωμμ±°℃∞αβγδεθλπσφωΔΣΠ")
        chars.update("kMGTpPnNfFuUhHzZ")

        # Remove non-printable characters
        chars = {c for c in chars if c.isprintable()}

        # Sort and write
        sorted_chars = sorted(chars)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, char in enumerate(sorted_chars):
                f.write(f"{char}\n")

        logger.info(f"Character dictionary: {len(sorted_chars)} chars → {output_path}")
        return sorted_chars

    def visualize_annotations(
        self,
        image_path: str,
        annotations: List[TextAnnotation],
        output_path: str,
    ):
        """
        Draw annotation bboxes on image for quality inspection.

        Args:
            image_path: Path to the rendered schematic image
            annotations: Annotations to visualize
            output_path: Path to save visualization
        """
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            # Color scheme by category
            colors = {
                "reference": "#FF0000",
                "value": "#00AA00",
                "net_label": "#0000FF",
                "pin": "#FF8800",
                "text": "#888888",
            }

            for ann in annotations:
                color = colors.get(ann.category, "#888888")

                # Draw bbox
                points = [(p[0], p[1]) for p in ann.bbox]
                draw.polygon(points, outline=color, fill=None)

                # Draw label
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except (OSError, IOError):
                    font = ImageFont.load_default()

                x, y = ann.bbox[0]
                label = f"{ann.category}: {ann.text[:20]}"
                draw.text((x, y - 14), label, fill=color, font=font)

            img.save(output_path)
            logger.info(f"Visualization saved: {output_path}")

        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    def crop_text_regions(
        self,
        image_path: str,
        annotations: List[TextAnnotation],
        output_dir: str,
        padding: int = 5,
    ) -> List[str]:
        """
        Crop text regions from image for recognition training.

        Args:
            image_path: Path to rendered schematic
            annotations: Text annotations
            output_dir: Directory for cropped images
            padding: Padding around text bbox

        Returns:
            List of paths to cropped images
        """
        from PIL import Image

        img = Image.open(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        crop_paths = []
        for i, ann in enumerate(annotations):
            # Get bbox bounds
            xs = [p[0] for p in ann.bbox]
            ys = [p[1] for p in ann.bbox]
            x1, x2 = max(0, min(xs) - padding), min(img.width, max(xs) + padding)
            y1, y2 = max(0, min(ys) - padding), min(img.height, max(ys) + padding)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop_path = str(output_dir / f"crop_{i:06d}.png")
            crop.save(crop_path)
            crop_paths.append(crop_path)

        return crop_paths


# Import KiCadParser here to avoid circular import
from .kicad_parser import KiCadParser


if __name__ == "__main__":
    import argparse
    from .kicad_parser import KiCadParser

    logging.basicConfig(level=logging.INFO)

    parser_cli = argparse.ArgumentParser(description="Generate annotations")
    parser_cli.add_argument("sch_path", help="Path to .kicad_sch file")
    parser_cli.add_argument("--image-path", help="Path to rendered PNG")
    parser_cli.add_argument("--output-dir", default="data/annotations")
    parser_cli.add_argument("--dpi", type=int, default=300)
    args = parser_cli.parse_args()

    parser = KiCadParser()
    data = parser.parse(args.sch_path)

    gen = AnnotationGenerator(mm_to_px=args.dpi / 25.4)
    annotations = gen.generate_annotations(data, 2000, 1500)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PaddleOCR det format
    det_line = gen.to_paddleocr_det_format(annotations, args.image_path or "test.png")
    (output_dir / "det_label.txt").write_text(det_line, encoding="utf-8")

    # Save structured JSON
    structured = gen.to_structured_json(data, annotations)
    (output_dir / "structured.json").write_text(
        json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Generated {len(annotations)} annotations")
    for ann in annotations[:5]:
        print(f"  [{ann.category}] {ann.text} @ {ann.bbox}")
