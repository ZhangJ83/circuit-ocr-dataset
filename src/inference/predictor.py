"""
Circuit Schematic Predictor
============================
End-to-end prediction pipeline: image → structured output.

Loads fine-tuned PaddleOCR models and runs detection + recognition
on circuit schematic images.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DetectedText:
    """A detected and recognized text region."""
    text: str
    bbox: List[List[int]]  # Quadrilateral [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    confidence: float
    category: str = ""  # reference, value, net_label, pin, text
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self):
        if self.bbox:
            self.center_x = sum(p[0] for p in self.bbox) / 4
            self.center_y = sum(p[1] for p in self.bbox) / 4


@dataclass
class PredictionResult:
    """Complete prediction result for a single image."""
    image_path: str
    image_width: int = 0
    image_height: int = 0
    texts: List[DetectedText] = field(default_factory=list)
    components: List[Dict] = field(default_factory=list)
    nets: Dict[str, List[str]] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "texts": [
                {
                    "text": t.text,
                    "bbox": t.bbox,
                    "confidence": t.confidence,
                    "category": t.category,
                }
                for t in self.texts
            ],
            "components": self.components,
            "nets": self.nets,
            "processing_time_ms": self.processing_time_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class CircuitPredictor:
    """End-to-end predictor for circuit schematics."""

    def __init__(
        self,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        use_paddleocr: bool = True,
        device: str = "gpu",
    ):
        """
        Args:
            det_model_dir: Path to detection model directory
            rec_model_dir: Path to recognition model directory
            use_paddleocr: Use PaddleOCR pipeline (vs. raw model)
            device: "gpu" or "cpu"
        """
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.use_paddleocr = use_paddleocr
        self.device = device
        self._ocr = None

    def _init_paddleocr(self):
        """Initialize PaddleOCR pipeline."""
        if self._ocr is not None:
            return

        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=(self.device == "gpu"),
                det_model_dir=self.det_model_dir,
                rec_model_dir=self.rec_model_dir,
                show_log=False,
            )
            logger.info("PaddleOCR initialized successfully")

        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def predict(
        self,
        image_path: str,
        classify_text: bool = True,
    ) -> PredictionResult:
        """
        Run prediction on a single circuit schematic image.

        Args:
            image_path: Path to input image
            classify_text: Whether to classify text categories

        Returns:
            PredictionResult with detected texts, components, and nets
        """
        import time
        from PIL import Image

        start_time = time.time()

        # Load image to get dimensions
        img = Image.open(image_path)
        result = PredictionResult(
            image_path=image_path,
            image_width=img.width,
            image_height=img.height,
        )

        # Run OCR
        self._init_paddleocr()
        ocr_results = self._ocr.ocr(image_path, cls=True)

        if not ocr_results or not ocr_results[0]:
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        # Parse OCR results
        for line in ocr_results[0]:
            bbox_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = line[1][0]
            confidence = line[1][1]

            if not text.strip():
                continue

            # Convert bbox to integer coordinates
            bbox = [[int(p[0]), int(p[1])] for p in bbox_points]

            detected = DetectedText(
                text=text,
                bbox=bbox,
                confidence=confidence,
            )

            # Classify text category
            if classify_text:
                detected.category = self._classify_text(text, detected)

            result.texts.append(detected)

        # Post-process: aggregate components
        result.components = self._aggregate_components(result.texts)

        result.processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Predicted {len(result.texts)} texts, "
            f"{len(result.components)} components in "
            f"{result.processing_time_ms:.1f}ms"
        )

        return result

    def _classify_text(self, text: str, detected: DetectedText) -> str:
        """
        Classify text into categories based on pattern matching.

        Categories:
        - reference: Component reference (R1, C1, U1, etc.)
        - value: Component value (10k, 100nF, etc.)
        - net_label: Net label (VCC, GND, CLK, etc.)
        - pin: Pin name (PA0, VDD, etc.)
        - text: General text annotation
        """
        import re

        # Reference patterns: letter(s) + number(s)
        if re.match(r'^[A-Z]{1,3}\d+$', text):
            return "reference"

        # Value patterns: number + unit
        if re.match(r'^\d+\.?\d*\s*[kKmMnNpPuUfFhHzZ]', text):
            return "value"
        if re.match(r'^\d+\.?\d*[Ee][+-]?\d+$', text):
            return "value"

        # Net label patterns
        power_nets = {"VCC", "GND", "VDD", "VEE", "VIN", "VOUT", "3V3", "5V",
                      "12V", "3.3V", "1.8V", "VBUS", "AVCC", "AGND", "DGND"}
        if text.upper() in power_nets:
            return "net_label"

        # Signal net labels (all caps, 2-10 chars)
        if re.match(r'^[A-Z][A-Z0-9_]{1,10}$', text) and len(text) <= 12:
            return "net_label"

        # Pin names
        pin_patterns = [
            r'^[PA]\d{1,2}$',     # PA0, PB0, etc.
            r'^[A-Z]{1,3}\d{1,2}$', # Generic pin
            r'^VDD|VCC|GND|VIN$',  # Power pins
        ]
        for pat in pin_patterns:
            if re.match(pat, text):
                return "pin"

        # Default
        return "text"

    def _aggregate_components(self, texts: List[DetectedText]) -> List[Dict]:
        """Aggregate reference and value texts into component objects."""
        import re

        refs = {}
        vals = {}

        for t in texts:
            if t.category == "reference":
                refs[t.text] = t
            elif t.category == "value":
                # Find nearest reference
                vals[t.text] = t

        components = []
        used_vals = set()

        for ref_text, ref_det in refs.items():
            comp = {
                "ref": ref_text,
                "value": "",
                "type": self._guess_component_type(ref_text),
                "bbox": ref_det.bbox,
                "position": {"x": ref_det.center_x, "y": ref_det.center_y},
            }

            # Find closest value text
            best_val = None
            best_dist = float("inf")
            for val_text, val_det in vals.items():
                if val_text in used_vals:
                    continue
                dist = ((ref_det.center_x - val_det.center_x)**2 +
                        (ref_det.center_y - val_det.center_y)**2)**0.5
                if dist < best_dist and dist < 100:  # Within 100 pixels
                    best_dist = dist
                    best_val = (val_text, val_det)

            if best_val:
                comp["value"] = best_val[0]
                used_vals.add(best_val[0])

            components.append(comp)

        return components

    @staticmethod
    def _guess_component_type(ref: str) -> str:
        """Guess component type from reference prefix."""
        prefix = ''.join(c for c in ref if c.isalpha())
        type_map = {
            "R": "Resistor", "C": "Capacitor", "L": "Inductor",
            "D": "Diode", "LED": "LED", "Q": "Transistor",
            "U": "IC", "J": "Connector", "TP": "TestPoint",
            "F": "Fuse", "SW": "Switch", "Y": "Crystal",
            "X": "Misc", "LS": "Buzzer", "RT": "Thermistor",
        }
        return type_map.get(prefix, "Unknown")

    def predict_batch(
        self,
        image_paths: List[str],
        classify_text: bool = True,
    ) -> List[PredictionResult]:
        """Run prediction on multiple images."""
        results = []
        for i, path in enumerate(image_paths):
            try:
                result = self.predict(path, classify_text)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for {path}: {e}")
                results.append(PredictionResult(image_path=path))

            if (i + 1) % 10 == 0:
                logger.info(f"Predicted {i+1}/{len(image_paths)}")

        return results
