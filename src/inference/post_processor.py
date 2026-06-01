"""
Post-Processor
==============
Post-process OCR results to:
1. Associate text with components (spatial matching)
2. Classify text types
3. Aggregate component instances
4. Infer basic connectivity
"""

import math
import logging
from typing import List, Dict, Tuple, Optional
from .predictor import DetectedText, PredictionResult

logger = logging.getLogger(__name__)


class PostProcessor:
    """Post-process OCR results for circuit schematic understanding."""

    def __init__(self, distance_threshold: float = 50.0):
        """
        Args:
            distance_threshold: Max pixel distance for text-component association
        """
        self.distance_threshold = distance_threshold

    def process(self, result: PredictionResult) -> PredictionResult:
        """
        Apply all post-processing steps to a prediction result.

        Args:
            result: Raw prediction result

        Returns:
            Enhanced prediction result
        """
        # Step 1: Refine text classification
        self._refine_classifications(result)

        # Step 2: Associate reference and value texts
        self._associate_ref_value(result)

        # Step 3: Group nearby net labels
        self._group_net_labels(result)

        # Step 4: Infer component types
        self._infer_component_types(result)

        return result

    def _refine_classifications(self, result: PredictionResult):
        """Refine text classifications using spatial context."""
        for text in result.texts:
            if text.category != "text":
                continue

            # Check if near a component reference
            nearest_ref = self._find_nearest_category(
                text, result.texts, "reference"
            )
            if nearest_ref:
                dist = self._distance(text, nearest_ref)
                if dist < self.distance_threshold:
                    # Likely a value for this component
                    text.category = "value"
                    continue

    def _associate_ref_value(self, result: PredictionResult):
        """Associate reference designators with their values."""
        refs = [t for t in result.texts if t.category == "reference"]
        vals = [t for t in result.texts if t.category == "value"]

        for val in vals:
            best_ref = None
            best_dist = float("inf")

            for ref in refs:
                dist = self._distance(val, ref)
                if dist < best_dist and dist < self.distance_threshold:
                    best_dist = dist
                    best_ref = ref

            if best_ref:
                # Update component in result
                for comp in result.components:
                    if comp.get("ref") == best_ref.text and not comp.get("value"):
                        comp["value"] = val.text
                        break

    def _group_net_labels(self, result: PredictionResult):
        """Group identical net labels that appear in different locations."""
        net_labels = [t for t in result.texts if t.category == "net_label"]

        nets = {}
        for label in net_labels:
            name = label.text.upper()
            if name not in nets:
                nets[name] = []
            nets[name].append({
                "position": (label.center_x, label.center_y),
                "bbox": label.bbox,
            })

        result.nets = {
            name: [f"pos:{p['position'][0]:.0f},{p['position'][1]:.0f}"
                   for p in positions]
            for name, positions in nets.items()
        }

    def _infer_component_types(self, result: PredictionResult):
        """Infer component types from reference prefixes."""
        for comp in result.components:
            if comp.get("type") == "Unknown" or not comp.get("type"):
                ref = comp.get("ref", "")
                prefix = ''.join(c for c in ref if c.isalpha())
                type_map = {
                    "R": "Resistor", "C": "Capacitor", "L": "Inductor",
                    "D": "Diode", "LED": "LED", "Q": "Transistor",
                    "U": "IC", "J": "Connector", "F": "Fuse",
                    "SW": "Switch", "Y": "Crystal", "TP": "TestPoint",
                }
                comp["type"] = type_map.get(prefix, "Unknown")

    def _find_nearest_category(
        self,
        text: DetectedText,
        all_texts: List[DetectedText],
        category: str,
    ) -> Optional[DetectedText]:
        """Find nearest text with given category."""
        best = None
        best_dist = float("inf")

        for other in all_texts:
            if other is text or other.category != category:
                continue
            dist = self._distance(text, other)
            if dist < best_dist:
                best_dist = dist
                best = other

        return best

    @staticmethod
    def _distance(a: DetectedText, b: DetectedText) -> float:
        """Euclidean distance between two text centers."""
        return math.sqrt(
            (a.center_x - b.center_x)**2 +
            (a.center_y - b.center_y)**2
        )
