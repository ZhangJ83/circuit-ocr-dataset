"""
Evaluator
=========
End-to-end evaluation pipeline for circuit schematic OCR.

Evaluates across multiple dimensions:
- Different circuit complexities
- Different text types
- Different degradation types
- Different circuit types
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from .metrics import (
    TextDetMetrics, TextRecMetrics, ComponentMetrics, NetlistMetrics,
    text_detection_metrics, text_recognition_metrics,
    component_detection_metrics, netlist_metrics,
    compute_composite_score,
)

logger = logging.getLogger(__name__)


class CircuitEvaluator:
    """Evaluate circuit schematic OCR system."""

    def __init__(self, eval_dir: str):
        """
        Args:
            eval_dir: Directory containing evaluation data
        """
        self.eval_dir = Path(eval_dir)

    def evaluate_dataset(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict:
        """
        Run full evaluation on a dataset.

        Args:
            predictions: List of prediction dicts
            ground_truths: List of ground truth dicts

        Returns:
            Evaluation results dict
        """
        # Level 1: Text Detection
        pred_bboxes = [
            [t["bbox"] for t in p.get("texts", [])]
            for p in predictions
        ]
        gt_bboxes = [
            [t["bbox"] for t in g.get("texts", [])]
            for g in ground_truths
        ]
        det_metrics = text_detection_metrics(pred_bboxes, gt_bboxes)

        # Level 2: Text Recognition
        pred_texts = []
        gt_texts = []
        for pred, gt in zip(predictions, ground_truths):
            for t in pred.get("texts", []):
                pred_texts.append(t["text"])
            for t in gt.get("texts", []):
                gt_texts.append(t["text"])

        # Pad to same length
        max_len = max(len(pred_texts), len(gt_texts))
        pred_texts.extend([""] * (max_len - len(pred_texts)))
        gt_texts.extend([""] * (max_len - len(gt_texts)))

        rec_metrics = text_recognition_metrics(pred_texts, gt_texts)

        # Level 3: Component Detection
        comp_metrics = component_detection_metrics(
            [c for p in predictions for c in p.get("components", [])],
            [c for g in ground_truths for c in g.get("components", [])],
        )

        # Level 4: Netlist Extraction
        net_results = []
        for pred, gt in zip(predictions, ground_truths):
            pred_nets = pred.get("nets", {})
            gt_nets = gt.get("nets", {})
            if gt_nets:
                net_results.append(netlist_metrics(pred_nets, gt_nets))

        avg_net = NetlistMetrics(
            component_recall=sum(n.component_recall for n in net_results) / len(net_results) if net_results else 0,
            connection_accuracy=sum(n.connection_accuracy for n in net_results) / len(net_results) if net_results else 0,
            net_name_accuracy=sum(n.net_name_accuracy for n in net_results) / len(net_results) if net_results else 0,
            exact_match=sum(n.exact_match for n in net_results) / len(net_results) if net_results else 0,
        )

        # Composite score
        composite = compute_composite_score(det_metrics, rec_metrics, comp_metrics, avg_net)

        results = {
            "text_detection": {
                "precision": det_metrics.precision,
                "recall": det_metrics.recall,
                "f1": det_metrics.f1,
            },
            "text_recognition": {
                "char_accuracy": rec_metrics.char_accuracy,
                "word_accuracy": rec_metrics.word_accuracy,
                "normalized_edit_distance": rec_metrics.normalized_edit_distance,
            },
            "component_detection": {
                "mAP": comp_metrics.mAP,
                "type_accuracy": comp_metrics.type_accuracy,
                "component_recall": comp_metrics.component_recall,
            },
            "netlist_extraction": {
                "component_recall": avg_net.component_recall,
                "connection_accuracy": avg_net.connection_accuracy,
                "net_name_accuracy": avg_net.net_name_accuracy,
                "exact_match": avg_net.exact_match,
            },
            "composite_score": composite,
            "num_samples": len(predictions),
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"  Text Detection F1: {det_metrics.f1:.4f}")
        logger.info(f"  Text Recognition Word Acc: {rec_metrics.word_accuracy:.4f}")
        logger.info(f"  Component mAP: {comp_metrics.mAP:.4f}")
        logger.info(f"  Netlist Connection Acc: {avg_net.connection_accuracy:.4f}")
        logger.info(f"  Composite Score: {composite:.2f}/100")

        return results

    def evaluate_by_complexity(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict[str, Dict]:
        """Evaluate separately by circuit complexity."""
        complexity_groups = {"simple": [], "medium": [], "complex": []}

        for pred, gt in zip(predictions, ground_truths):
            n_components = len(gt.get("components", []))
            if n_components < 15:
                complexity_groups["simple"].append((pred, gt))
            elif n_components < 40:
                complexity_groups["medium"].append((pred, gt))
            else:
                complexity_groups["complex"].append((pred, gt))

        results = {}
        for complexity, pairs in complexity_groups.items():
            if not pairs:
                continue
            preds, gts = zip(*pairs)
            results[complexity] = self.evaluate_dataset(list(preds), list(gts))

        return results

    def evaluate_by_category(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict[str, Dict]:
        """Evaluate separately by text category."""
        categories = ["reference", "value", "net_label", "pin"]
        results = {}

        for cat in categories:
            pred_texts = []
            gt_texts = []

            for pred, gt in zip(predictions, ground_truths):
                for t in pred.get("texts", []):
                    if t.get("category") == cat:
                        pred_texts.append(t["text"])
                for t in gt.get("texts", []):
                    if t.get("category") == cat:
                        gt_texts.append(t["text"])

            if gt_texts:
                max_len = max(len(pred_texts), len(gt_texts))
                pred_texts.extend([""] * (max_len - len(pred_texts)))
                gt_texts.extend([""] * (max_len - len(gt_texts)))

                results[cat] = {
                    "word_accuracy": text_recognition_metrics(
                        pred_texts, gt_texts
                    ).word_accuracy,
                    "count": len(gt_texts),
                }

        return results
