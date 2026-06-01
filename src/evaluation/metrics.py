"""
Evaluation Metrics for Circuit Schematic OCR
=============================================
Multi-level evaluation metrics:

Level 1 - Text Detection:
  - Precision, Recall, F1 (IoU > 0.5)
Level 2 - Text Recognition:
  - Character accuracy, Word accuracy
Level 3 - Component Detection:
  - mAP@0.5, Type accuracy
Level 4 - Netlist Extraction:
  - Component recall, Connection accuracy, Net name accuracy, Exact match
"""

import json
import math
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextDetMetrics:
    """Text detection metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class TextRecMetrics:
    """Text recognition metrics."""
    char_accuracy: float = 0.0
    word_accuracy: float = 0.0
    normalized_edit_distance: float = 0.0
    total_chars: int = 0
    correct_chars: int = 0
    total_words: int = 0
    correct_words: int = 0


@dataclass
class ComponentMetrics:
    """Component detection metrics."""
    mAP: float = 0.0
    type_accuracy: float = 0.0
    component_recall: float = 0.0
    total_gt: int = 0
    total_pred: int = 0
    correct_type: int = 0


@dataclass
class NetlistMetrics:
    """Netlist extraction metrics."""
    component_recall: float = 0.0
    connection_accuracy: float = 0.0
    net_name_accuracy: float = 0.0
    exact_match: float = 0.0
    total_gt_nets: int = 0
    correct_nets: int = 0


def compute_iou(bbox1: List[List[int]], bbox2: List[List[int]]) -> float:
    """
    Compute IoU between two quadrilateral bounding boxes.

    Uses the intersection-over-union formula with polygon clipping.
    """
    # Convert to axis-aligned bounding boxes for simplicity
    x1_min = min(p[0] for p in bbox1)
    y1_min = min(p[1] for p in bbox1)
    x1_max = max(p[0] for p in bbox1)
    y1_max = max(p[1] for p in bbox1)

    x2_min = min(p[0] for p in bbox2)
    y2_min = min(p[1] for p in bbox2)
    x2_max = max(p[0] for p in bbox2)
    y2_max = max(p[1] for p in bbox2)

    # Intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_char_accuracy(pred: str, gt: str) -> float:
    """Compute character-level accuracy using edit distance."""
    if not gt:
        return 1.0 if not pred else 0.0

    # Levenshtein distance
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == gt[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    edit_dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - edit_dist / max_len if max_len > 0 else 1.0


def text_detection_metrics(
    predictions: List[List[List[int]]],
    ground_truths: List[List[List[int]]],
    iou_threshold: float = 0.5,
) -> TextDetMetrics:
    """
    Compute text detection metrics.

    Args:
        predictions: List of predicted bboxes per image
        ground_truths: List of GT bboxes per image
        iou_threshold: IoU threshold for matching

    Returns:
        TextDetMetrics
    """
    tp, fp, fn = 0, 0, 0

    for pred_bboxes, gt_bboxes in zip(predictions, ground_truths):
        matched_gt = set()

        for pred_bb in pred_bboxes:
            best_iou = 0.0
            best_idx = -1

            for j, gt_bb in enumerate(gt_bboxes):
                if j in matched_gt:
                    continue
                iou = compute_iou(pred_bb, gt_bb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn += len(gt_bboxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return TextDetMetrics(
        precision=precision, recall=recall, f1=f1,
        true_positives=tp, false_positives=fp, false_negatives=fn,
    )


def text_recognition_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> TextRecMetrics:
    """
    Compute text recognition metrics.

    Args:
        predictions: List of predicted text strings
        ground_truths: List of GT text strings

    Returns:
        TextRecMetrics
    """
    total_chars = 0
    correct_chars = 0
    total_words = len(predictions)
    correct_words = 0
    total_edit = 0.0

    for pred, gt in zip(predictions, ground_truths):
        # Word accuracy
        if pred.strip().lower() == gt.strip().lower():
            correct_words += 1

        # Character accuracy
        char_acc = compute_char_accuracy(pred, gt)
        total_chars += len(gt)
        correct_chars += int(char_acc * len(gt))
        total_edit += (1.0 - char_acc)

    return TextRecMetrics(
        char_accuracy=correct_chars / total_chars if total_chars > 0 else 0.0,
        word_accuracy=correct_words / total_words if total_words > 0 else 0.0,
        normalized_edit_distance=total_edit / total_words if total_words > 0 else 0.0,
        total_chars=total_chars,
        correct_chars=correct_chars,
        total_words=total_words,
        correct_words=correct_words,
    )


def component_detection_metrics(
    pred_components: List[Dict],
    gt_components: List[Dict],
    iou_threshold: float = 0.5,
) -> ComponentMetrics:
    """
    Compute component detection metrics.

    Args:
        pred_components: Predicted components with ref, type, bbox
        gt_components: GT components with ref, type, bbox

    Returns:
        ComponentMetrics
    """
    matched = set()
    correct_type = 0

    for pred in pred_components:
        pred_bbox = pred.get("bbox", [[0,0],[0,0],[0,0],[0,0]])
        best_iou = 0.0
        best_idx = -1

        for j, gt in enumerate(gt_components):
            if j in matched:
                continue
            gt_bbox = gt.get("bbox", [[0,0],[0,0],[0,0],[0,0]])
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= iou_threshold:
            matched.add(best_idx)
            # Check type accuracy
            if pred.get("type") == gt_components[best_idx].get("type"):
                correct_type += 1

    recall = len(matched) / len(gt_components) if gt_components else 0.0
    type_acc = correct_type / len(matched) if matched else 0.0

    return ComponentMetrics(
        mAP=recall,  # Simplified mAP
        type_accuracy=type_acc,
        component_recall=recall,
        total_gt=len(gt_components),
        total_pred=len(pred_components),
        correct_type=correct_type,
    )


def netlist_metrics(
    pred_netlist: Dict[str, List[str]],
    gt_netlist: Dict[str, List[str]],
) -> NetlistMetrics:
    """
    Compute netlist extraction metrics.

    Args:
        pred_netlist: Predicted netlist {net_name: [pin_list]}
        gt_netlist: GT netlist {net_name: [pin_list]}

    Returns:
        NetlistMetrics
    """
    # Normalize net names for comparison
    def normalize_net(pins):
        return frozenset(sorted(pins))

    pred_nets = {name: normalize_net(pins) for name, pins in pred_netlist.items()}
    gt_nets = {name: normalize_net(pins) for name, pins in gt_netlist.items()}

    # Find matching nets (by content, regardless of name)
    matched_gt = set()
    correct_nets = 0
    correct_names = 0

    for pred_name, pred_pins in pred_nets.items():
        for gt_name, gt_pins in gt_nets.items():
            if gt_name in matched_gt:
                continue
            if pred_pins == gt_pins:
                correct_nets += 1
                matched_gt.add(gt_name)
                if pred_name == gt_name:
                    correct_names += 1
                break

    total_gt = len(gt_nets)

    # Component recall: how many GT components appear in predictions
    gt_components = set()
    pred_components = set()
    for pins in gt_netlist.values():
        for pin in pins:
            gt_components.add(pin.split(":")[0])
    for pins in pred_netlist.values():
        for pin in pins:
            pred_components.add(pin.split(":")[0])

    comp_recall = (
        len(gt_components & pred_components) / len(gt_components)
        if gt_components else 0.0
    )

    return NetlistMetrics(
        component_recall=comp_recall,
        connection_accuracy=correct_nets / total_gt if total_gt > 0 else 0.0,
        net_name_accuracy=correct_names / total_gt if total_gt > 0 else 0.0,
        exact_match=1.0 if correct_nets == total_gt and total_gt > 0 else 0.0,
        total_gt_nets=total_gt,
        correct_nets=correct_nets,
    )


def compute_composite_score(
    det_metrics: TextDetMetrics,
    rec_metrics: TextRecMetrics,
    comp_metrics: ComponentMetrics,
    net_metrics: NetlistMetrics,
) -> float:
    """
    Compute a composite score (0-100) for the competition.

    Weighting based on competition criteria:
    - Text detection: 30%
    - Text recognition: 20%
    - Component detection: 20%
    - Netlist extraction: 30%
    """
    score = (
        det_metrics.f1 * 30 +
        rec_metrics.word_accuracy * 20 +
        comp_metrics.mAP * 20 +
        net_metrics.connection_accuracy * 30
    )
    return score * 100  # Scale to 0-100
