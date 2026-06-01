"""
Evaluation Report Generator
============================
Generate comprehensive evaluation reports in Markdown format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate evaluation reports."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: Dict,
        model_name: str = "CircuitOCR",
        complexity_results: Optional[Dict] = None,
        category_results: Optional[Dict] = None,
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            results: Main evaluation results
            model_name: Model name for the report
            complexity_results: Results by circuit complexity
            category_results: Results by text category

        Returns:
            Path to generated report file
        """
        report = []
        report.append(f"# {model_name} - Evaluation Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nSamples evaluated: {results.get('num_samples', 'N/A')}")
        report.append(f"\n## Composite Score: **{results.get('composite_score', 0):.2f}/100**")

        # Level 1: Text Detection
        det = results.get("text_detection", {})
        report.append("\n## Level 1: Text Detection")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Precision | {det.get('precision', 0):.4f} |")
        report.append(f"| Recall | {det.get('recall', 0):.4f} |")
        report.append(f"| **F1 Score** | **{det.get('f1', 0):.4f}** |")

        # Level 2: Text Recognition
        rec = results.get("text_recognition", {})
        report.append("\n## Level 2: Text Recognition")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Character Accuracy | {rec.get('char_accuracy', 0):.4f} |")
        report.append(f"| **Word Accuracy** | **{rec.get('word_accuracy', 0):.4f}** |")
        report.append(f"| Normalized Edit Distance | {rec.get('normalized_edit_distance', 0):.4f} |")

        # Level 3: Component Detection
        comp = results.get("component_detection", {})
        report.append("\n## Level 3: Component Detection")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| **mAP@0.5** | **{comp.get('mAP', 0):.4f}** |")
        report.append(f"| Type Accuracy | {comp.get('type_accuracy', 0):.4f} |")
        report.append(f"| Component Recall | {comp.get('component_recall', 0):.4f} |")

        # Level 4: Netlist Extraction
        net = results.get("netlist_extraction", {})
        report.append("\n## Level 4: Netlist Extraction")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Component Recall | {net.get('component_recall', 0):.4f} |")
        report.append(f"| **Connection Accuracy** | **{net.get('connection_accuracy', 0):.4f}** |")
        report.append(f"| Net Name Accuracy | {net.get('net_name_accuracy', 0):.4f} |")
        report.append(f"| Exact Match | {net.get('exact_match', 0):.4f} |")

        # Complexity breakdown
        if complexity_results:
            report.append("\n## Results by Circuit Complexity")
            report.append("| Complexity | Det F1 | Rec Word Acc | Comp mAP | Net Conn | Score |")
            report.append("|------------|--------|--------------|----------|----------|-------|")
            for complexity, r in complexity_results.items():
                report.append(
                    f"| {complexity} | "
                    f"{r.get('text_detection', {}).get('f1', 0):.3f} | "
                    f"{r.get('text_recognition', {}).get('word_accuracy', 0):.3f} | "
                    f"{r.get('component_detection', {}).get('mAP', 0):.3f} | "
                    f"{r.get('netlist_extraction', {}).get('connection_accuracy', 0):.3f} | "
                    f"**{r.get('composite_score', 0):.1f}** |"
                )

        # Category breakdown
        if category_results:
            report.append("\n## Results by Text Category")
            report.append("| Category | Word Accuracy | Count |")
            report.append("|----------|---------------|-------|")
            for cat, r in category_results.items():
                report.append(
                    f"| {cat} | {r.get('word_accuracy', 0):.4f} | {r.get('count', 0)} |"
                )

        # Write report
        report_text = "\n".join(report)
        report_path = self.output_dir / "evaluation_report.md"
        report_path.write_text(report_text, encoding="utf-8")

        # Also save raw results as JSON
        json_path = self.output_dir / "evaluation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Report saved to {report_path}")
        return str(report_path)
