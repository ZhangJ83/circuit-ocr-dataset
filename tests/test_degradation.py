"""Tests for degradation module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from PIL import Image

from src.data_pipeline.degradation import SchematicDegradation


class TestDegradation:
    def setup_method(self):
        # Create a test image
        self.test_image = Image.new("RGB", (200, 200), "white")

    def test_paper_aging(self):
        result = SchematicDegradation.paper_aging(self.test_image, severity=0.5, seed=42)
        assert result.size == self.test_image.size
        assert result.mode == "RGB"

    def test_scan_noise(self):
        result = SchematicDegradation.scan_noise(self.test_image, severity=0.5, seed=42)
        assert result.size == self.test_image.size

    def test_perspective_distortion(self):
        result = SchematicDegradation.perspective_distortion(self.test_image, severity=0.3, seed=42)
        assert result.size == self.test_image.size

    def test_handwriting_overlay(self):
        result = SchematicDegradation.handwriting_overlay(self.test_image, severity=0.3, seed=42)
        assert result.size == self.test_image.size

    def test_low_resolution(self):
        result = SchematicDegradation.low_resolution(self.test_image, severity=0.5, seed=42)
        assert result.size == self.test_image.size

    def test_random_degradation(self):
        result, applied = SchematicDegradation.apply_random_degradation(
            self.test_image, severity_range=(0.3, 0.7), num_degradations=2, seed=42
        )
        assert result.size == self.test_image.size
        assert len(applied) == 2
        for deg_type in applied:
            assert deg_type in SchematicDegradation.TYPES

    def test_severity_zero(self):
        result = SchematicDegradation.paper_aging(self.test_image, severity=0.0)
        # With severity 0, image should be nearly unchanged
        np_original = np.array(self.test_image)
        np_result = np.array(result)
        diff = np.abs(np_original.astype(float) - np_result.astype(float)).mean()
        assert diff < 5  # Very small difference

    def test_severity_one(self):
        result = SchematicDegradation.paper_aging(self.test_image, severity=1.0)
        np_original = np.array(self.test_image)
        np_result = np.array(result)
        diff = np.abs(np_original.astype(float) - np_result.astype(float)).mean()
        assert diff > 0  # Should have noticeable difference

    def test_generate_variants(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save test image
            img_path = os.path.join(tmpdir, "test.png")
            self.test_image.save(img_path)

            out_dir = os.path.join(tmpdir, "degraded")
            variants = SchematicDegradation.generate_degraded_variants(
                img_path, out_dir, num_variants=3
            )

            assert len(variants) == 3
            for v in variants:
                assert Path(v["path"]).exists()
                assert len(v["degradations"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
