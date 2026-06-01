"""Tests for KiCad parser module."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.data_pipeline.kicad_parser import KiCadParser, Point, SchematicData


class TestPoint:
    def test_creation(self):
        p = Point(1.0, 2.0)
        assert p.x == 1.0
        assert p.y == 2.0

    def test_equality(self):
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 2.0)
        assert p1 == p2

    def test_near_equality(self):
        p1 = Point(1.0, 2.0)
        p2 = Point(1.005, 2.005)
        assert p1 == p2  # Within tolerance

    def test_inequality(self):
        p1 = Point(1.0, 2.0)
        p2 = Point(3.0, 4.0)
        assert p1 != p2

    def test_hash(self):
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 2.0)
        assert hash(p1) == hash(p2)

    def test_to_tuple(self):
        p = Point(1.123, 2.456)
        assert p.to_tuple() == (1.123, 2.456)


class TestKiCadParser:
    def setup_method(self):
        self.parser = KiCadParser()

    def test_rotate_point_zero(self):
        x, y = KiCadParser._rotate_point(1.0, 0.0, 0)
        assert abs(x - 1.0) < 0.001
        assert abs(y - 0.0) < 0.001

    def test_rotate_point_90(self):
        x, y = KiCadParser._rotate_point(1.0, 0.0, 90)
        assert abs(x - 0.0) < 0.001
        assert abs(y - 1.0) < 0.001

    def test_rotate_point_180(self):
        x, y = KiCadParser._rotate_point(1.0, 0.0, 180)
        assert abs(x - (-1.0)) < 0.001
        assert abs(y - 0.0) < 0.001

    def test_rotate_point_270(self):
        x, y = KiCadParser._rotate_point(1.0, 0.0, 270)
        assert abs(x - 0.0) < 0.001
        assert abs(y - (-1.0)) < 0.001

    def test_parse_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/file.kicad_sch")

    def test_parse_manual_fallback(self):
        """Test the manual parser fallback with a minimal .kicad_sch content."""
        content = """(kicad_sch (version 20230121)
  (wire (pts (xy 10 20) (xy 30 40)))
  (wire (pts (xy 30 40) (xy 50 60)))
  (label "VCC" (at 10 20 0))
  (label "GND" (at 50 60 0))
)
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".kicad_sch", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()

            data = self.parser._parse_manual(Path(f.name))

            assert len(data.wires) == 2
            assert len(data.labels) == 2
            assert data.labels[0].name == "VCC"
            assert data.labels[1].name == "GND"

        import os
        os.unlink(f.name)


class TestNetlistExtraction:
    def test_simple_netlist(self):
        """Test netlist extraction with a simple connected circuit."""
        parser = KiCadParser()
        data = SchematicData(file_path="test")

        # Create two components with pins at the same location
        from src.data_pipeline.kicad_parser import (
            ComponentInstance, PinInfo, Wire
        )

        comp1 = ComponentInstance(
            reference="R1", value="10k", lib_name="R",
            position=Point(10, 10),
        )
        comp1.pins = [
            PinInfo(pin_number="1", pin_name="1",
                    relative_pos=Point(0, -2.54),
                    absolute_pos=Point(10, 7.46)),
            PinInfo(pin_number="2", pin_name="2",
                    relative_pos=Point(0, 2.54),
                    absolute_pos=Point(10, 12.54)),
        ]

        comp2 = ComponentInstance(
            reference="C1", value="100nF", lib_name="C",
            position=Point(30, 10),
        )
        comp2.pins = [
            PinInfo(pin_number="1", pin_name="1",
                    relative_pos=Point(0, -2.54),
                    absolute_pos=Point(30, 7.46)),
            PinInfo(pin_number="2", pin_name="2",
                    relative_pos=Point(0, 2.54),
                    absolute_pos=Point(30, 12.54)),
        ]

        data.components = [comp1, comp2]

        # Wire connecting R1:2 to C1:1
        data.wires = [
            Wire(start=Point(10, 12.54), end=Point(30, 7.46)),
        ]

        nets = parser.extract_netlist(data)
        # Should have at least one net
        assert len(nets) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
