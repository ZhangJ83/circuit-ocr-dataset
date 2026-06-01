"""Tests for netlist extractor module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.inference.predictor import PredictionResult, DetectedText
from src.inference.post_processor import PostProcessor
from src.inference.netlist_extractor import NetlistExtractor


class TestPostProcessor:
    def setup_method(self):
        self.processor = PostProcessor(distance_threshold=50.0)

    def test_classify_reference(self):
        from src.inference.predictor import CircuitPredictor
        predictor = CircuitPredictor(use_paddleocr=False)
        assert predictor._classify_text("R1", None) == "reference"
        assert predictor._classify_text("C10", None) == "reference"
        assert predictor._classify_text("U1", None) == "reference"
        assert predictor._classify_text("LED1", None) == "reference"

    def test_classify_value(self):
        from src.inference.predictor import CircuitPredictor
        predictor = CircuitPredictor(use_paddleocr=False)
        assert predictor._classify_text("10k", None) == "value"
        assert predictor._classify_text("100nF", None) == "value"
        assert predictor._classify_text("4.7uF", None) == "value"

    def test_classify_net_label(self):
        from src.inference.predictor import CircuitPredictor
        predictor = CircuitPredictor(use_paddleocr=False)
        assert predictor._classify_text("VCC", None) == "net_label"
        assert predictor._classify_text("GND", None) == "net_label"
        assert predictor._classify_text("VIN", None) == "net_label"


class TestNetlistExtractor:
    def setup_method(self):
        self.extractor = NetlistExtractor(proximity_threshold=80.0)

    def test_build_components(self):
        result = PredictionResult(image_path="test.png")
        result.components = [
            {"ref": "R1", "value": "10k", "type": "Resistor",
             "position": {"x": 100, "y": 100}},
            {"ref": "C1", "value": "100nF", "type": "Capacitor",
             "position": {"x": 200, "y": 100}},
        ]

        components = self.extractor._build_components(result)
        assert len(components) == 2
        assert components[0].ref == "R1"
        assert components[0].pins == ["1", "2"]

    def test_identify_power_nets(self):
        result = PredictionResult(image_path="test.png")
        result.texts = [
            DetectedText(text="VCC", bbox=[[0,0],[50,0],[50,20],[0,20]],
                        confidence=0.99, category="net_label"),
            DetectedText(text="GND", bbox=[[100,0],[150,0],[150,20],[100,20]],
                        confidence=0.99, category="net_label"),
        ]

        power_nets = self.extractor._identify_power_nets(result)
        assert "VCC" in power_nets
        assert "GND" in power_nets

    def test_spice_generation(self):
        from src.inference.netlist_extractor import Component, NetNode

        components = [
            Component(ref="R1", value="10k", component_type="Resistor",
                     pins=["1", "2"], position=(100, 100)),
            Component(ref="C1", value="100nF", component_type="Capacitor",
                     pins=["1", "2"], position=(200, 100)),
        ]

        nets = {
            "VCC": NetNode(name="VCC", pins=["R1:1"]),
            "NET_1": NetNode(name="NET_1", pins=["R1:2", "C1:1"]),
            "GND": NetNode(name="GND", pins=["C1:2"]),
        }

        power_nets = {"VCC": ["pos:100,75"], "GND": ["pos:200,125"]}

        spice = self.extractor._generate_spice_netlist(components, nets, power_nets)

        assert "R1" in spice
        assert "C1" in spice
        assert "VCC" in spice
        assert "GND" in spice
        assert ".end" in spice


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
