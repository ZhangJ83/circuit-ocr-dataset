"""
Netlist Extractor
=================
Extract SPICE-compatible netlist from OCR results.

Uses spatial analysis and pattern matching to infer connectivity
from detected text positions and categories.

This is the highest-value output: converting an image to a usable netlist.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from .predictor import PredictionResult, DetectedText

logger = logging.getLogger(__name__)


@dataclass
class NetNode:
    """A node in the circuit net."""
    name: str
    pins: List[str] = field(default_factory=list)  # "REF:PIN" strings


@dataclass
class Component:
    """A circuit component with pins."""
    ref: str
    value: str
    component_type: str
    pins: List[str] = field(default_factory=list)
    position: Tuple[float, float] = (0, 0)


class NetlistExtractor:
    """Extract netlist from OCR prediction results."""

    # Pin mapping for common components
    DEFAULT_PINS = {
        "Resistor": ["1", "2"],
        "Capacitor": ["1", "2"],
        "Inductor": ["1", "2"],
        "Diode": ["A", "K"],
        "LED": ["A", "K"],
        "Transistor": ["B", "C", "E"],
        "IC": ["1", "2", "3", "4"],
        "Connector": ["1", "2"],
        "Fuse": ["1", "2"],
        "Switch": ["1", "2"],
        "Crystal": ["1", "2"],
        "TestPoint": ["1"],
    }

    # SPICE model mapping
    SPICE_MODELS = {
        "Resistor": "R",
        "Capacitor": "C",
        "Inductor": "L",
        "Diode": "D",
        "Transistor_NPN": "Q",
        "Transistor_PNP": "Q",
        "IC": "X",
        "VoltageSource": "V",
        "CurrentSource": "I",
    }

    def __init__(self, proximity_threshold: float = 80.0):
        """
        Args:
            proximity_threshold: Max pixel distance for pin-to-net association
        """
        self.proximity_threshold = proximity_threshold

    def extract(self, result: PredictionResult) -> Dict:
        """
        Extract netlist from prediction result.

        Args:
            result: OCR prediction result

        Returns:
            Netlist dict with components, nets, and SPICE netlist text
        """
        # Step 1: Build component list
        components = self._build_components(result)

        # Step 2: Identify power nets
        power_nets = self._identify_power_nets(result)

        # Step 3: Infer connectivity from spatial proximity
        nets = self._infer_connectivity(result, components)

        # Step 4: Generate SPICE netlist
        spice_netlist = self._generate_spice_netlist(components, nets, power_nets)

        return {
            "components": [
                {
                    "ref": c.ref,
                    "value": c.value,
                    "type": c.component_type,
                    "pins": c.pins,
                    "position": list(c.position),
                }
                for c in components
            ],
            "nets": {name: node.pins for name, node in nets.items()},
            "power_nets": power_nets,
            "spice_netlist": spice_netlist,
            "summary": {
                "total_components": len(components),
                "total_nets": len(nets),
                "power_nets": len(power_nets),
            },
        }

    def _build_components(self, result: PredictionResult) -> List[Component]:
        """Build component objects from OCR results."""
        components = []

        for comp_data in result.components:
            comp_type = comp_data.get("type", "Unknown")
            pins = self.DEFAULT_PINS.get(comp_type, ["1", "2"])

            # Add power pins for ICs
            if comp_type == "IC":
                ref = comp_data.get("ref", "")
                if ref.startswith("U"):
                    pins = ["VDD", "GND", "1", "2", "3", "4"]

            pos = comp_data.get("position", {})
            components.append(Component(
                ref=comp_data.get("ref", ""),
                value=comp_data.get("value", ""),
                component_type=comp_type,
                pins=pins,
                position=(pos.get("x", 0), pos.get("y", 0)),
            ))

        return components

    def _identify_power_nets(self, result: PredictionResult) -> Dict[str, List[str]]:
        """Identify power nets from net labels."""
        power_keywords = {
            "VCC", "VDD", "VIN", "VOUT", "VBUS", "3V3", "5V", "12V",
            "GND", "VEE", "AGND", "DGND", "PGND",
        }

        power_nets = {}
        for text in result.texts:
            if text.category == "net_label" and text.text.upper() in power_keywords:
                name = text.text.upper()
                if name not in power_nets:
                    power_nets[name] = []
                power_nets[name].append(f"pos:{text.center_x:.0f},{text.center_y:.0f}")

        return power_nets

    def _infer_connectivity(
        self,
        result: PredictionResult,
        components: List[Component],
    ) -> Dict[str, NetNode]:
        """Infer connectivity using spatial proximity."""
        nets: Dict[str, NetNode] = {}
        net_counter = 0

        # Create pin position estimates for each component
        pin_positions = {}
        for comp in components:
            cx, cy = comp.position
            n_pins = len(comp.pins)

            for i, pin_name in enumerate(comp.pins):
                # Estimate pin position (spread vertically)
                offset_y = (i - n_pins / 2) * 10
                pin_pos = (cx, cy + offset_y)
                pin_key = f"{comp.ref}:{pin_name}"
                pin_positions[pin_key] = pin_pos

        # Group nearby pins into nets
        assigned = set()
        for pin1, pos1 in pin_positions.items():
            if pin1 in assigned:
                continue

            net_name = f"NET_{net_counter}"
            net_counter += 1
            net = NetNode(name=net_name, pins=[pin1])
            assigned.add(pin1)

            # Find nearby pins
            for pin2, pos2 in pin_positions.items():
                if pin2 in assigned:
                    continue
                dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                if dist < self.proximity_threshold:
                    net.pins.append(pin2)
                    assigned.add(pin2)

            # Check if any net label is nearby
            for text in result.texts:
                if text.category == "net_label":
                    dist = ((pos1[0] - text.center_x)**2 +
                            (pos1[1] - text.center_y)**2)**0.5
                    if dist < self.proximity_threshold:
                        net_name = text.text.upper()
                        net.name = net_name
                        break

            nets[net.name] = net

        return nets

    def _generate_spice_netlist(
        self,
        components: List[Component],
        nets: Dict[str, NetNode],
        power_nets: Dict[str, List[str]],
    ) -> str:
        """Generate SPICE-compatible netlist text."""
        lines = []
        lines.append("* Circuit Schematic OCR - Auto-generated Netlist")
        lines.append("* Generated by CircuitOCR")
        lines.append("")

        # Component cards
        for comp in components:
            spice_prefix = self.SPICE_MODELS.get(comp.component_type, "X")
            ref = comp.ref

            # Replace prefix with SPICE prefix if needed
            if not ref.startswith(spice_prefix):
                ref = spice_prefix + ref

            # Find net connections for this component's pins
            node_names = []
            for pin in comp.pins:
                pin_key = f"{comp.ref}:{pin}"
                found_net = None
                for net_name, net in nets.items():
                    if pin_key in net.pins:
                        found_net = net_name
                        break
                node_names.append(found_net or f"NC_{comp.ref}_{pin}")

            # Format SPICE line
            nodes_str = " ".join(node_names)
            value_str = comp.value if comp.value else ""

            if comp.component_type == "Resistor":
                lines.append(f"{ref} {nodes_str} {value_str}")
            elif comp.component_type == "Capacitor":
                lines.append(f"{ref} {nodes_str} {value_str}")
            elif comp.component_type == "Inductor":
                lines.append(f"{ref} {nodes_str} {value_str}")
            elif comp.component_type == "Diode":
                lines.append(f"{ref} {nodes_str} {comp.value or 'D1N4148'}")
            elif comp.component_type == "IC":
                lines.append(f"{ref} {nodes_str} {comp.value or 'GENERIC_IC'}")
            else:
                lines.append(f"{ref} {nodes_str} {value_str}")

        # Add power sources
        lines.append("")
        lines.append("* Power Sources")
        if "VCC" in power_nets or "5V" in power_nets:
            lines.append("V_VCC VCC 0 DC 5V")
        if "3V3" in power_nets:
            lines.append("V_3V3 3V3 0 DC 3.3V")
        if "12V" in power_nets:
            lines.append("V_12V 12V 0 DC 12V")

        lines.append("")
        lines.append(".end")

        return "\n".join(lines)
