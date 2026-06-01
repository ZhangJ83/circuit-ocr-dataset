"""
Synthetic Schematic Generator
=============================
Generate synthetic circuit schematics for training data augmentation.

Features:
- 30+ component symbol types (R, C, L, D, LED, Q, U, J, Y, etc.)
- Random circuit topologies (series, parallel, bridge, differential, bus)
- Configurable complexity (5-100 components)
- Automatic netlist generation
- Focus on digital/mixed-signal circuits (differentiation strategy A)

Key innovation: ZERO-cost data generation from component library templates.
"""

import random
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ComponentTemplate:
    """Template for a component type."""
    prefix: str               # e.g., "R", "C", "U"
    name: str                 # e.g., "Resistor", "Capacitor"
    pin_names: List[str]      # Pin names, e.g., ["1", "2"] for R
    values: List[str]         # Common values, e.g., ["10k", "100k", "1M"]
    width_mm: float           # Symbol width in mm
    height_mm: float          # Symbol height in mm


@dataclass
class PlacedComponent:
    """A component placed on the synthetic schematic."""
    template: ComponentTemplate
    reference: str
    value: str
    x: float                  # Position in mm
    y: float
    rotation: float = 0.0     # 0, 90, 180, 270
    pin_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class SyntheticSchematicGenerator:
    """Generate synthetic circuit schematics."""

    # Component library
    COMPONENT_TEMPLATES = {
        "R": ComponentTemplate("R", "Resistor", ["1", "2"],
                              ["100", "220", "330", "470", "1k", "2.2k", "4.7k",
                               "10k", "22k", "47k", "100k", "1M"], 4, 2),
        "C": ComponentTemplate("C", "Capacitor", ["1", "2"],
                              ["10pF", "22pF", "100pF", "1nF", "10nF", "100nF",
                               "1uF", "10uF", "100uF"], 4, 2),
        "L": ComponentTemplate("L", "Inductor", ["1", "2"],
                              ["1uH", "10uH", "100uH", "1mH", "10mH"], 4, 2),
        "D": ComponentTemplate("D", "Diode", ["A", "K"],
                              ["1N4148", "1N4007", "BAT54", "BZX84"], 3, 3),
        "LED": ComponentTemplate("LED", "LED", ["A", "K"],
                                ["Red", "Green", "Blue", "White", "Yellow"], 3, 3),
        "Q_NPN": ComponentTemplate("Q", "NPN_Transistor", ["B", "C", "E"],
                                   ["2N2222", "BC547", "S8050"], 4, 4),
        "Q_PNP": ComponentTemplate("Q", "PNP_Transistor", ["B", "C", "E"],
                                   ["2N2907", "BC557", "S8550"], 4, 4),
        "Q_NMOS": ComponentTemplate("Q", "NMOS", ["G", "D", "S"],
                                    ["2N7002", "AO3400", "IRLML6344"], 4, 4),
        "Q_PMOS": ComponentTemplate("Q", "PMOS", ["G", "D", "S"],
                                    ["AO3401", "IRLML6402"], 4, 4),
        "U_MCU": ComponentTemplate("U", "MCU", [
            "VDD", "GND", "PA0", "PA1", "PA2", "PA3", "PA4", "PA5",
            "PB0", "PB1", "PB2", "PB3", "PB4", "PB5",
            "NRST", "BOOT0"
        ], ["STM32F103C8", "STM32F407VG", "ESP32-WROOM", "RP2040",
            "ATmega328P", "GD32F303"], 12, 20),
        "U_OA": ComponentTemplate("U", "OpAmp", ["V+", "V-", "VCC", "VEE", "OUT"],
                                 ["LM358", "LM324", "TL072", "OPA2134",
                                  "MCP6002"], 6, 6),
        "U_REG": ComponentTemplate("U", "VoltageRegulator", ["IN", "GND", "OUT"],
                                   ["LM7805", "LM7812", "AMS1117-3.3",
                                    "LM1117-1.8"], 6, 6),
        "U_COMP": ComponentTemplate("U", "Comparator", ["V+", "V-", "VCC", "VEE", "OUT"],
                                    ["LM393", "LM339", "TLV3701"], 6, 6),
        "U_TIMER": ComponentTemplate("U", "Timer555", [
            "GND", "TRIG", "OUT", "RESET", "CTRL", "THRES", "DISCH", "VCC"
        ], ["NE555", "LMC555", "TLC555"], 8, 8),
        "U_LOGIC": ComponentTemplate("U", "LogicGate", ["A", "B", "VCC", "GND", "Y"],
                                     ["74HC00", "74HC04", "74HC08", "74HC32",
                                      "74HC74", "74HC138", "74HC595"], 6, 6),
        "U_ADC": ComponentTemplate("U", "ADC", [
            "VDD", "GND", "AIN0", "AIN1", "AIN2", "AIN3",
            "SCLK", "MISO", "MOSI", "CS"
        ], ["ADS1115", "MCP3008", "ADC0804"], 10, 10),
        "U_DAC": ComponentTemplate("U", "DAC", [
            "VDD", "GND", "VOUT", "SCLK", "DIN", "CS"
        ], ["MCP4725", "DAC0832"], 8, 8),
        "U_LDO": ComponentTemplate("U", "LDO", ["IN", "GND", "OUT", "EN"],
                                   ["AMS1117-3.3", "MIC5219-3.3", "RT9013-33"], 5, 5),
        "U_DRV": ComponentTemplate("U", "GateDriver", ["IN", "VCC", "GND", "OUT"],
                                   ["IR2104", "UCC27211", "TC4427"], 6, 6),
        "J_HDR": ComponentTemplate("J", "Header", ["1", "2", "3", "4"],
                                   ["2x5", "2x10", "1x4", "1x8"], 8, 4),
        "J_USB": ComponentTemplate("J", "USB_Connector", ["VBUS", "D-", "D+", "GND", "SHIELD"],
                                   ["USB-C", "Micro-USB", "USB-A"], 8, 6),
        "J_CONN": ComponentTemplate("J", "ScrewTerminal", ["1", "2"],
                                    ["2P", "3P", "4P"], 6, 3),
        "TP": ComponentTemplate("TP", "TestPoint", ["1"],
                                ["TP"], 2, 2),
        "F": ComponentTemplate("F", "Fuse", ["1", "2"],
                               ["500mA", "1A", "2A", "5A"], 4, 2),
        "SW": ComponentTemplate("SW", "Switch", ["1", "2"],
                                ["SPST", "SPDT", "Push"], 4, 3),
        "Y": ComponentTemplate("Y", "Crystal", ["1", "2"],
                               ["8MHz", "12MHz", "16MHz", "32.768kHz"], 4, 3),
        "XTAL": ComponentTemplate("Y", "CrystalOsc", ["1", "2", "GND"],
                                  ["3225-8MHz", "3225-16MHz"], 5, 3),
        "BUZZER": ComponentTemplate("LS", "Buzzer", ["1", "2"],
                                    ["Passive", "Active"], 4, 3),
        "THERM": ComponentTemplate("RT", "Thermistor", ["1", "2"],
                                   ["10k NTC", "100k NTC"], 4, 2),
        "PHOTO": ComponentTemplate("PH", "Photodiode", ["A", "K"],
                                   ["BPW34", "TEMD6200"], 3, 3),
    }

    # Circuit templates (predefined topologies)
    CIRCUIT_TEMPLATES = {
        "voltage_divider": {
            "description": "Simple voltage divider with two resistors",
            "components": ["R", "R"],
            "nets": [("R1:2", "R2:1"), ("R2:2", "GND")],
            "power_nets": ["VCC"],
        },
        "rc_filter": {
            "description": "RC low-pass filter",
            "components": ["R", "C"],
            "nets": [("R1:2", "C1:1"), ("C1:2", "GND")],
            "power_nets": ["VCC"],
        },
        "led_driver": {
            "description": "LED with current-limiting resistor",
            "components": ["R", "LED"],
            "nets": [("R1:2", "LED1:A"), ("LED1:K", "GND")],
            "power_nets": ["VCC"],
        },
        "mcu_minimal": {
            "description": "MCU minimal system with decoupling and crystal",
            "components": ["U_MCU", "C", "C", "C", "Y", "R"],
            "nets": [],
            "power_nets": ["VCC", "GND"],
        },
        "ldo_power": {
            "description": "LDO voltage regulator with input/output caps",
            "components": ["U_LDO", "C", "C"],
            "nets": [],
            "power_nets": ["VIN", "VCC", "GND"],
        },
        "opamp_gain": {
            "description": "Non-inverting amplifier",
            "components": ["U_OA", "R", "R"],
            "nets": [],
            "power_nets": ["VCC", "VEE"],
        },
        "buck_converter": {
            "description": "Buck DC-DC converter",
            "components": ["U_DRV", "Q_NMOS", "L", "D", "C", "R", "R"],
            "nets": [],
            "power_nets": ["VIN", "VOUT", "GND"],
        },
        "sensor_interface": {
            "description": "I2C sensor with pullups",
            "components": ["U_ADC", "R", "R", "C", "C"],
            "nets": [],
            "power_nets": ["VCC", "GND"],
        },
        "555_timer": {
            "description": "555 timer astable circuit",
            "components": ["U_TIMER", "R", "R", "C", "C"],
            "nets": [],
            "power_nets": ["VCC", "GND"],
        },
        "usb_interface": {
            "description": "USB to serial interface",
            "components": ["J_USB", "U_MCU", "R", "R", "C", "C"],
            "nets": [],
            "power_nets": ["VBUS", "VCC", "GND"],
        },
    }

    def __init__(
        self,
        output_dir: str,
        grid_spacing_mm: float = 2.54,
        sheet_width_mm: float = 297.0,
        sheet_height_mm: float = 210.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grid_spacing = grid_spacing_mm
        self.sheet_width = sheet_width_mm
        self.sheet_height = sheet_height_mm
        self._ref_counters: Dict[str, int] = {}

    def _next_ref(self, prefix: str) -> str:
        """Generate next reference designator."""
        if prefix not in self._ref_counters:
            self._ref_counters[prefix] = 0
        self._ref_counters[prefix] += 1
        return f"{prefix}{self._ref_counters[prefix]}"

    def _snap_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Snap coordinates to grid."""
        gs = self.grid_spacing
        return (round(x / gs) * gs, round(y / gs) * gs)

    def generate_random_circuit(
        self,
        num_components: int = 10,
        circuit_type: str = "mixed",
        complexity: str = "medium",
    ) -> Dict:
        """
        Generate a random circuit schematic specification.

        Args:
            num_components: Target number of components
            circuit_type: "analog", "digital", "mixed", "power"
            complexity: "simple", "medium", "complex"

        Returns:
            Circuit specification dict
        """
        self._ref_counters = {}

        # Select component mix based on circuit type
        component_pool = self._select_component_pool(circuit_type, num_components)

        # Place components with spacing
        placed = self._place_components(component_pool)

        # Generate connections
        connections = self._generate_connections(placed)

        # Generate net labels
        net_labels = self._generate_net_labels(placed, connections)

        return {
            "components": placed,
            "connections": connections,
            "net_labels": net_labels,
            "circuit_type": circuit_type,
            "complexity": complexity,
            "num_components": len(placed),
        }

    def _select_component_pool(
        self, circuit_type: str, count: int
    ) -> List[str]:
        """Select components based on circuit type."""
        pool = []

        if circuit_type == "analog":
            weights = {"R": 30, "C": 25, "L": 5, "D": 10, "Q_NPN": 8,
                       "Q_PNP": 4, "U_OA": 8, "U_COMP": 3, "TP": 3, "F": 2}
        elif circuit_type == "digital":
            weights = {"U_MCU": 15, "U_LOGIC": 20, "U_ADC": 8, "U_DAC": 5,
                       "R": 15, "C": 10, "LED": 8, "J_HDR": 5, "Y": 5,
                       "SW": 4, "TP": 3, "F": 2}
        elif circuit_type == "power":
            weights = {"U_REG": 15, "U_LDO": 15, "U_DRV": 10, "Q_NMOS": 10,
                       "Q_PMOS": 5, "D": 10, "L": 8, "C": 15, "R": 8,
                       "F": 3, "TP": 1}
        else:  # mixed
            weights = {"R": 20, "C": 15, "U_MCU": 10, "U_LOGIC": 8,
                       "U_OA": 5, "U_REG": 5, "D": 5, "LED": 5, "Q_NPN": 5,
                       "J_HDR": 5, "Y": 3, "SW": 3, "TP": 3, "U_ADC": 3,
                       "L": 3, "F": 2}

        types = list(weights.keys())
        w = list(weights.values())
        total_w = sum(w)

        for _ in range(count):
            r = random.random() * total_w
            cumulative = 0
            for t, weight in zip(types, w):
                cumulative += weight
                if r <= cumulative:
                    pool.append(t)
                    break

        return pool

    def _place_components(self, pool: List[str]) -> List[PlacedComponent]:
        """Place components on a grid layout with tighter spacing for connectivity."""
        placed = []
        margin = 20
        col_width = 15  # Tighter spacing for better connectivity
        row_height = 12
        max_cols = max(1, int((self.sheet_width - 2 * margin) / col_width))

        for i, comp_type in enumerate(pool):
            template = self.COMPONENT_TEMPLATES.get(comp_type)
            if not template:
                continue

            col = i % max_cols
            row = i // max_cols

            x, y = self._snap_to_grid(
                margin + col * col_width + random.uniform(-2, 2),
                margin + row * row_height + random.uniform(-2, 2),
            )

            rotation = random.choice([0, 0, 0, 90, 180, 270])
            ref = self._next_ref(template.prefix)
            value = random.choice(template.values)

            comp = PlacedComponent(
                template=template,
                reference=ref,
                value=value,
                x=x, y=y,
                rotation=rotation,
            )

            # Compute pin positions
            self._compute_pin_positions(comp)
            placed.append(comp)

        return placed

    def _compute_pin_positions(self, comp: PlacedComponent):
        """Compute absolute pin positions for a placed component."""
        n_pins = len(comp.template.pin_names)
        spacing = 2.54  # 100mil

        for i, pin_name in enumerate(comp.template.pin_names):
            # Default: pins along the component's height
            if comp.rotation in (0, 180):
                px = comp.x
                py = comp.y + (i - n_pins / 2) * spacing
            else:
                px = comp.x + (i - n_pins / 2) * spacing
                py = comp.y

            # Apply rotation
            if comp.rotation == 90:
                dx, dy = px - comp.x, py - comp.y
                px = comp.x - dy
                py = comp.y + dx
            elif comp.rotation == 180:
                dx, dy = px - comp.x, py - comp.y
                px = comp.x - dx
                py = comp.y - dy
            elif comp.rotation == 270:
                dx, dy = px - comp.x, py - comp.y
                px = comp.x + dy
                py = comp.y - dx

            comp.pin_positions[pin_name] = self._snap_to_grid(px, py)

    def _generate_connections(
        self, placed: List[PlacedComponent]
    ) -> List[Dict]:
        """Generate wire connections between components."""
        connections = []
        all_pins = []

        for comp in placed:
            for pin_name, pos in comp.pin_positions.items():
                all_pins.append({
                    "ref": comp.reference,
                    "pin": pin_name,
                    "pos": pos,
                    "connected": False,
                })

        # Connect nearby pins with more aggressive threshold
        for i, pin1 in enumerate(all_pins):
            if pin1["connected"]:
                continue

            # Find closest compatible pin
            best_j = -1
            best_dist = 30.0  # Max connection distance

            for j, pin2 in enumerate(all_pins):
                if i == j or pin2["connected"]:
                    continue
                if pin1["ref"] == pin2["ref"]:
                    continue

                dx = abs(pin1["pos"][0] - pin2["pos"][0])
                dy = abs(pin1["pos"][1] - pin2["pos"][1])
                dist = math.sqrt(dx**2 + dy**2)

                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                pin2 = all_pins[best_j]
                connections.append({
                    "from": f"{pin1['ref']}:{pin1['pin']}",
                    "to": f"{pin2['ref']}:{pin2['pin']}",
                    "from_pos": pin1["pos"],
                    "to_pos": pin2["pos"],
                })
                pin1["connected"] = True
                pin2["connected"] = True

        # Force-connect remaining unconnected pins to power nets
        for pin in all_pins:
            if pin["connected"]:
                continue
            pin_name = pin["pin"].upper()
            if pin_name in ("VCC", "VDD", "VIN", "VOUT", "VBUS"):
                pin["connected"] = True  # Will be handled by net labels
            elif pin_name in ("GND", "VEE"):
                pin["connected"] = True

        return connections

    def _generate_net_labels(
        self, placed: List[PlacedComponent], connections: List[Dict]
    ) -> List[Dict]:
        """Generate net labels for power nets and named signals."""
        labels = []
        power_nets = ["VCC", "GND", "VIN", "3V3", "5V"]

        # Find unconnected pins and assign power nets
        connected_pins = set()
        for conn in connections:
            connected_pins.add(conn["from"])
            connected_pins.add(conn["to"])

        for comp in placed:
            for pin_name in comp.template.pin_names:
                pin_key = f"{comp.reference}:{pin_name}"
                if pin_key in connected_pins:
                    continue

                # Assign power net labels to power pins
                if pin_name.upper() in ("VCC", "VDD", "VIN"):
                    labels.append({
                        "name": random.choice(["VCC", "VIN", "5V", "3V3"]),
                        "pos": comp.pin_positions.get(pin_name, (comp.x, comp.y)),
                    })
                elif pin_name.upper() in ("GND", "VEE"):
                    labels.append({
                        "name": "GND",
                        "pos": comp.pin_positions.get(pin_name, (comp.x, comp.y)),
                    })

        return labels

    def render_synthetic_schematic(
        self,
        circuit_spec: Dict,
        output_path: str,
    ) -> str:
        """
        Render a synthetic circuit schematic to PNG.

        Args:
            circuit_spec: Circuit specification from generate_random_circuit
            output_path: Path for output PNG

        Returns:
            Path to generated PNG
        """
        from PIL import Image, ImageDraw, ImageFont

        dpi = 300
        mm_to_px = dpi / 25.4
        width = int(self.sheet_width * mm_to_px)
        height = int(self.sheet_height * mm_to_px)

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        def to_px(x_mm, y_mm):
            return (int(x_mm * mm_to_px), int(y_mm * mm_to_px))

        # Draw grid
        gs_px = int(self.grid_spacing * mm_to_px)
        for x in range(0, width, gs_px):
            draw.line([(x, 0), (x, height)], fill="#E8E8E8", width=1)
        for y in range(0, height, gs_px):
            draw.line([(0, y), (width, y)], fill="#E8E8E8", width=1)

        # Draw connections (wires)
        for conn in circuit_spec.get("connections", []):
            p1 = to_px(*conn["from_pos"])
            p2 = to_px(*conn["to_pos"])
            draw.line([p1, p2], fill="#000000", width=2)

            # Draw intermediate wire segments (Manhattan routing)
            mid_x = (p1[0] + p2[0]) // 2
            draw.line([p1, (mid_x, p1[1])], fill="#000000", width=2)
            draw.line([(mid_x, p1[1]), (mid_x, p2[1])], fill="#000000", width=2)
            draw.line([(mid_x, p2[1]), p2], fill="#000000", width=2)

        # Draw components
        for comp in circuit_spec.get("components", []):
            cx, cy = to_px(comp.x, comp.y)
            w = int(comp.template.width_mm * mm_to_px / 2)
            h = int(comp.template.height_mm * mm_to_px / 2)

            # Draw component box
            draw.rectangle(
                [(cx - w, cy - h), (cx + w, cy + h)],
                outline="#0000CC", width=2, fill="#F0F0FF"
            )

            # Draw reference text
            try:
                font = ImageFont.truetype("arial.ttf", max(10, h))
            except (OSError, IOError):
                font = ImageFont.load_default()

            draw.text((cx - w, cy - h - h), comp.reference,
                      fill="#CC0000", font=font)
            draw.text((cx - w, cy + h + 2), comp.value,
                      fill="#006600", font=font)

            # Draw pins
            for pin_name, pin_pos in comp.pin_positions.items():
                px, py = to_px(*pin_pos)
                r = max(2, int(0.5 * mm_to_px))
                draw.ellipse([(px-r, py-r), (px+r, py+r)], fill="#000000")

        # Draw net labels
        for label in circuit_spec.get("net_labels", []):
            lx, ly = to_px(*label["pos"])
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()
            draw.text((lx, ly - 8), label["name"], fill="#0000CC", font=font)

        img.save(output_path)
        return output_path

    def generate_kicad_sch(
        self,
        circuit_spec: Dict,
        output_path: str,
    ) -> str:
        """
        Generate a .kicad_sch file from circuit specification.

        This creates a valid KiCad schematic that can be parsed by the
        standard KiCadParser for consistent annotation extraction.
        """
        lines = []
        lines.append('(kicad_sch (version 20230121) (generator "circuit_ocr_synth")')
        lines.append('  (uuid "00000000-0000-0000-0000-000000000001")')
        lines.append(f'  (paper "A")')
        lines.append('')

        # Library symbols section
        lines.append('  (lib_symbols')

        seen_libs = set()
        for comp in circuit_spec.get("components", []):
            lib_name = f"circuit_ocr:{comp.template.name}"
            if lib_name in seen_libs:
                continue
            seen_libs.add(lib_name)

            lines.append(f'    (symbol "{lib_name}"')
            lines.append(f'      (pin_names (offset 1.016))')
            for i, pin_name in enumerate(comp.template.pin_names):
                lines.append(
                    f'      (pin passive line'
                    f' (at 0 {i * 2.54 - len(comp.template.pin_names) * 1.27:.2f} 0)'
                    f' (length 2.54)'
                    f' (name "{pin_name}" (effects (font (size 1.27 1.27))))'
                    f' (number "{pin_name}" (effects (font (size 1.27 1.27))))'
                    f')'
                )
            lines.append('    )')

        lines.append('  )')
        lines.append('')

        # Component instances
        for comp in circuit_spec.get("components", []):
            lib_name = f"circuit_ocr:{comp.template.name}"
            rotation = comp.rotation if hasattr(comp, 'rotation') else 0
            lines.append(f'  (symbol (lib_id "{lib_name}")')
            lines.append(f'    (at {comp.x:.2f} {comp.y:.2f} {rotation})')
            lines.append(f'    (unit 1)')
            lines.append(f'    (uuid "{id(comp):08x}-0000-0000-0000-000000000000")')
            lines.append(f'    (property "Reference" "{comp.reference}"')
            lines.append(f'      (at {comp.x:.2f} {comp.y - 3:.2f} 0)')
            lines.append(f'      (effects (font (size 1.27 1.27)))')
            lines.append(f'    )')
            lines.append(f'    (property "Value" "{comp.value}"')
            lines.append(f'      (at {comp.x:.2f} {comp.y + 3:.2f} 0)')
            lines.append(f'      (effects (font (size 1.27 1.27)))')
            lines.append(f'    )')
            lines.append(f'  )')

        lines.append('')

        # Wires
        for conn in circuit_spec.get("connections", []):
            x1, y1 = conn["from_pos"]
            x2, y2 = conn["to_pos"]
            lines.append(f'  (wire (pts (xy {x1:.2f} {y1:.2f}) (xy {x2:.2f} {y2:.2f}))')
            lines.append(f'    (stroke (width 0) (type default))')
            lines.append(f'    (uuid "{hash((x1,y1,x2,y2)) & 0xFFFFFFFF:08x}-0000-0000-0000-000000000000")')
            lines.append(f'  )')

        lines.append('')

        # Net labels
        for label in circuit_spec.get("net_labels", []):
            lx, ly = label["pos"]
            lines.append(f'  (label "{label["name"]}" (at {lx:.2f} {ly:.2f} 0)')
            lines.append(f'    (effects (font (size 1.27 1.27)))')
            lines.append(f'    (uuid "{hash((lx,ly,label["name"])) & 0xFFFFFFFF:08x}-0000-0000-0000-000000000000")')
            lines.append(f'  )')

        lines.append(')')
        lines.append('')

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(lines), encoding='utf-8')

        return str(output_path)

    def batch_generate(
        self,
        count: int,
        circuit_type: str = "mixed",
        complexity: str = "medium",
        base_name: str = "synth",
    ) -> List[Dict]:
        """
        Generate a batch of synthetic circuits.

        Args:
            count: Number of circuits to generate
            circuit_type: Type of circuits
            complexity: Complexity level
            base_name: Base filename

        Returns:
            List of generation results with paths
        """
        results = []

        # Component count ranges by complexity
        count_ranges = {
            "simple": (5, 15),
            "medium": (15, 40),
            "complex": (40, 100),
        }
        min_c, max_c = count_ranges.get(complexity, (10, 30))

        for i in range(count):
            try:
                n_components = random.randint(min_c, max_c)

                spec = self.generate_random_circuit(
                    num_components=n_components,
                    circuit_type=circuit_type,
                    complexity=complexity,
                )

                # Generate KiCad file
                sch_path = str(self.output_dir / f"{base_name}_{i:04d}.kicad_sch")
                self.generate_kicad_sch(spec, sch_path)

                # Render PNG
                png_path = str(self.output_dir / f"{base_name}_{i:04d}.png")
                self.render_synthetic_schematic(spec, png_path)

                # Save spec JSON
                spec_path = str(self.output_dir / f"{base_name}_{i:04d}_spec.json")
                spec_json = {
                    "circuit_type": circuit_type,
                    "complexity": complexity,
                    "num_components": len(spec["components"]),
                    "num_connections": len(spec["connections"]),
                    "num_labels": len(spec["net_labels"]),
                    "components": [
                        {"ref": c.reference, "value": c.value,
                         "type": c.template.name}
                        for c in spec["components"]
                    ],
                }
                with open(spec_path, "w", encoding="utf-8") as f:
                    json.dump(spec_json, f, indent=2)

                results.append({
                    "sch_path": sch_path,
                    "png_path": png_path,
                    "spec_path": spec_path,
                    "num_components": len(spec["components"]),
                    "success": True,
                })

            except Exception as e:
                logger.error(f"Failed to generate synthetic circuit {i}: {e}")
                results.append({"index": i, "success": False, "error": str(e)})

        success = sum(1 for r in results if r.get("success"))
        logger.info(f"Batch generation: {success}/{count} succeeded")
        return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate synthetic schematics")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--type", default="mixed",
                       choices=["analog", "digital", "mixed", "power"])
    parser.add_argument("--complexity", default="medium",
                       choices=["simple", "medium", "complex"])
    parser.add_argument("--output-dir", default="data/synthetic")
    args = parser.parse_args()

    gen = SyntheticSchematicGenerator(output_dir=args.output_dir)
    results = gen.batch_generate(
        count=args.count,
        circuit_type=args.type,
        complexity=args.complexity,
    )

    print(f"Generated {sum(1 for r in results if r.get('success'))} synthetic circuits")
