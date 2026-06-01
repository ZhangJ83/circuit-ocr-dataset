"""
KiCad Schematic File Parser
============================
Parse .kicad_sch files (S-expression format) and extract all topological elements:
- Symbols (component instances with Reference, Value, Pin list)
- lib_symbols (component library definitions with pin relative coordinates)
- Wires (two-endpoint connections)
- Junctions (3+ wire intersection points)
- Labels (net names)
- No-connect markers

Key innovation: All annotations can be programmatically extracted from the source
file with ZERO manual labeling cost.

Connection relationship mechanism:
  KiCad stores connections implicitly via coordinate coincidence:
  - wire endpoint == pin absolute coordinate → wire connects to pin
  - two wires share endpoint → connected at that point
  - label coordinate == wire endpoint → wire's net name = label text
"""

import math
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Point:
    """2D point with optional rotation."""
    x: float
    y: float

    def __hash__(self):
        return hash((round(self.x, 4), round(self.y, 4)))

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return (abs(self.x - other.x) < 0.01 and
                abs(self.y - other.y) < 0.01)

    def to_tuple(self) -> Tuple[float, float]:
        return (round(self.x, 4), round(self.y, 4))


@dataclass
class PinInfo:
    """Pin information for a component."""
    pin_number: str
    pin_name: str
    relative_pos: Point      # Relative to component origin
    absolute_pos: Optional[Point] = None  # After rotation + translation
    pin_type: str = ""       # passive, input, output, power_in, power_out


@dataclass
class ComponentInstance:
    """A placed component instance on the schematic."""
    reference: str           # e.g., "R1", "C1", "U1"
    value: str              # e.g., "10k", "100nF", "STM32F103"
    lib_name: str           # Library symbol name
    position: Point          # Placement position
    rotation: float = 0.0   # Rotation in degrees
    footprint: str = ""
    pins: List[PinInfo] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)
    bbox: Optional[Tuple[Point, Point]] = None  # Bounding box


@dataclass
class Wire:
    """A wire connecting two points."""
    start: Point
    end: Point
    uuid: str = ""


@dataclass
class Junction:
    """A junction point (3+ wires meeting)."""
    position: Point


@dataclass
class NetLabel:
    """A net label with a name and position."""
    name: str
    position: Point
    rotation: float = 0.0


@dataclass
class NoConnect:
    """A no-connect marker."""
    position: Point


@dataclass
class TextAnnotation:
    """Free text annotation on the schematic."""
    text: str
    position: Point
    rotation: float = 0.0


@dataclass
class SchematicData:
    """Complete parsed schematic data."""
    file_path: str
    components: List[ComponentInstance] = field(default_factory=list)
    wires: List[Wire] = field(default_factory=list)
    junctions: List[Junction] = field(default_factory=list)
    labels: List[NetLabel] = field(default_factory=list)
    no_connects: List[NoConnect] = field(default_factory=list)
    texts: List[TextAnnotation] = field(default_factory=list)
    lib_symbols: Dict[str, Dict] = field(default_factory=dict)
    sheet_size: Tuple[float, float] = (297.0, 210.0)  # A4 default


class KiCadParser:
    """
    Parser for KiCad .kicad_sch files.

    Extracts all 6 types of topological elements and computes
    pin absolute coordinates considering rotation.
    """

    # Standard component types for classification
    COMPONENT_TYPES = {
        "R": "Resistor",
        "C": "Capacitor",
        "L": "Inductor",
        "D": "Diode",
        "LED": "LED",
        "Q": "Transistor",
        "U": "IC",
        "J": "Connector",
        "TP": "TestPoint",
        "F": "Fuse",
        "SW": "Switch",
        "Y": "Crystal",
        "X": "Miscellaneous",
    }

    def __init__(self):
        self._lib_cache: Dict[str, Dict] = {}

    def parse(self, sch_path: str) -> SchematicData:
        """
        Parse a .kicad_sch file and return structured data.

        Args:
            sch_path: Path to .kicad_sch file

        Returns:
            SchematicData with all extracted elements
        """
        sch_path = Path(sch_path)
        if not sch_path.exists():
            raise FileNotFoundError(f"Schematic file not found: {sch_path}")

        try:
            from kiutils.schematic import Schematic
            sch = Schematic.from_file(str(sch_path))
        except ImportError:
            logger.warning("kiutils not available, falling back to manual parser")
            return self._parse_manual(sch_path)

        data = SchematicData(file_path=str(sch_path))

        # Extract sheet size
        if hasattr(sch, 'paper') and sch.paper:
            try:
                parts = sch.paper.split()
                if len(parts) >= 2:
                    data.sheet_size = (float(parts[0]), float(parts[1]))
            except (ValueError, AttributeError):
                pass

        # Extract library symbol definitions (for pin positions)
        self._extract_lib_symbols(sch, data)

        # Extract component instances
        self._extract_components(sch, data)

        # Extract wires
        self._extract_wires(sch, data)

        # Extract junctions
        self._extract_junctions(sch, data)

        # Extract labels
        self._extract_labels(sch, data)

        # Extract no-connects
        self._extract_no_connects(sch, data)

        # Extract text annotations
        self._extract_texts(sch, data)

        # Compute pin absolute coordinates
        self._compute_pin_positions(data)

        logger.info(
            f"Parsed {sch_path.name}: "
            f"{len(data.components)} components, "
            f"{len(data.wires)} wires, "
            f"{len(data.labels)} labels"
        )

        return data

    def _extract_lib_symbols(self, sch, data: SchematicData):
        """Extract library symbol definitions (pin templates)."""
        try:
            if not hasattr(sch, 'libSymbols'):
                return

            for lib_sym in sch.libSymbols:
                sym_name = lib_sym.libId if hasattr(lib_sym, 'libId') else ""
                if not sym_name:
                    continue

                pins = []
                if hasattr(lib_sym, 'pins') and lib_sym.pins:
                    for pin in lib_sym.pins:
                        pin_num = ""
                        pin_name = ""
                        pin_type = ""
                        pos = Point(0, 0)

                        if hasattr(pin, 'number') and pin.number:
                            pin_num = str(pin.number) if hasattr(pin.number, 'name') else str(pin.number)
                        if hasattr(pin, 'name') and pin.name:
                            pin_name = str(pin.name) if hasattr(pin.name, 'name') else str(pin.name)
                        if hasattr(pin, 'electricalType'):
                            pin_type = str(pin.electricalType)
                        if hasattr(pin, 'position') and pin.position:
                            pos = Point(
                                float(pin.position.X) if hasattr(pin.position, 'X') else 0,
                                float(pin.position.Y) if hasattr(pin.position, 'Y') else 0,
                            )

                        pins.append({
                            "number": pin_num,
                            "name": pin_name,
                            "type": pin_type,
                            "x": pos.x,
                            "y": pos.y,
                        })

                self._lib_cache[sym_name] = {"name": sym_name, "pins": pins}
                data.lib_symbols[sym_name] = self._lib_cache[sym_name]

        except Exception as e:
            logger.warning(f"Error extracting lib symbols: {e}")

    def _extract_components(self, sch, data: SchematicData):
        """Extract placed component instances."""
        try:
            if not hasattr(sch, 'schematicSymbols'):
                return

            for sym in sch.schematicSymbols:
                comp = ComponentInstance(
                    reference="",
                    value="",
                    lib_name=sym.libId if hasattr(sym, 'libId') else "",
                    position=Point(0, 0),
                    rotation=0.0,
                )

                # Position
                if hasattr(sym, 'position') and sym.position:
                    comp.position = Point(
                        float(sym.position.X) if hasattr(sym.position, 'X') else 0,
                        float(sym.position.Y) if hasattr(sym.position, 'Y') else 0,
                    )

                # Rotation
                if hasattr(sym, 'position') and sym.position:
                    if hasattr(sym.position, 'angle'):
                        comp.rotation = float(sym.position.angle) if sym.position.angle else 0.0

                # Properties (Reference, Value, Footprint, etc.)
                if hasattr(sym, 'properties') and sym.properties:
                    for prop in sym.properties:
                        key = prop.key if hasattr(prop, 'key') else ""
                        value = prop.value if hasattr(prop, 'value') else ""
                        if key == "Reference":
                            comp.reference = value
                        elif key == "Value":
                            comp.value = value
                        elif key == "Footprint":
                            comp.footprint = value
                        if key and value:
                            comp.properties[key] = value

                # Look up pin template from lib
                if comp.lib_name in self._lib_cache:
                    lib_pins = self._lib_cache[comp.lib_name]["pins"]
                    for lp in lib_pins:
                        comp.pins.append(PinInfo(
                            pin_number=lp["number"],
                            pin_name=lp["name"],
                            relative_pos=Point(lp["x"], lp["y"]),
                            pin_type=lp["type"],
                        ))

                # Classify component type
                if comp.reference:
                    prefix = ''.join(c for c in comp.reference if c.isalpha())
                    comp.properties["_type"] = self.COMPONENT_TYPES.get(
                        prefix, "Unknown"
                    )

                data.components.append(comp)

        except Exception as e:
            logger.warning(f"Error extracting components: {e}")

    def _extract_wires(self, sch, data: SchematicData):
        """Extract wires from sch.wires or graphicalItems (Connection objects)."""
        try:
            # Try sch.wires first (standard KiCad format)
            if hasattr(sch, 'wires') and sch.wires:
                for wire in sch.wires:
                    points = []
                    if hasattr(wire, 'points') and wire.points:
                        for pt in wire.points:
                            x = float(pt.X) if hasattr(pt, 'X') else 0
                            y = float(pt.Y) if hasattr(pt, 'Y') else 0
                            points.append(Point(x, y))

                    if len(points) >= 2:
                        uuid = wire.uuid if hasattr(wire, 'uuid') else ""
                        data.wires.append(Wire(
                            start=points[0],
                            end=points[1],
                            uuid=str(uuid),
                        ))

            # Also check graphicalItems for Connection objects (kiutils format)
            if hasattr(sch, 'graphicalItems') and sch.graphicalItems:
                for item in sch.graphicalItems:
                    if type(item).__name__ == 'Connection':
                        points = []
                        if hasattr(item, 'points') and item.points:
                            for pt in item.points:
                                x = float(pt.X) if hasattr(pt, 'X') else 0
                                y = float(pt.Y) if hasattr(pt, 'Y') else 0
                                points.append(Point(x, y))

                        if len(points) >= 2:
                            uuid = item.uuid if hasattr(item, 'uuid') else ""
                            data.wires.append(Wire(
                                start=points[0],
                                end=points[1],
                                uuid=str(uuid),
                            ))

        except Exception as e:
            logger.warning(f"Error extracting wires: {e}")

    def _extract_junctions(self, sch, data: SchematicData):
        """Extract junction points."""
        try:
            if not hasattr(sch, 'junctions'):
                return

            for junc in sch.junctions:
                pos = Point(0, 0)
                if hasattr(junc, 'position') and junc.position:
                    pos = Point(
                        float(junc.position.X) if hasattr(junc.position, 'X') else 0,
                        float(junc.position.Y) if hasattr(junc.position, 'Y') else 0,
                    )
                data.junctions.append(Junction(position=pos))

        except Exception as e:
            logger.warning(f"Error extracting junctions: {e}")

    def _extract_labels(self, sch, data: SchematicData):
        """Extract net labels."""
        try:
            # Regular labels
            if hasattr(sch, 'labels'):
                for label in sch.labels:
                    name = label.name if hasattr(label, 'name') else ""
                    pos = Point(0, 0)
                    rotation = 0.0
                    if hasattr(label, 'position') and label.position:
                        pos = Point(
                            float(label.position.X) if hasattr(label.position, 'X') else 0,
                            float(label.position.Y) if hasattr(label.position, 'Y') else 0,
                        )
                        if hasattr(label.position, 'angle'):
                            rotation = float(label.position.angle) if label.position.angle else 0.0
                    if name:
                        data.labels.append(NetLabel(
                            name=name, position=pos, rotation=rotation
                        ))

            # Global labels (VCC, GND, etc.)
            if hasattr(sch, 'globalLabels'):
                for label in sch.globalLabels:
                    name = label.name if hasattr(label, 'name') else ""
                    pos = Point(0, 0)
                    if hasattr(label, 'position') and label.position:
                        pos = Point(
                            float(label.position.X) if hasattr(label.position, 'X') else 0,
                            float(label.position.Y) if hasattr(label.position, 'Y') else 0,
                        )
                    if name:
                        data.labels.append(NetLabel(name=name, position=pos))

            # Hierarchical labels
            if hasattr(sch, 'hierarchicalLabels'):
                for label in sch.hierarchicalLabels:
                    name = label.name if hasattr(label, 'name') else ""
                    pos = Point(0, 0)
                    if hasattr(label, 'position') and label.position:
                        pos = Point(
                            float(label.position.X) if hasattr(label.position, 'X') else 0,
                            float(label.position.Y) if hasattr(label.position, 'Y') else 0,
                        )
                    if name:
                        data.labels.append(NetLabel(name=name, position=pos))

        except Exception as e:
            logger.warning(f"Error extracting labels: {e}")

    def _extract_no_connects(self, sch, data: SchematicData):
        """Extract no-connect markers."""
        try:
            if not hasattr(sch, 'noConnects'):
                return

            for nc in sch.noConnects:
                pos = Point(0, 0)
                if hasattr(nc, 'position') and nc.position:
                    pos = Point(
                        float(nc.position.X) if hasattr(nc.position, 'X') else 0,
                        float(nc.position.Y) if hasattr(nc.position, 'Y') else 0,
                    )
                data.no_connects.append(NoConnect(position=pos))

        except Exception as e:
            logger.warning(f"Error extracting no-connects: {e}")

    def _extract_texts(self, sch, data: SchematicData):
        """Extract text annotations."""
        try:
            if not hasattr(sch, 'texts'):
                return

            for text in sch.texts:
                content = text.text if hasattr(text, 'text') else ""
                pos = Point(0, 0)
                rotation = 0.0
                if hasattr(text, 'position') and text.position:
                    pos = Point(
                        float(text.position.X) if hasattr(text.position, 'X') else 0,
                        float(text.position.Y) if hasattr(text.position, 'Y') else 0,
                    )
                    if hasattr(text.position, 'angle'):
                        rotation = float(text.position.angle) if text.position.angle else 0.0
                if content:
                    data.texts.append(TextAnnotation(
                        text=content, position=pos, rotation=rotation
                    ))

        except Exception as e:
            logger.warning(f"Error extracting texts: {e}")

    @staticmethod
    def _rotate_point(x: float, y: float, angle_deg: float) -> Tuple[float, float]:
        """Rotate a point around origin by angle_deg degrees."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        return (new_x, new_y)

    def _compute_pin_positions(self, data: SchematicData):
        """Compute absolute pin positions considering component rotation and placement."""
        for comp in data.components:
            for pin in comp.pins:
                # Rotate relative position by component rotation
                rx, ry = self._rotate_point(
                    pin.relative_pos.x, pin.relative_pos.y, comp.rotation
                )
                # Translate to absolute position
                pin.absolute_pos = Point(
                    comp.position.x + rx,
                    comp.position.y + ry,
                )

    def extract_netlist(self, data: SchematicData) -> Dict[str, List[str]]:
        """
        Extract netlist from parsed schematic data.

        Uses BFS/DFS on coordinate-matching to find connected components.
        Each connected component = one net.

        Returns:
            Dict mapping net_name -> list of "REF:PIN" strings
        """
        # Build coordinate → element mapping
        pin_points: Dict[Point, List[Tuple[str, str]]] = {}  # point -> [(ref, pin)]
        wire_adj: Dict[Point, List[Point]] = {}  # adjacency list

        # Register pin positions
        for comp in data.components:
            for pin in comp.pins:
                if pin.absolute_pos:
                    pt = pin.absolute_pos
                    if pt not in pin_points:
                        pin_points[pt] = []
                    pin_points[pt].append((comp.reference, pin.pin_number or pin.pin_name))

        # Build wire adjacency
        for wire in data.wires:
            start, end = wire.start, wire.end
            if start not in wire_adj:
                wire_adj[start] = []
            if end not in wire_adj:
                wire_adj[end] = []
            wire_adj[start].append(end)
            wire_adj[end].append(start)

        # Build label → point mapping
        label_at_point: Dict[Point, str] = {}
        for label in data.labels:
            label_at_point[label.position] = label.name

        # BFS to find connected components
        all_points = set(pin_points.keys()) | set(wire_adj.keys()) | set(label_at_point.keys())
        visited = set()
        nets = {}

        for start_pt in all_points:
            if start_pt in visited:
                continue

            # BFS
            queue = [start_pt]
            component_points = set()
            while queue:
                pt = queue.pop(0)
                if pt in visited:
                    continue
                visited.add(pt)
                component_points.add(pt)

                # Follow wires
                for neighbor in wire_adj.get(pt, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Collect all pins and labels in this component
            net_pins = []
            net_name = None

            for pt in component_points:
                for ref, pin_num in pin_points.get(pt, []):
                    net_pins.append(f"{ref}:{pin_num}")
                if pt in label_at_point:
                    net_name = label_at_point[pt]

            if net_pins:
                if not net_name:
                    # Auto-generate net name
                    net_name = f"NET_{len(nets)}"
                if net_name not in nets:
                    nets[net_name] = []
                nets[net_name].extend(net_pins)

        return nets

    def _parse_manual(self, sch_path: Path) -> SchematicData:
        """Fallback manual parser when kiutils is not available."""
        import re

        data = SchematicData(file_path=str(sch_path))
        content = sch_path.read_text(encoding="utf-8")

        # Extract component references and values with regex
        # This is a simplified parser for fallback
        ref_pattern = re.compile(r'\(property\s+"Reference"\s+"([^"]+)"')
        val_pattern = re.compile(r'\(property\s+"Value"\s+"([^"]+)"')
        wire_pattern = re.compile(
            r'\(wire\s+\(pts\s+\(xy\s+([\d.]+)\s+([\d.]+)\)\s+\(xy\s+([\d.]+)\s+([\d.]+)\)\)'
        )
        label_pattern = re.compile(
            r'\(label\s+"([^"]+)"\s+\(at\s+([\d.]+)\s+([\d.]+)'
        )

        # Parse wires
        for m in wire_pattern.finditer(content):
            data.wires.append(Wire(
                start=Point(float(m.group(1)), float(m.group(2))),
                end=Point(float(m.group(3)), float(m.group(4))),
            ))

        # Parse labels
        for m in label_pattern.finditer(content):
            data.labels.append(NetLabel(
                name=m.group(1),
                position=Point(float(m.group(2)), float(m.group(3))),
            ))

        logger.info(
            f"Manual parse {sch_path.name}: "
            f"{len(data.wires)} wires, {len(data.labels)} labels"
        )
        return data


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python kicad_parser.py <file.kicad_sch>")
        sys.exit(1)

    parser = KiCadParser()
    data = parser.parse(sys.argv[1])

    print(f"Components: {len(data.components)}")
    for c in data.components[:5]:
        print(f"  {c.reference} = {c.value} @ ({c.position.x:.1f}, {c.position.y:.1f})")

    print(f"\nWires: {len(data.wires)}")
    print(f"Labels: {len(data.labels)}")
    print(f"Junctions: {len(data.junctions)}")

    nets = parser.extract_netlist(data)
    print(f"\nNets: {len(nets)}")
    for name, pins in list(nets.items())[:10]:
        print(f"  {name}: {pins}")
