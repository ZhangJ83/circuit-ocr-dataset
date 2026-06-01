"""
Auto-Corrector
==============
Automatically fix common netlist issues based on SPICE verification feedback.

Common fixes:
1. Add missing ground connection
2. Add missing voltage source
3. Fix node naming issues
4. Add missing component parameters
"""

import re
import logging
from typing import str, List
from .spice_verifier import VerificationResult

logger = logging.getLogger(__name__)


class AutoCorrector:
    """Auto-correct common netlist issues."""

    def correct(self, netlist_text: str, verification: VerificationResult) -> str:
        """
        Apply auto-corrections based on verification results.

        Args:
            netlist_text: Original netlist text
            verification: Verification result with issues

        Returns:
            Corrected netlist text
        """
        corrected = netlist_text

        # Fix 1: Add missing ground
        if not verification.has_ground:
            corrected = self._add_ground(corrected)

        # Fix 2: Add missing voltage source
        if not verification.has_voltage_source:
            corrected = self._add_voltage_source(corrected)

        # Fix 3: Ensure .end statement
        if ".end" not in corrected.lower():
            corrected = corrected.rstrip() + "\n.end\n"

        # Fix 4: Fix common value format issues
        corrected = self._fix_values(corrected)

        return corrected

    def _add_ground(self, netlist: str) -> str:
        """Add a ground node connection."""
        # Find first component and connect it to ground via a large resistor
        lines = netlist.strip().split("\n")
        insert_idx = len(lines) - 1  # Before .end

        # Find a node to connect to ground
        first_node = None
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and re.match(r'^[RCLDQMU]', parts[0], re.IGNORECASE):
                first_node = parts[1] if parts[1] != "0" else parts[2]
                break

        if first_node:
            ground_line = f"R_GND_FIX {first_node} 0 1G"
            lines.insert(insert_idx, ground_line)
            logger.info(f"Added ground connection: {ground_line}")

        return "\n".join(lines)

    def _add_voltage_source(self, netlist: str) -> str:
        """Add a default voltage source."""
        lines = netlist.strip().split("\n")
        insert_idx = len(lines) - 1  # Before .end

        # Find VCC-like nodes
        vcc_nodes = set()
        for line in lines:
            if re.search(r'\b(VCC|VDD|VIN|5V|3V3|12V)\b', line, re.IGNORECASE):
                match = re.search(r'\b(VCC|VDD|VIN|5V|3V3|12V)\b', line, re.IGNORECASE)
                if match:
                    vcc_nodes.add(match.group(0))

        if vcc_nodes:
            for node in vcc_nodes:
                voltage = "5"
                if "3V3" in node.upper() or "3.3" in node:
                    voltage = "3.3"
                elif "12V" in node.upper() or "12" in node:
                    voltage = "12"
                elif "1.8" in node:
                    voltage = "1.8"

                vsrc = f"V_{node} {node} 0 DC {voltage}V"
                lines.insert(insert_idx, vsrc)
                logger.info(f"Added voltage source: {vsrc}")
                break
        else:
            # Default: add 5V between first two unique nodes
            nodes = set()
            for line in lines:
                parts = line.split()
                if len(parts) >= 3 and re.match(r'^[RCLDQMU]', parts[0], re.IGNORECASE):
                    for p in parts[1:3]:
                        if not re.match(r'^\d+\.?\d*', p):
                            nodes.add(p)
            if len(nodes) >= 2:
                node_list = list(nodes)
                vsrc = f"V_VCC {node_list[0]} {node_list[1]} DC 5V"
                lines.insert(insert_idx, vsrc)
                logger.info(f"Added voltage source: {vsrc}")

        return "\n".join(lines)

    def _fix_values(self, netlist: str) -> str:
        """Fix common SPICE value format issues."""
        # Replace common unit suffixes
        replacements = [
            (r'(\d+)\s*k\b', r'\1k'),      # 10 k → 10k
            (r'(\d+)\s*M\b', r'\1meg'),     # 10 M → 10meg
            (r'(\d+)\s*u\b', r'\1u'),       # 100 u → 100u
            (r'(\d+)\s*n\b', r'\1n'),       # 100 n → 100n
            (r'(\d+)\s*p\b', r'\1p'),       # 10 p → 10p
        ]

        corrected = netlist
        for pattern, replacement in replacements:
            corrected = re.sub(pattern, replacement, corrected)

        return corrected
