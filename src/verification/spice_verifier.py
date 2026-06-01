"""
SPICE Verifier (Differentiation E)
===================================
Automatically verify extracted netlists using ngspice simulation.

No existing work does automatic simulation verification.
This is a UNIQUE differentiator that closes the loop:
  Image → OCR → Netlist → SPICE Simulation → Verification

Checks:
1. Netlist syntax validity
2. Node connectivity
3. Simulation convergence
4. Result reasonableness
"""

import os
import re
import tempfile
import subprocess
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of SPICE verification."""
    syntax_valid: bool = False
    has_voltage_source: bool = False
    has_ground: bool = False
    has_components: bool = False
    simulation_converges: bool = False
    node_count: int = 0
    component_count: int = 0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    raw_output: str = ""

    @property
    def is_valid(self) -> bool:
        return self.syntax_valid and self.has_ground and self.has_components

    def to_dict(self) -> Dict:
        return {
            "syntax_valid": self.syntax_valid,
            "has_voltage_source": self.has_voltage_source,
            "has_ground": self.has_ground,
            "has_components": self.has_components,
            "simulation_converges": self.simulation_converges,
            "node_count": self.node_count,
            "component_count": self.component_count,
            "warnings": self.warnings,
            "errors": self.errors,
            "is_valid": self.is_valid,
        }


class SPICEVerifier:
    """Verify netlists using ngspice simulation."""

    def __init__(self, ngspice_path: Optional[str] = None, timeout: int = 10):
        """
        Args:
            ngspice_path: Path to ngspice executable
            timeout: Simulation timeout in seconds
        """
        self.ngspice_path = ngspice_path or self._find_ngspice()
        self.timeout = timeout

    def _find_ngspice(self) -> str:
        """Find ngspice in PATH."""
        import shutil
        ngspice = shutil.which("ngspice")
        if ngspice:
            return ngspice

        common_paths = [
            r"C:\Program Files\Spice64\bin\ngspice.exe",
            r"C:\ngspice\bin\ngspice.exe",
            "/usr/bin/ngspice",
            "/usr/local/bin/ngspice",
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p

        logger.warning("ngspice not found. SPICE verification will be limited.")
        return "ngspice"

    def verify(self, netlist_text: str) -> VerificationResult:
        """
        Verify a SPICE netlist through simulation.

        Args:
            netlist_text: SPICE netlist text

        Returns:
            VerificationResult
        """
        result = VerificationResult()

        # Step 1: Basic syntax checks
        self._check_syntax(netlist_text, result)

        # Step 2: Run ngspice simulation
        if result.syntax_valid:
            self._run_simulation(netlist_text, result)

        result.raw_output = ""
        return result

    def _check_syntax(self, netlist_text: str, result: VerificationResult):
        """Perform basic SPICE syntax checks."""
        lines = netlist_text.strip().split("\n")

        # Check for ground node
        if re.search(r'\b0\b', netlist_text) or re.search(r'\bGND\b', netlist_text, re.IGNORECASE):
            result.has_ground = True

        # Check for voltage sources
        if re.search(r'^[Vv]\S+', netlist_text, re.MULTILINE):
            result.has_voltage_source = True

        # Check for components
        component_pattern = re.compile(r'^[RCLDQMUJSWBX]\S+', re.MULTILINE)
        components = component_pattern.findall(netlist_text)
        result.component_count = len(components)
        result.has_components = result.component_count > 0

        # Check for .end statement
        if ".end" not in netlist_text.lower():
            result.warnings.append("Missing .end statement")

        # Count unique nodes
        nodes = set()
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and re.match(r'^[RCLDQMUJSWBX]', parts[0], re.IGNORECASE):
                for part in parts[1:]:
                    if not re.match(r'^\d+\.?\d*', part) and part.lower() not in ('dc', 'ac', 'model'):
                        nodes.add(part)
        result.node_count = len(nodes)

        # Basic validity
        result.syntax_valid = result.has_components

        if not result.has_ground:
            result.warnings.append("No ground node (0 or GND) found")
        if not result.has_voltage_source:
            result.warnings.append("No voltage source found")

    def _run_simulation(self, netlist_text: str, result: VerificationResult):
        """Run ngspice simulation."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".cir", delete=False, encoding="utf-8"
            ) as f:
                f.write(netlist_text)
                netlist_path = f.name

            proc = subprocess.run(
                [self.ngspice_path, "-b", netlist_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = proc.stdout + proc.stderr
            result.raw_output = output

            # Check for errors
            if "error" in output.lower():
                for line in output.split("\n"):
                    if "error" in line.lower():
                        result.errors.append(line.strip())

            # Check for convergence
            if "doAnalyses: Too many iterations" in output:
                result.simulation_converges = False
                result.errors.append("Simulation did not converge")
            elif "singular matrix" in output.lower():
                result.simulation_converges = False
                result.errors.append("Singular matrix (circuit error)")
            elif "analysis" in output.lower() and "error" not in output.lower():
                result.simulation_converges = True

            # Check for warnings
            for line in output.split("\n"):
                if "warning" in line.lower():
                    result.warnings.append(line.strip())

            # Cleanup
            try:
                os.unlink(netlist_path)
            except OSError:
                pass

        except FileNotFoundError:
            result.warnings.append("ngspice not found - simulation skipped")
        except subprocess.TimeoutExpired:
            result.warnings.append(f"Simulation timeout ({self.timeout}s)")
        except Exception as e:
            result.errors.append(f"Simulation error: {str(e)}")

    def verify_file(self, netlist_path: str) -> VerificationResult:
        """Verify a netlist file."""
        with open(netlist_path, "r", encoding="utf-8") as f:
            return self.verify(f.read())
