"""
Schematic Renderer
==================
Render KiCad .kicad_sch files to PNG images using:
1. kicad-cli SVG export (primary)
2. cairosvg SVG → PNG conversion
3. Optional: PIL-based synthetic rendering (fallback)

Supports configurable DPI (150/300/600) and coordinate calibration.
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SchematicRenderer:
    """Render KiCad schematics to PNG images."""

    # KiCad SVG export produces mm-based coordinates
    # Standard conversion: 1 inch = 25.4 mm
    # At 300 DPI: 1 mm = 300/25.4 ≈ 11.811 pixels
    DPI_MM_TO_PX = {
        150: 150.0 / 25.4,   # ≈ 5.906 px/mm
        300: 300.0 / 25.4,   # ≈ 11.811 px/mm
        600: 600.0 / 25.4,   # ≈ 23.622 px/mm
    }

    def __init__(
        self,
        output_dir: str,
        dpi: int = 300,
        kicad_cli_path: Optional[str] = None,
        max_workers: int = 4,
    ):
        """
        Args:
            output_dir: Directory for rendered PNG files
            dpi: Output resolution (150, 300, or 600)
            kicad_cli_path: Path to kicad-cli executable
            max_workers: Parallel rendering workers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.mm_to_px = self.DPI_MM_TO_PX.get(dpi, 300.0 / 25.4)
        self.kicad_cli = kicad_cli_path or self._find_kicad_cli()
        self.max_workers = max_workers

    def _find_kicad_cli(self) -> str:
        """Find kicad-cli in PATH."""
        import shutil
        cli = shutil.which("kicad-cli")
        if cli:
            return cli

        # Common install locations
        common_paths = [
            r"C:\Program Files\KiCad\8.0\bin\kicad-cli.exe",
            r"C:\Program Files\KiCad\7.0\bin\kicad-cli.exe",
            "/usr/bin/kicad-cli",
            "/usr/local/bin/kicad-cli",
            "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p

        logger.warning("kicad-cli not found. SVG export will not be available.")
        return "kicad-cli"

    def render_svg(self, sch_path: str, output_svg: Optional[str] = None) -> Optional[str]:
        """
        Export schematic to SVG using kicad-cli.

        Args:
            sch_path: Path to .kicad_sch file
            output_svg: Output SVG path (auto-generated if None)

        Returns:
            Path to generated SVG, or None on failure
        """
        sch_path = Path(sch_path)
        if not sch_path.exists():
            logger.error(f"Schematic file not found: {sch_path}")
            return None

        if output_svg is None:
            output_svg = str(self.output_dir / f"{sch_path.stem}.svg")

        try:
            cmd = [
                self.kicad_cli, "sch", "export", "svg",
                "--output", output_svg,
                "--page-size-mode", "0",  # Bounding box mode
                str(sch_path),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                logger.warning(
                    f"SVG export failed for {sch_path.name}: {result.stderr[:200]}"
                )
                return None

            if os.path.exists(output_svg):
                return output_svg

            # kicad-cli might create files with different naming
            svg_dir = Path(output_svg).parent
            for f in svg_dir.glob(f"{sch_path.stem}*.svg"):
                return str(f)

            return None

        except FileNotFoundError:
            logger.error(f"kicad-cli not found at: {self.kicad_cli}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"SVG export timeout for {sch_path.name}")
            return None
        except Exception as e:
            logger.error(f"SVG export error: {e}")
            return None

    def svg_to_png(
        self,
        svg_path: str,
        output_png: Optional[str] = None,
        scale: Optional[float] = None,
    ) -> Optional[str]:
        """
        Convert SVG to PNG using cairosvg.

        Args:
            svg_path: Path to input SVG
            output_png: Output PNG path
            scale: Scale factor (overrides DPI)

        Returns:
            Path to generated PNG, or None on failure
        """
        svg_path = Path(svg_path)
        if not svg_path.exists():
            logger.error(f"SVG file not found: {svg_path}")
            return None

        if output_png is None:
            output_png = str(self.output_dir / f"{svg_path.stem}.png")

        try:
            import cairosvg

            dpi = self.dpi if scale is None else 300
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=output_png,
                dpi=dpi,
                scale=scale or 1.0,
            )

            if os.path.exists(output_png):
                return output_png
            return None

        except ImportError:
            logger.error("cairosvg not installed. Install with: pip install cairosvg")
            return None
        except Exception as e:
            logger.warning(f"SVG→PNG conversion failed: {e}")
            return None

    def render_schematic(
        self,
        sch_path: str,
        output_png: Optional[str] = None,
    ) -> Optional[str]:
        """
        Full render pipeline: .kicad_sch → SVG → PNG.

        Args:
            sch_path: Path to .kicad_sch file
            output_png: Final PNG output path

        Returns:
            Path to rendered PNG, or None on failure
        """
        sch_path = Path(sch_path)
        if output_png is None:
            output_png = str(self.output_dir / f"{sch_path.stem}.png")

        # Step 1: Export to SVG
        svg_path = self.render_svg(str(sch_path))
        if not svg_path:
            logger.warning(f"SVG export failed, trying PIL fallback for {sch_path.name}")
            return self._render_with_pil(sch_path, output_png)

        # Step 2: Convert SVG to PNG
        png_path = self.svg_to_png(svg_path, output_png)
        if not png_path:
            logger.warning(f"SVG→PNG failed, trying PIL fallback for {sch_path.name}")
            return self._render_with_pil(sch_path, output_png)

        # Cleanup intermediate SVG
        try:
            os.remove(svg_path)
        except OSError:
            pass

        return png_path

    def _render_with_pil(self, sch_path: Path, output_png: str) -> Optional[str]:
        """
        Fallback rendering using PIL.
        Creates a simple visualization from parsed schematic data.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            from .kicad_parser import KiCadParser

            parser = KiCadParser()
            data = parser.parse(str(sch_path))

            # Calculate canvas size
            width_mm, height_mm = data.sheet_size
            width_px = int(width_mm * self.mm_to_px)
            height_px = int(height_mm * self.mm_to_px)

            # Create image
            img = Image.new("RGB", (width_px, height_px), "white")
            draw = ImageDraw.Draw(img)

            # Draw grid
            grid_spacing = int(2.54 * self.mm_to_px)  # 100mil grid
            for x in range(0, width_px, grid_spacing):
                draw.line([(x, 0), (x, height_px)], fill="#F0F0F0", width=1)
            for y in range(0, height_px, grid_spacing):
                draw.line([(0, y), (width_px, y)], fill="#F0F0F0", width=1)

            def mm_to_px(x_mm: float, y_mm: float) -> Tuple[int, int]:
                return (int(x_mm * self.mm_to_px), int(y_mm * self.mm_to_px))

            # Draw wires
            for wire in data.wires:
                x1, y1 = mm_to_px(wire.start.x, wire.start.y)
                x2, y2 = mm_to_px(wire.end.x, wire.end.y)
                draw.line([(x1, y1), (x2, y2)], fill="#000000", width=2)

            # Draw component placeholders
            for comp in data.components:
                cx, cy = mm_to_px(comp.position.x, comp.position.y)
                size = int(5 * self.mm_to_px)
                draw.rectangle(
                    [(cx - size, cy - size), (cx + size, cy + size)],
                    outline="#0000CC", width=2
                )
                # Draw reference text
                try:
                    font = ImageFont.truetype("arial.ttf", max(12, size // 2))
                except (OSError, IOError):
                    font = ImageFont.load_default()
                draw.text((cx - size, cy - size - 15), comp.reference,
                          fill="#0000CC", font=font)

            # Draw labels
            for label in data.labels:
                lx, ly = mm_to_px(label.position.x, label.position.y)
                try:
                    font = ImageFont.truetype("arial.ttf", max(10, int(3 * self.mm_to_px)))
                except (OSError, IOError):
                    font = ImageFont.load_default()
                draw.text((lx, ly), label.name, fill="#CC0000", font=font)

            # Draw junctions
            for junc in data.junctions:
                jx, jy = mm_to_px(junc.position.x, junc.position.y)
                r = max(2, int(0.5 * self.mm_to_px))
                draw.ellipse([(jx-r, jy-r), (jx+r, jy+r)], fill="#000000")

            # Draw no-connects
            for nc in data.no_connects:
                nx, ny = mm_to_px(nc.position.x, nc.position.y)
                s = max(3, int(1 * self.mm_to_px))
                draw.line([(nx-s, ny-s), (nx+s, ny+s)], fill="#CC0000", width=2)
                draw.line([(nx-s, ny+s), (nx+s, ny-s)], fill="#CC0000", width=2)

            img.save(output_png)
            logger.info(f"PIL fallback render: {output_png}")
            return output_png

        except Exception as e:
            logger.error(f"PIL fallback rendering failed: {e}")
            return None

    def batch_render(
        self,
        sch_paths: List[str],
        progress_callback=None,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Render multiple schematics in parallel.

        Args:
            sch_paths: List of .kicad_sch file paths
            progress_callback: Optional callback(completed, total)

        Returns:
            List of (sch_path, png_path or None) tuples
        """
        results = []
        total = len(sch_paths)

        if self.max_workers <= 1:
            for i, sch_path in enumerate(sch_paths):
                png_path = self.render_schematic(sch_path)
                results.append((sch_path, png_path))
                if progress_callback:
                    progress_callback(i + 1, total)
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.render_schematic, sp): sp
                    for sp in sch_paths
                }

                completed = 0
                for future in as_completed(futures):
                    sch_path = futures[future]
                    try:
                        png_path = future.result(timeout=120)
                        results.append((sch_path, png_path))
                    except Exception as e:
                        logger.error(f"Batch render error for {sch_path}: {e}")
                        results.append((sch_path, None))

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

        success = sum(1 for _, p in results if p is not None)
        logger.info(f"Batch render complete: {success}/{total} succeeded")
        return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Render KiCad schematics to PNG")
    parser.add_argument("input", help="Input .kicad_sch file or directory")
    parser.add_argument("--output-dir", default="data/rendered")
    parser.add_argument("--dpi", type=int, default=300, choices=[150, 300, 600])
    args = parser.parse_args()

    renderer = SchematicRenderer(output_dir=args.output_dir, dpi=args.dpi)

    input_path = Path(args.input)
    if input_path.is_file():
        result = renderer.render_schematic(str(input_path))
        print(f"Rendered: {result}")
    elif input_path.is_dir():
        sch_files = list(input_path.rglob("*.kicad_sch"))
        results = renderer.batch_render([str(f) for f in sch_files])
        for sch, png in results:
            print(f"  {Path(sch).name}: {'✓' if png else '✗'}")
