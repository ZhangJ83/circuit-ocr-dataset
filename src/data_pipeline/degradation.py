"""
Schematic Degradation Pipeline (Differentiation B)
===================================================
Simulate 5 types of real-world image degradation for circuit schematics:

1. paper_aging:     Paper yellowing, spots, foxing marks
2. scan_noise:      Gaussian noise + horizontal scan lines
3. perspective_distortion: Camera perspective warp (photo of paper)
4. handwriting_overlay: Handwritten annotations/notes
5. low_resolution:  Downscale + upscale (low DPI scan)

All existing work (Image2Net, CircuitVision, etc.) only tests on clean
digital renders. This degradation pipeline is a UNIQUE differentiator.
"""

import random
import math
import logging
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SchematicDegradation:
    """Apply realistic degradation effects to circuit schematic images."""

    # Degradation types
    TYPES = [
        "paper_aging",
        "scan_noise",
        "perspective_distortion",
        "handwriting_overlay",
        "low_resolution",
    ]

    @staticmethod
    def paper_aging(
        image: "PIL.Image.Image",
        severity: float = 0.5,
        seed: Optional[int] = None,
    ) -> "PIL.Image.Image":
        """
        Simulate paper aging: yellowing, foxing spots, and discoloration.

        Args:
            image: Input PIL image
            severity: 0.0 (none) to 1.0 (extreme)
            seed: Random seed for reproducibility

        Returns:
            Degraded image
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        from PIL import Image

        img = np.array(image).astype(np.float32)
        h, w = img.shape[:2]

        # 1. Color shift (yellowing)
        yellow_shift = 0.15 * severity
        img[:, :, 0] = img[:, :, 0] * (1 + yellow_shift * 0.3)  # Slight red
        img[:, :, 1] = img[:, :, 1] * (1 + yellow_shift * 0.1)  # Slight green
        img[:, :, 2] = img[:, :, 2] * (1 - yellow_shift * 0.5)  # Reduce blue

        # 2. Foxing spots (brown spots)
        num_spots = int(20 * severity * random.uniform(0.5, 1.5))
        for _ in range(num_spots):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            radius = random.randint(3, int(30 * severity))

            y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x_grid**2 + y_grid**2 <= radius**2

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and dx**2 + dy**2 <= radius**2:
                        intensity = random.uniform(0.1, 0.4) * severity
                        img[ny, nx, 0] = img[ny, nx, 0] * (1 + intensity * 0.5)
                        img[ny, nx, 1] = img[ny, nx, 1] * (1 - intensity * 0.2)
                        img[ny, nx, 2] = img[ny, nx, 2] * (1 - intensity * 0.6)

        # 3. Vignetting (darker edges)
        center_x, center_y = w / 2, h / 2
        max_dist = math.sqrt(center_x**2 + center_y**2)
        for y in range(h):
            for x in range(w):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                darken = 1.0 - (dist / max_dist) * 0.3 * severity
                img[y, x, :] *= darken

        # 4. Random discoloration patches
        num_patches = int(5 * severity)
        for _ in range(num_patches):
            px = random.randint(0, w - 50)
            py = random.randint(0, h - 50)
            pw = random.randint(20, 80)
            ph = random.randint(20, 80)
            patch_color = np.array([
                random.uniform(-10, 20),
                random.uniform(-10, 10),
                random.uniform(-30, -5)
            ]) * severity
            img[py:py+ph, px:px+pw, :] += patch_color

        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    @staticmethod
    def scan_noise(
        image: "PIL.Image.Image",
        severity: float = 0.5,
        seed: Optional[int] = None,
    ) -> "PIL.Image.Image":
        """
        Simulate scanner noise: Gaussian noise + scan lines.

        Args:
            image: Input PIL image
            severity: 0.0 (none) to 1.0 (extreme)
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        from PIL import Image

        img = np.array(image).astype(np.float32)
        h, w = img.shape[:2]

        # 1. Gaussian noise
        noise = np.random.normal(0, 15 * severity, img.shape)
        img += noise

        # 2. Horizontal scan lines (lighter bands)
        line_spacing = random.randint(3, 8)
        line_intensity = 0.3 * severity
        for y in range(0, h, line_spacing):
            img[y:y+1, :, :] *= (1 - line_intensity)

        # 3. Vertical streaks (scanner artifact)
        num_streaks = int(3 * severity)
        for _ in range(num_streaks):
            sx = random.randint(0, w - 1)
            streak_width = random.randint(1, 3)
            intensity = random.uniform(0.05, 0.2) * severity
            img[:, sx:sx+streak_width, :] *= (1 - intensity)

        # 4. Salt-and-pepper noise
        num_sp = int(h * w * 0.001 * severity)
        for _ in range(num_sp):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            img[y, x, :] = random.choice([0, 255])

        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    @staticmethod
    def perspective_distortion(
        image: "PIL.Image.Image",
        severity: float = 0.5,
        seed: Optional[int] = None,
    ) -> "PIL.Image.Image":
        """
        Simulate perspective distortion from photographing a paper.

        Args:
            image: Input PIL image
            severity: 0.0 (none) to 1.0 (extreme)
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        import cv2

        w, h = image.size
        magnitude = int(min(w, h) * 0.05 * severity)

        # Source corners
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Destination corners (random perspective)
        dst_pts = np.float32([
            [random.randint(0, magnitude), random.randint(0, magnitude)],
            [w - random.randint(0, magnitude), random.randint(0, magnitude)],
            [w - random.randint(0, magnitude), h - random.randint(0, magnitude)],
            [random.randint(0, magnitude), h - random.randint(0, magnitude)],
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(
            np.array(image), M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Add slight rotation for realism
        angle = random.uniform(-3, 3) * severity
        center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, R, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

        from PIL import Image
        return Image.fromarray(img)

    @staticmethod
    def handwriting_overlay(
        image: "PIL.Image.Image",
        severity: float = 0.3,
        seed: Optional[int] = None,
    ) -> "PIL.Image.Image":
        """
        Overlay handwritten annotations and marks.

        Args:
            image: Input PIL image
            severity: 0.0 (none) to 1.0 (extreme)
            seed: Random seed
        """
        if seed is not None:
            random.seed(seed)

        from PIL import Image, ImageDraw, ImageFont

        img = image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Handwriting-like marks
        marks = [
            "?", "OK", "✓", "✗", "×", "!", "←", "→", "↑", "↓",
            "注意", "检查", "修", "断开", "短路", "R?", "C?",
            "0.1uF", "10k", "100nF", "test", "NC", "DNP",
        ]

        num_marks = int(5 * severity * random.uniform(0.5, 2.0))
        ink_colors = [(0, 0, 180), (0, 0, 200), (180, 0, 0), (0, 100, 0)]

        for _ in range(num_marks):
            x = random.randint(20, w - 80)
            y = random.randint(20, h - 30)
            mark = random.choice(marks)
            color = random.choice(ink_colors)
            size = random.randint(12, 28)

            try:
                # Try to use a handwriting-like font
                font = ImageFont.truetype("comic.ttf", size)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("arial.ttf", size)
                except (OSError, IOError):
                    font = ImageFont.load_default()

            # Slight rotation for handwriting effect
            draw.text((x, y), mark, fill=color, font=font)

        # Draw some circles/arrows (annotation marks)
        num_circles = int(3 * severity)
        for _ in range(num_circles):
            cx = random.randint(50, w - 50)
            cy = random.randint(50, h - 50)
            r = random.randint(20, 60)
            color = random.choice(ink_colors)
            draw.ellipse(
                [(cx-r, cy-r), (cx+r, cy+r)],
                outline=color, width=2
            )

        # Draw some underlines
        num_underlines = int(2 * severity)
        for _ in range(num_underlines):
            x1 = random.randint(20, w - 100)
            y1 = random.randint(20, h - 20)
            x2 = x1 + random.randint(30, 100)
            color = random.choice(ink_colors)
            draw.line([(x1, y1), (x2, y1)], fill=color, width=2)

        return img

    @staticmethod
    def low_resolution(
        image: "PIL.Image.Image",
        target_dpi: int = 150,
        severity: float = 0.5,
        seed: Optional[int] = None,
    ) -> "PIL.Image.Image":
        """
        Simulate low-resolution scanning.

        Args:
            image: Input PIL image
            target_dpi: Target DPI (lower = more degraded)
            severity: 0.0 (none) to 1.0 (extreme)
            seed: Random seed
        """
        from PIL import Image

        if seed is not None:
            random.seed(seed)

        w, h = image.size

        # Adjust target DPI based on severity
        effective_dpi = int(300 - (300 - target_dpi) * severity)
        scale = effective_dpi / 300.0

        # Downscale
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        small = image.resize((new_w, new_h), Image.BILINEAR)

        # Add slight blur before upscaling (simulates optical blur)
        from PIL import ImageFilter
        small = small.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Upscale back with interpolation artifacts
        result = small.resize((w, h), Image.NEAREST)

        return result

    @classmethod
    def apply_random_degradation(
        cls,
        image: "PIL.Image.Image",
        severity_range: Tuple[float, float] = (0.2, 0.8),
        num_degradations: int = 2,
        seed: Optional[int] = None,
    ) -> Tuple["PIL.Image.Image", List[str]]:
        """
        Apply a random combination of degradation effects.

        Args:
            image: Input image
            severity_range: Range for random severity
            num_degradations: Number of degradation types to apply
            seed: Random seed

        Returns:
            (degraded_image, list_of_applied_degradation_names)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Select random degradation types
        selected = random.sample(cls.TYPES, min(num_degradations, len(cls.TYPES)))

        result = image.copy()
        applied = []

        for deg_type in selected:
            severity = random.uniform(*severity_range)

            if deg_type == "paper_aging":
                result = cls.paper_aging(result, severity)
            elif deg_type == "scan_noise":
                result = cls.scan_noise(result, severity)
            elif deg_type == "perspective_distortion":
                result = cls.perspective_distortion(result, severity)
            elif deg_type == "handwriting_overlay":
                result = cls.handwriting_overlay(result, severity)
            elif deg_type == "low_resolution":
                result = cls.low_resolution(result, severity=severity)

            applied.append(deg_type)

        return result, applied

    @classmethod
    def generate_degraded_variants(
        cls,
        image_path: str,
        output_dir: str,
        num_variants: int = 5,
        base_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate multiple degraded variants of a single image.

        Args:
            image_path: Path to original clean image
            output_dir: Directory for degraded images
            num_variants: Number of variants to generate
            base_name: Base filename (defaults to stem of image_path)

        Returns:
            List of variant info dicts
        """
        from PIL import Image

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if base_name is None:
            base_name = image_path.stem

        img = Image.open(image_path).convert("RGB")
        variants = []

        for i in range(num_variants):
            try:
                severity = random.uniform(0.2, 0.8)
                n_deg = random.randint(1, 3)

                # Apply random degradation
                degraded, applied = cls.apply_random_degradation(
                    img, severity_range=(severity, severity),
                    num_degradations=n_deg,
                )

                # Save
                variant_name = f"{base_name}_deg{i:03d}"
                out_path = output_dir / f"{variant_name}.png"
                degraded.save(str(out_path), "PNG", dpi=(300, 300))

                variants.append({
                    "path": str(out_path),
                    "degradations": applied,
                    "severity": severity,
                    "variant_index": i,
                })

            except Exception as e:
                logger.error(f"Failed to generate variant {i}: {e}")

        logger.info(
            f"Generated {len(variants)} degraded variants for {image_path.name}"
        )
        return variants


if __name__ == "__main__":
    import argparse
    from PIL import Image

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Apply degradation to schematics")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("--output-dir", default="data/degraded")
    parser.add_argument("--variants", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        variants = SchematicDegradation.generate_degraded_variants(
            str(input_path), args.output_dir, args.variants
        )
        for v in variants:
            print(f"  {v['path']}: {v['degradations']}")
    elif input_path.is_dir():
        for img_file in input_path.glob("*.png"):
            SchematicDegradation.generate_degraded_variants(
                str(img_file), args.output_dir, args.variants
            )
