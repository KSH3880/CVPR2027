#!/usr/bin/env python3
"""Generate an exact legend for the Isaac Gym coordinator viewer colors."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "coord_viewer_speed_colors.png"
WIDTH, HEIGHT = 1400, 900
BG = (19, 23, 31)
FG = (239, 243, 250)
MUTED = (169, 180, 197)
AGENTS = (
    ("Agent 0 bottom path ribbon", (1.00, 0.15, 0.65)),
    ("Agent 1 bottom path ribbon", (1.00, 0.55, 0.10)),
)
SPEEDS = (0.375, 0.750, 1.125, 1.500)


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def rgb(base, speed):
    scale = speed / 1.5
    shade = 0.28 + 0.72 * scale
    return tuple(round(255 * channel * shade) for channel in base)


image = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(image)
draw.text((70, 50), "TokenHSI-coord viewer speed colors", font=font(48, True), fill=FG)
draw.text(
    (70, 116),
    "Bottom path brightness encodes planned speed continuously from 0.375 to 1.500 m/s.",
    font=font(25), fill=MUTED,
)
draw.text(
    (70, 154),
    "The four swatches below are reference levels; the C2 model is not quantized to four classes.",
    font=font(23), fill=MUTED,
)

for row, (label, base) in enumerate(AGENTS):
    y = 235 + row * 230
    draw.text((70, y), label, font=font(30, True), fill=FG)
    for index, speed in enumerate(SPEEDS):
        x = 70 + index * 320
        color = rgb(base, speed)
        draw.rounded_rectangle((x, y + 58, x + 270, y + 142), radius=13, fill=color)
        draw.text((x, y + 154), f"{speed:.3f} m/s", font=font(25, True), fill=FG)
        draw.text((x, y + 188), f"RGB {color}", font=font(20), fill=MUTED)

draw.line((70, 696, 1330, 696), fill=(63, 72, 87), width=2)
draw.text((70, 726), "Waist-high active steer window (length = actual rate-limited command)", font=font(27, True), fill=FG)
fixed = (
    ("Agent 0 window", (26, 242, 255)),
    ("Agent 1 window", (89, 255, 89)),
    ("Aim marker", (255, 230, 26)),
    ("K=1 candidate centerline", (89, 89, 89)),
)
for index, (label, color) in enumerate(fixed):
    x = 70 + index * 320
    draw.rounded_rectangle((x, 779, x + 58, 837), radius=8, fill=color)
    draw.text((x + 72, 785), label, font=font(18, True), fill=FG)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT)
print(OUTPUT)
