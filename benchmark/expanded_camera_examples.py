"""Show the costliest camera rejection in each new dataset beside a supported view of the same build."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import SOURCE, digest, read


def font(size):
    for family in ("segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(family, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pair(cases, dataset):
    rows = [r for r in cases if r["dataset"] == dataset]
    failed = sorted((r for r in rows if not r["camera_usable"]),
                    key=lambda r: (-r["required_truth"], r["id"]))
    if not failed:
        return []
    # Select by cost first, then prefer the same palette when a companion view is available.
    for miss in failed:
        matches = [r for r in rows if r["layout"] == miss["layout"] and r["camera_usable"]]
        if matches:
            best = max(matches, key=lambda r: (common_prefix(miss["id"], r["id"]), r["id"]))
            return [miss, best]
    return failed[:1]


def common_prefix(a, b):
    for index, (one, two) in enumerate(zip(a.split("-"), b.split("-"))):
        if one != two:
            return index
    return min(len(a.split("-")), len(b.split("-")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--review", type=Path, default=Path(__file__).parent / "reviews/expanded-baseline-01")
    args = parser.parse_args()
    cases = read(args.review / "cases.json")
    width, height, gap = 600, 375, 16
    tile = height + 95
    sheet = Image.new("RGB", (width * 2 + gap * 3, 2 * tile + gap * 3), "#18252a")
    draw = ImageDraw.Draw(sheet)
    for index, dataset in enumerate(("visible-builds-v3", "decorative-builds-v1")):
        for column, row in enumerate(pair(cases, dataset)):
            x, y = gap + column * (width + gap), gap + index * (tile + gap)
            frame = args.source / dataset / "data" / row["split"] / "frames" / row["id"] / "rgb.png"
            if digest(frame) != row["source_sha256"]:
                raise ValueError(f"Source differs from scored case: {row['id']}")
            with Image.open(frame) as image:
                sheet.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (x, y))
            y += height + 3
            supported = row["camera_usable"]
            tag = "Camera supported" if supported else "Camera failure"
            color = "#a7eabe" if supported else "#ffb4a4"
            draw.text((x, y), f"{tag}  ·  {row['id']}", font=font(16), fill=color)
            draw.text((x, y + 27), row["camera_reason"] or "Reason unavailable", font=font(15), fill="#eef2f0")
            counts = row["geometry"]["strict"]
            draw.text((x, y + 52), f"{counts['correct']:,} correct    {counts['missed']:,} missed    "
                      f"{counts['extras']:,} extra", font=font(16), fill="#eef2f0")
    sheet.save(args.review / "camera-examples.jpg", quality=89, optimize=True)


if __name__ == "__main__":
    main()
