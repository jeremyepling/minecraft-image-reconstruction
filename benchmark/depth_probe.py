"""Run a local Depth Anything V2 Small relative-depth ONNX checkpoint on one RGB frame.

The processor configuration is from onnx-community/depth-anything-v2-small at
34f40a4c93991327074907b51792dbc27e7d6f1c. The CPU probe used onnx/model_uint8.onnx.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.atlas import digest


def preprocess(image):
    # DPTImageProcessor fits the dimension requiring the smallest scale change,
    # then rounds each side to a multiple of the ViT patch size (14 pixels).
    scale = min((518 / side for side in image.size), key=lambda value: abs(1 - value))
    size = tuple(max(14, 14 * round(side * scale / 14)) for side in image.size)
    rgb = np.array(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255
    rgb -= np.array([.485, .456, .406], dtype=np.float32)
    rgb /= np.array([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def comparison(image, depth):
    size = (min(image.width, 720), round(image.height * min(image.width, 720) / image.width))
    low, high = np.percentile(depth, (1, 99))
    values = np.clip((depth - low) / max(1.e-9, high - low), 0, 1)
    # A monotonic dark-to-light color ramp for relative inverse depth.
    stops = np.array([[18, 24, 44], [48, 73, 119], [42, 130, 155], [111, 194, 173], [254, 240, 167]])
    colors = np.stack([np.interp(values, np.linspace(0, 1, len(stops)), channel)
                       for channel in stops.T], axis=-1).astype(np.uint8)
    sheet = Image.new("RGB", (2*size[0], size[1]+40), (24, 30, 40))
    sheet.paste(image.resize(size, Image.Resampling.LANCZOS), (0, 40))
    sheet.paste(Image.fromarray(colors).resize(size, Image.Resampling.BICUBIC), (size[0], 40))
    pen, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=18)
    pen.text((12, 9), "Minecraft source", fill="white", font=font)
    pen.text((size[0]+12, 9), "Relative depth   |   lighter is nearer", fill="white", font=font)
    return sheet


def run(model, source, out, threads=4):
    import onnxruntime as ort

    if threads < 1:
        raise ValueError("Thread count must be positive")
    with Image.open(source) as original:
        image = original.convert("RGB")
    tensor = preprocess(image)
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = threads
    settings.inter_op_num_threads = 1
    begin = time.perf_counter()
    session = ort.InferenceSession(str(model), sess_options=settings, providers=["CPUExecutionProvider"])
    loaded = time.perf_counter()
    predicted, = session.run(["predicted_depth"], {"pixel_values": tensor})
    elapsed = time.perf_counter() - loaded
    if predicted.ndim != 3 or predicted.shape[0] != 1 or not np.isfinite(predicted).all():
        raise ValueError(f"Expected one finite 2D depth map, got {predicted.shape}")
    depth = predicted[0]
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "relative-depth.npy", depth)
    comparison(image, depth).save(out / "comparison.png")
    report = dict(model=model.name, model_sha256=digest(model), bytes=model.stat().st_size,
                  source=source.name, source_sha256=digest(source), source_size=image.size,
                  runtime=ort.__version__, provider=session.get_providers(), intra_op_threads=threads,
                  input_shape=list(tensor.shape), output_shape=list(predicted.shape), output_pixels=depth.size,
                  output_range=[float(depth.min()), float(depth.max())], output_units="relative inverse depth",
                  session_load_seconds=round(loaded-begin, 4), one_cpu_inference_seconds=round(elapsed, 4),
                  display_percentiles=[1, 99], artifacts=["relative-depth.npy", "comparison.png"])
    (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.model, args.image, args.out, args.threads)
