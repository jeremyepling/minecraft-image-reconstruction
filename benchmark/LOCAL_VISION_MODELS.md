# Local vision models for visible Minecraft geometry

These candidates accept one RGB frame. They address different measurements in the visible-surface benchmark. Train and select any adaptation on development layouts. Reserve new underlying layouts for the next confirmation.

| Candidate | Local download | Proposed test |
| --- | --- | --- |
| NAVER M-LSD Tiny 512 FP16 | 1.28 MB TensorFlow Lite checkpoint, CPU inference | Feed its line endpoints into the existing junction and plane assembly. Measure plane matches, openings covered and native-pixel corner errors. The network can also pick up a horizon, clouds, texture joints and palette changes. [Authors' project](https://github.com/navervision/mlsd) · [Checkpoint](https://huggingface.co/spaces/navervision/MLSD/blob/main/tflite_models/M-LSD_512_tiny_fp16.tflite) |
| Depth Anything V2 Small | 24.8 million parameters, 27.3 MB for the UINT8 ONNX conversion | Compare relative depth across a proposed edge. A consistent near-to-far step is evidence of occlusion. Score the ordering of a pillar, opening and rear wall. Recover block units from the texture pass. [Authors' model](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) · [Conversion files](https://huggingface.co/onnx-community/depth-anything-v2-small/tree/main/onnx) |
| MoGe-2 ViT-S with normals | 35 million parameters, 141 MB PyTorch checkpoint | Test whether front, side and underside pixels fall into three mutually perpendicular normal directions. The full pipeline also estimates camera intrinsics and a point map. Compare its estimated focal length and per-plane depth against the existing camera-space lift. [Authors' implementation](https://github.com/microsoft/MoGe) · [Checkpoint](https://huggingface.co/Ruicheng/moge-2-vits-normal/tree/main) |

A contrast line can come from a material boundary, a shadow, a change in surface orientation or an object ending. A normal change across it supports a fold. A depth jump plus a line ending at a T-junction supports an occlusion. A shadow on a flat wall should preserve the fitted normal and plane depth. Repetition within that plane can establish block spacing even when its outer outline is uncertain. These are hypotheses for one-stage-at-a-time tests on the paired palettes.

## What ran on this PC

All three models ran locally on the same 1440 × 900 development screenshot of the masonry shopfront on light gray concrete. Each predictor saw only pixels. Recorded visible faces and camera were read for the subsequent diagnostic. These measurements describe one layout from one angle. [Visual comparison](reviews/local-models-01/preview.jpg) · [Checkpoint hashes, settings and measurements](reviews/local-models-01/measurements.json)

| Probe | Observed result | First CPU inference |
| --- | --- | ---: |
| M-LSD Tiny | 54 segments. Within the building and a five-pixel exterior band, 74.7% of its line pixels were within five pixels of a physical boundary. It covered 66.9% of true boundary pixels. The existing line detector scored 49.2% and 88.3% respectively. | 0.107 s, one thread |
| Depth Anything V2 Small, UINT8 | Correct nearer/farther order for 8 of 8 eligible pairs of adjacent visible surfaces. Evaluated at recorded depth jumps of at least 0.75 blocks, with at least four samples per pair. Units of the map are relative inverse depth. | 0.738 s, four threads |
| MoGe-2 Small, ONNX normals | Median signed orientation error of 7.3° over 10,338 visible building samples. The front wall had 6.4° median error, while downward-facing surfaces had 36.8° median error over 275 samples. | 1.05 s, four threads |

For a first geometry challenger, keep the existing contrast-line proposals and use MoGe normals to score the two sides of each proposed edge. Fit dominant Minecraft axis directions across the image, look for consistent normals inside each patch, and compare patch matches and the recall of small soffits against the frozen default. In a later one-stage comparison, use relative depth at T-junctions to test which surface continues behind which. The absolute scale measurement remains the building-face texture pitch.

### Offline probes

The optional Python modules accept local checkpoints and PNGs. M-LSD used `ai-edge-litert==2.4.0`. The depth and normal probes used `onnxruntime==1.23.2` with its CPU provider. The UINT8 depth export worked on this PC. The signed INT8 graph requested a ConvInteger implementation absent from this runtime. The published round requires only NumPy and Pillow.

```console
python -m benchmark.mlsd_probe --checkpoint M-LSD_512_tiny_fp16.tflite --image screenshot.png --out runs/local-lines
python -m benchmark.depth_probe --model model_uint8.onnx --image screenshot.png --out runs/local-depth
python -m benchmark.moge_probe --checkpoint model.onnx --image screenshot.png --out runs/local-normals
```

The source and model names in each measurement provide a way to verify the downloaded files. [Exact line weights](https://huggingface.co/spaces/navervision/MLSD/blob/main/tflite_models/M-LSD_512_tiny_fp16.tflite) · [UINT8 depth export](https://huggingface.co/onnx-community/depth-anything-v2-small/blob/34f40a4c93991327074907b51792dbc27e7d6f1c/onnx/model_uint8.onnx) · [Authors' ONNX normals export](https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/blob/e50ffda41565591092adea54c6ac83d6212e1e23/model.onnx)

## Running the full MoGe-2 geometry pipeline

After installing the manufacturer's Python package in its own environment, the following upstream CLI selects the small checkpoint explicitly. It produces maps and a visible-surface mesh. Omit the field of view for the ordinary run. The upstream `--fov_x` option takes **horizontal** degrees for the known-camera diagnostic. [Installation and inference](https://github.com/microsoft/MoGe) · [CLI options](https://github.com/microsoft/MoGe/blob/main/moge/scripts/infer.py)

```console
moge infer --version v2 --pretrained Ruicheng/moge-2-vits-normal --device cpu -i screenshot.png -o output --maps --glb
```

The authors also provide an ONNX forward pass. It produces intermediate points, normals, a mask and a scale value. Reproducing the calibrated point cloud requires the focal-length and shift recovery and reprojection in their full inference code. The normals alone can be tested directly. [Authors' ONNX instructions](https://github.com/microsoft/MoGe/blob/main/docs/onnx.md)
