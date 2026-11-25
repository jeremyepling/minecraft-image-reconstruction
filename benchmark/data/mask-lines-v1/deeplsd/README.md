# DeepLSD line evidence

Each compressed file holds the endpoints detected on one source screenshot by the official MegaDepth checkpoint. The 56 frame IDs map to 52 distinct source PNG hashes. `manifest.json` gives the mapping, artifact hashes, software revisions and timings. The input list and wrapper use only screenshot paths, hashes and frame identifiers. Reference geometry is outside the inference path.

The runner resizes each full 1440 × 900 frame to 960 × 600 using OpenCV area resampling. Padding is zero on all four sides. Each `(x, y)` output uses the center of the top-left model pixel as its origin. The pinned PyTLSD implementation states this convention and comments out the classic half-pixel gradient correction. Our native screenshot geometry uses the upper-left **edge** of that pixel as its origin. The conversion is `(model_xy + 0.5) * (native_size / resized_size)` with separate horizontal and vertical ratios. On these frames a model coordinate of `(0, 0)` maps to `(0.75, 0.75)` native pixels. [Pinned PyTLSD coordinate convention](https://github.com/iago-suarez/pytlsd/blob/21381ca4c5c8da297ef44ce07dda115075c5a648/src/lsd.cpp#L2093-L2102)

DeepLSD predicts line segments. Each recorded junction is the midpoint of two segment endpoints within five native pixels, where the segments meet at an angle of at least 20 degrees. This derivation uses only the predicted segments. The manifest reports elapsed inference time separately from model loading and cached geometry processing. [Official pretrained model and inference instructions](https://github.com/cvg/DeepLSD/tree/f7d9d6258c0cd25d4f6eea882853565403d289be)

The [Linux workflow](../../../../.github/workflows/mask-deeplsd.yml) pins the inference packages, downloads the checkpoint and runs the wrapper. Once those dependencies are set up, regenerate the complete cache from the repository root with

```sh
python -m benchmark.mask_deeplsd --upstream runs/upstream-deeplsd --checkpoint runs/deeplsd_md.tar
```
