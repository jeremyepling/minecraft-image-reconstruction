# Replaying the region and line comparison

Run the full experiment from this checkout in PowerShell.

```powershell
.\Invoke-MaskRound.ps1
```

The command prints a fresh output directory under `runs/`. Open its `comparison.jpg` for the visual comparison and `report.html` for the numbers. Each building expands into its individual pictures, with a link to the original image. A building total counts its surfaces each time they appear in a picture. The full report also distinguishes block coordinates decoded from an exported pack from the diagnostic that supplies the actual camera and planes.

The full round compares the archived B predictor, regions from MoGe-2 directions and Depth Anything V2, and the same regions refined with ScaleLSD v2 or DeepLSD. The two line models' numerical outputs for these screenshots are frozen in `benchmark/data/mask-lines-v1`. Replaying the comparison reads those files and reruns the refinement and scoring. Recorded line inference time and fresh processing time have separate labels in the report.

For the original two-method, 20-picture development check, use `.\Invoke-MaskRound.ps1 -Phase first`. All inputs come from the saved Java 26.3 suite. The command sets image-only mode in the runner and its child processes.

### Cache and environment

The wrapper looks for a compatible numerical direction/depth cache in `runs/normal-fields`, then in the sibling `normal-round/runs/normal-fields` checkout. A complete compatible cache is read in place. A partial cache contributes its compatible entries to a new run-specific directory. The missing fields are generated from the original screenshots using the suite's frozen ONNX models. The existing model preparation helper uses local checked files when available and otherwise retrieves the pinned Hugging Face releases and verifies their SHA-256 hashes.

The published numerical recipe uses Python with NumPy 2.3.5, Pillow 13.0.0 and ONNX Runtime 1.23.2 for fresh inference. The wrapper first tries the project virtual environment, the local Codex Python and the system Python. With Python 3.11 or newer and compatible public wheels, it can install missing libraries in a run-specific environment. A complete compatible cache needs only NumPy and Pillow. Fresh downloads require network access.

Use `-Python`, `-Fields`, `-Lines` and `-Out` to select an interpreter, numerical cache, frozen line-output directory or new output directory. Relative paths are resolved from the checkout. Use a fresh output directory to keep each run.

```powershell
.\Invoke-MaskRound.ps1 -Fields ..\normal-round\runs\normal-fields -Out runs\boundary-review-02
```
