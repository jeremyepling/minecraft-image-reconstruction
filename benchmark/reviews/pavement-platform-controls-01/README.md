# Would a real platform survive the pavement check?

**The placement-only rule preserved the tested platforms. The rule that discards a whole surface failed on a bordered flush floor.** The controls use the saved right-hand Colonnade 01 source image, the ordinary estimated camera and the same detected post and sill masks as the original opening. They exercise the frozen pavement check from `experiment/pavement-opening-grid` and the local block check from `experiment/pavement-local-placement` against the default at `a6e0195`. [Controlled source comparisons](platform-comparison.jpg) · [Per-control inputs, texture scores and outcomes](controls.json)

| Input behind the measured rim | Phase check identifies pavement | Effect on placed blocks |
| --- | --- | --- |
| Actual Colonnade opening | Yes | Removed seven of the eight phantom shelf blocks and kept every correct coordinate. |
| Top of a raised platform in the opening, rendered with the same pavement texture three blocks above the ground | No | Retained 60 of 60 visible platform cells. Overriding the phase check would erase 35, showing that the platform presented a real deletion opportunity. |
| Same-pavement floor flush with the ground, bordered with a contrasting one-block-wide trim | Yes | Retained all 320 supplied floor and trim cells. Removed seven original shelf extras in front. Discarding the detected surface would erase the mask on 5,396 real floor pixels. |
| Separate 3 × 3 flush pad in the unobstructed foreground | No | Retained all nine cells, including when its mask was forced through to the coordinate check. |
| Original opening mask spilling onto 15 real foreground pixels along its border | No | Every previously correct neighboring block remained in the coordinate map. |

The controlled raised top uses the source pavement's native texture estimated outside the structure and reprojects its individual texels onto an elevated plane. The finite flush control covers 16 × 20 blocks on the inferred ground plane. Its one-block perimeter has a contrasting color. In the source image, 1,066 perimeter pixels are visible, including 60 within the original opening mask. A further 2,736 interior pixels are visible beyond that mask. The floor's measured texture inside the aperture genuinely continues the ground grid. The rule can use this evidence to remove elevated obstructions while keeping the floor eligible for placement.

The finite floor is an authored counterfactual with known grid coordinates. Only the parts visible through or outside the Colonnade were rendered. Its 320 supplied cells test deletion behavior on a partly occluded fixture. Their count measures retention of supplied coordinates, not accuracy of reconstruction from RGB. The raised control instead supplies the 60 cells that contribute pixels to the target aperture plus the original correctly placed Colonnade cells.

## Native platform images

The recorded fixture geometry names the Loading Dock's offset platform, the Platform Shelter's two raised bays, and the Switchback's stair flights, landings and lookout. These six source images were captured in Minecraft and their placement labels enter only after the candidate has evaluated the image. Counts below are visible upward block faces or source pixels summed over the two angles. The same block can appear in both views. The method preserved every previously covered platform pixel.

| Structure, two views each | Visible platform top faces | Visible pixels | Pixels covered by the original masks | Previously covered pixels lost |
| --- | ---: | ---: | ---: | ---: |
| Switchback | 102 | 36,517 | 35,762 | 0 |
| Loading Dock | 117 | 30,919 | 24,635 | 0 |
| Platform Shelter | 98 | 15,200 | 14,690 | 0 |

Seven original masks containing real platform pixels reached the ground-texture check across five of the six views. None passed its final opening threshold. The remaining named tops were present for evaluation even when the finder had no usable mask. These controls measure directly on the native source images. [Capture and authored parts](../blind-shapes-01/README.md) · [Additional recorded shelters](../framed-depth-01/README.md)

## Replay

Use the registered fixed inputs and candidate modules identified by SHA-256 in `controls.json`. When both modules are installed in the current checkout, run the following command.

```powershell
python -m benchmark.pavement_platform_controls
```

For worktree development, select each module's checkout explicitly.

```powershell
python -m benchmark.pavement_platform_controls --candidate-root ../pavement-opening-grid --local-root ../pavement-local-placement
```
