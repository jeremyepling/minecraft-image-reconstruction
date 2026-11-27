# Workshop foreground and roof placement

In the high-right plaster Workshop view the two long, thin outlines near the horizon were background. They contributed no blocks to the final export. A separate proposed surface above the spruce awning had a greater placement cost. The solver placed it one block above the real roof, where it hid valid roof candidates during projection. Of that proposal's 2,361 source pixels, the reference labels 2,360 as background. The exterior check removed 2,330 background pixels and retained the boundary where the evidence was ambiguous. [Inspect Workshop on the source image](../shared-face-round-01/viewer.html#stage=corners&view=legacy-light_gray_concrete-holdout-workshop-plaster-high-right)

The default uses two independent cues. First it checks which source colors continue uninterrupted to the frame margins on the same image rows. It protects upright surfaces and horizontal surfaces with evidence of an adjoining wall. The separate horizon check follows the sky-to-ground transition and preserves real roof pixels at the end of the line. The original scene support remains available for camera and plane fitting. Public structural surfaces, crops and counts use the cleaned supports.

For placement, a ground-level platform is protected. An elevated cube can be removed when its whole visible footprint lies on unclaimed exterior background. If a mostly background shelf borders a single measured roof, the placer also tests the observed roof pixels on the roof's fitted plane. New cells need visible top-face support when the revised build is projected into the source image. The camera must be above the proposed shelf. [Placement and surface rules](../../../engine/exterior_foreground.py) · [Platform, separate structure and camera controls](../../../tests/test_exterior_foreground.py) · [Horizon comparisons](../workshop-horizon-01/README.md)

## Placement

| Scope | Previous F1 | Current F1 | Correct | Missed | Extra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Workshop plaster high-right, strict and facade-priority | 85.25% | 91.42% | 229 | 36 | 7 |
| Fixed 104, strict | 91.1887% | 91.2678% | 20,773 | 2,929 | 1,046 |
| Fixed 104, facade-priority | 93.3307% | 93.4194% | 18,952 | 1,624 | 1,046 |

Across the 204 saved views the default retained every previously correct coordinate, recovered 18 and removed 144 extras. Of those extras, 18 were in the fixed suite and 126 were in the 100 additional controls. The camera-registered strict score is 91.2678%. The historical 112-view strict score is 90.5043%. Across both surface refinements 46,103 unique background pixels were removed and zero labeled build pixels were removed. Fifteen wholly false surface proposals disappeared. [Per-view positions and hashes](cases.json) · [Scored exports](default-exports.zip) · [Pooled scores](summary.json) · [Combined surface results](../workshop-horizon-01/combined-summary.json)

Fresh runs of the ordinary placement command reproduced the scored coordinates and material IDs on all ten changed views and four controls. The controls include Shopfront, Galleries, Portico and a real platform. On the Workshop plaster high-right view, the seven remaining extras project primarily onto spruce (four), white concrete (two) and background (one). [Fourteen ordinary exports checked against scored block sets](combined-ordinary.json)

## Planks in the other layouts

The previous Courtyard problem came from two counted wall faces whose independent depth estimates disagreed at a shared corner. Workshop, Galleries, Shopfront and Portico have a different pattern. Across the four representative views, none of the 21 extras projecting mostly over spruce came from a wrong locked texture-crop seed. Shopfront and Portico primarily extended a real underside too far. Galleries also has stray proposals at different depths. Removing the undersides or shifting their height sacrificed real blocks. The next targeted test is to compare only the ends of individual underside rows against their visible rim and adjoining face. [Four-layout audit and ablations](../workshop-horizon-01/plank-audit.md)

The broadest foreground alternative kept only the largest connected region. It would discard 123 correctly placed blocks across the 16 Workshop, Galleries and Shopfront views. The shipped checks preserve separated build pieces and use positive background evidence. [Foreground alternatives](foreground-trials.json)

## Replay and viewer

Run `python -m benchmark.exterior_foreground_release` for placement and `python -m benchmark.workshop_horizon_eval --combined` for combined visible-mask scoring. Regenerate the viewer with `python -m benchmark.current_placement_viewer`. Choose **Face corner placement** and select **Workshop**, **Galleries**, **Shopfront**, **Portico**, or **Courtyard**. The layer controls compare the source surface edges and the visible corners of the decoded export. When available, additional layers show the two counted corner fits or excluded background. The overlay opacity slider applies to the active layers. [Ordinary build evidence](ordinary.json)
