
# Assembling visible faces into a build-local grid

Keep this experiment separate from the ordinary placer. It exports signed relative block coordinates. The assembler has no ground plane, world position, whole-image camera model, or camera-to-build distance. One visible chart corner is the coordinate origin. A local chart maps unit-grid corners on one visible surface to source pixels.

## What changed

1. Measure a unit grid and visible cell support independently on each face. Group the wall directions from the estimated surface normals. Local 2D perspective fits show how individual surfaces appear in the source image. The denser atlas challenger uses a nominal focal-length prior to straighten a wall. It supplies no solved camera or distance to assembly.
2. For perpendicular faces, require projected unit edges to agree in position, length and orientation. This supplies an integer offset between their local grids. Group two faces into the same block when the resulting signed cell coordinates agree. Cycle checks reject contradictory offsets.
3. Test whether two separated charts belong to one plane. Their unit grids must agree in two dimensions and predict image points on both charts after a single integer shift. Agreement joins those charts while preserving the visible gap. A candidate inverse-depth check can flag an apparent junction when depths jump across the edge, in model units.
4. Export the largest group whose offsets were supported by the source. Charts outside the group stay unplaced. Exported Minecraft blocks require at least one observed face. No supports, shells, or gap fills are generated.

## Six matched open-sky views

The same four Visible Builds v3 and two Decorative Builds v1 source images from the paired backdrop round are used here. The synthetic sky excludes pavement and the horizon. The original structure pixels and source-view occlusion carry through. Three views look upward. This is a six-image exploration. For exact chart controls, per-plane cell coordinates, materials and signed axes come from the recorded visible labels. The generator independently changes each plane's tangent origin, omits its world offset and supplies native-pixel support for edge visibility. It supplies no camera matrix to assembly. Controls omit 76 decorative non-full-cube visible owners from their inputs and score all 4,041 visible target positions.

| Method | Joined views | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extras |
| --- | ---: | ---: | ---: | ---: |
| Ordinary default, archived | 0/6 | 0.00% / 0.00% | 0.00% / 0.00% | 0 / 4,041 / 0 |
| Previous native wall fallback, archived | 0/6 | 2.30% / 2.25% | 1.71% / 1.75% | 47 / 3,994 / 0 |
| Image only, verified atlas charts | 0/6 | 1.72% / 1.75% | 1.08% / 1.10% | 35 / 4,006 / 0 |
| Image only, denser atlas charts | 0/6 | 2.35% / 2.40% | 1.37% / 1.40% | 48 / 3,993 / 3 |
| Image only, exploratory counter charts | 0/6 | 6.42% / 6.55% | 0.00% / 0.00% | 134 / 3,907 / 0 |
| Exact charts, shared edges | 6/6 | 77.58% / 77.36% | 77.58% / 77.36% | 2,561 / 1,480 / 0 |
| Exact charts, shared edges and same-plane alignment | 6/6 | 93.71% / 93.70% | 93.71% / 93.70% | 3,563 / 478 / 0 |

Scoring allows one integer translation for the whole build. For the source-only predictions the evaluator fixes their compass axes by comparing a single source-predicted wall direction to the recorded viewpoint before computing placement overlap. The source predictor runs before the scorer loads the recorded inputs. The exact control uses the supplied face axes and one translation. Correct material labels are supplied to that control. Exact edge matching uses a 0.01 native-pixel tolerance. The broader source threshold cannot be validated by the exact-chart number.

The denser atlas input produced 14 charts, with 62 of 83 chart cells consistent under a per-chart diagnostic. It yielded zero usable inter-chart joins. The unverified counter picked a large Castle wall and placed 127 positions correctly with fallback materials. Its measured chart corners often failed to trace exact block edges. The optional relative inverse-depth continuity check saw zero eligible source edge matches in either source arm, so it had no effect. Its value for judging occlusion remains unmeasured on these cases.

The edges-only exact control found 594 of 1,471 visible Castle positions and 74 of 311 for Harbor. With verified same-plane alignment those counts reach 1,257 and 242. Viaduct reached 1,071 of 1,091 using shared visible edges alone. Components without a supported placement relationship remain unplaced.

## Cohort status

| Sky subset | Tested / available | Dense atlas placement | Exploratory counter placement | Exact charts with same-plane joins |
| --- | ---: | ---: | ---: | ---: |
| Visible Builds v3 | 4/144 | 2.48% / 2.51% | 7.68% / 7.74% | 94.83% / 94.86% |
| Decorative Builds v1 | 2/40 | 1.77% / 1.91% | 1.01% / 1.10% | 88.81% / 88.22% |

The fixed 104 carries forward from the unchanged default, with 91.27% strict and 93.42% facade-priority placement, 20,773 correct, 2,929 missed and 1,046 extras. This prototype was not tested on those 104 views.

## Replay

Use the frozen manifests in the stable checkout's runs/floating-build-grid-01. The camera-free source charts are on experiment/relative-face-source-charts and the labeled chart generator is on experiment/relative-face-junction-control. The source charts describe visible cell support, image homographies and estimated surface normals. From this checkout run python -m benchmark.relative_face_round atlas atlas-dense atlas-dense-depth count-exploratory count-depth-veto exact-edges exact-edges-coplanar. Run python -m benchmark.relative_face_report to verify input and pack hashes and rebuild the review image. The local examples and score receipts are under runs/relative-face-assembly-01.

Keep experiment/relative-face-assembly experimental. The diagnostic shows that a set of reliable visible grids can recover most relative positions with an arbitrary build origin. The source-only measurements have insufficient coverage and agreement at visible junctions. The next milestone is two independently measured surfaces registering to a single integer corner, followed by testing same-plane agreement and using ordinal depth to distinguish a contact from an overlap.
