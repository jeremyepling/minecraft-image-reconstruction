"""Test measured wall-grid extension on the six original no-floor composite captures."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import relative_face_round as round_, floating_cloud_replay as paths
from benchmark.expanded_baseline import ROOT, WORKTREES, read, save, digest
from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.floating_horizontal_bridges import horizontal_charts, solver
from engine.surface_perimeter import _native_alpha


def run(case):
    cv2.setNumThreads(1)
    root = WORKTREES / "floating-visible-builds/runs/floating-cloud-gate-01"
    out = ROOT / "runs/six-sky"
    name = case["id"]
    path = Path(case["variants"]["sky"]["image"])
    packet = read(root/"garden-anchor-extension/connected/sky"/name/"charts.json")
    handoff = root/"sky"/name/"source-handoff"
    paths.check_handoff(packet,handoff,digest(path))
    planes = {p["id"]:p for p in read(handoff/"surfaces.json")["planes"]}
    rgb = np.asarray(Image.open(path).convert("RGB"))
    output = []
    for chart in packet["charts"]:
        if not chart["cells"] or abs(chart["normal_camera"][1]) >= .67:
            output.append(chart)
            continue
        alpha,(x,y,x1,y1) = _native_alpha(planes[chart["id"]]["support"],rgb.shape[1::-1])
        mask = np.zeros(rgb.shape[:2],bool)
        mask[y:y1,x:x1] = np.asarray(alpha)>0
        points = np.column_stack(np.nonzero(mask)[::-1])+.5
        measured = resample(chart,np.asarray(chart["H"]),mask,rgb,points)
        output.append(preserve_and_rank(chart,measured,rgb,mask))
    save(out/"charts/sky"/name/"charts.json",dict(packet,charts=output))
    original_orient=solver.oriented
    def with_horizontal(charts):
        walls,hint=original_orient(charts)
        add=horizontal_charts(walls,list(planes.values()),rgb.shape[1::-1],rgb)
        return walls+add,hint
    for arm,charts in (("ordinary",root/"garden-anchor-extension/connected"),("whole-wall",out/"charts"),("horizontal-cells",out/"charts")):
        mode = (charts,"charts.json",False,1,2.5,.5,False)
        with patch.object(solver,"oriented",with_horizontal if arm=="horizontal-cells" else original_orient),patch.object(paths,"RUN",root),patch.object(round_,"ASSEMBLY",out/arm),patch.dict(
                round_.MODES,{"connected-atlas":mode}):
            round_.evaluate((case,"sky","connected-atlas"))


if __name__=="__main__":
    with ProcessPoolExecutor(2) as pool:
        list(pool.map(run,read(round_.SHARED/"fixtures.json")["cases"]))
