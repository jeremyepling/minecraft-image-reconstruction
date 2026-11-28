"""Optional post-join UV filtering for frozen Floating packs; ordinary assembly stays untouched."""
from functools import wraps
from unittest.mock import patch
import numpy as np
from benchmark import relative_face_assembly as solver


def with_visible_export(base):
    @wraps(base)
    def assemble(charts,**settings):
        if not any("export_uv" in c for c in charts):
            return base(charts,**settings)
        original_orient=solver.oriented
        by_id={}
        def record(source):
            oriented,hint=original_orient(source)
            by_id.update({c["id"]:c for c in oriented})
            # Establish joins and component ranking using the complete measured charts.
            return [{k:v for k,v in c.items() if k!="export_uv"} for c in oriented],hint
        with patch.object(solver,"oriented",record):
            result=base(charts,**settings)
        for group in result["groups"]:
            blocks={}
            assigned=set()
            ordered=sorted(group["origins"],key=lambda cid:by_id[cid].get("confidence",0.),reverse=True)
            for cid in ordered:
                chart=by_id[cid]
                a,b=[solver.axis_vector(v) for v in chart["axes"]]
                n=solver.axis_vector(chart["normal_axis"])
                offset=np.asarray(group["origins"][cid])+np.minimum(0,a)+np.minimum(0,b)-np.maximum(0,n)
                allowed=set(map(tuple,chart["export_uv"])) if "export_uv" in chart else None
                for cell in chart["cells"]:
                    uv=cell["u"],cell["v"]
                    if allowed is not None and uv not in allowed:
                        continue
                    xyz=tuple((offset+uv[0]*a+uv[1]*b).tolist())
                    if xyz not in blocks:
                        blocks[xyz]=cell["block_id"]
                        if not cell.get("fallback"):
                            assigned.add(xyz)
            group.update(blocks=blocks,assigned=assigned)
        return result
    return assemble
