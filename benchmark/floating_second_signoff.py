"""Build the second-round grid signoff from actual candidate grids and decoded exports."""
import json
from pathlib import Path
from benchmark.floating_line_shear import OUT, RUN as SHEAR
from benchmark.floating_grid_signoff import CASES, CROSS, ISOLATION, PART, WORKTREES, get_chart, make, read
from benchmark.floating_visible_row import RUN as OLD_ROW
from benchmark.floating_stair_strips import RUN as STAIRS
from benchmark.floating_bridges_report import read_arm
from benchmark.floating_extra_budget import LOCATION

RULER=WORKTREES/"floating-majority-edge/runs/floating-majority-edge"
REFIT=WORKTREES/"floating-depth-refit/benchmark/reviews/floating-depth-refit-01"

NOTES=[
 "Keep B on this wall. Each of the eight cells is now 98–100% supported by visible brick, and its top edge tracks the sandstone seam. The eight corrected cells are still on a wall omitted by the final assembly.",
 "B gives the projecting Mill region its own local ruler while preserving the surrounding grid. It ties the previous depth-aware build here. Check the local window cross and porch separately.",
 "B also separates three small brown base regions. The far-left fourth base still lacks any parent grid cells. Keeping the old spacing on those regions produced the same placement gain; assigning them a separate depth explains the gain.",
 "Keep B on Castle. The source depth correctly ordered all seven non-flat boundaries we checked. Only 12 of 21 near-seam contacts supported the forced one-block move. Withdrawing it restores 14 correct parapet blocks. This is a conservative correction; it still leaves other depth layers unresolved.",
 "B finds eight visible bands and samples 51 cells across them. The assembler exports the original wall alone and drops all eight bands. Correct placements fall from 60 to 41 and extras from 64 to 19. Check which strips follow an actual stair front and where a tread or adjoining wall needs another orientation. I would keep the right-wall evidence in A.",
 "B preserves the local scale at the center and reorients the lines using the building outlines. Principal-plane coverage rises from 77.1% to 94.6%, and median grid-line error falls from 0.29 to 0.24 cell. I would keep the size and still adjust alignment. The exported build does not change because it selects another wall.",
 "B adjusts the slope and origin using one set of source lines and checks the fit on other lines. The export gains two net correct positions and removes one extra. Check the left corner and whether alignment holds across material changes.",
]


def option(folder, rel, wall):
    return ([get_chart(folder,rel,wall)],{})


def depth_options(source, rec, rel, wall, guarded=False):
    chart=get_chart(source,rel,wall)
    parts=[p for p in rec["patches"] if p["chart"]==wall]
    extra=[p for p in rec["extra_patches"] if p["chart"]==wall]
    flatten=set()
    if guarded:
        flatten={tuple(k) for p in parts if not p["boundary_guard"]["accept"] for k in p["old_uv"]}
        parts=[p for p in parts if p["boundary_guard"]["accept"]]
        if not parts:
            extra=[]
    selected={tuple(key) for p in parts+extra for key in p["old_uv"]}
    main=dict(chart,cells=[dict(c,normal_offset=0) if (c["u"],c["v"]) in flatten else c for c in chart["cells"] if (c["u"],c["v"]) not in selected])
    children=[]
    for p in parts+extra:
        src=p.get("source-refit",p)
        children.append(dict(H=src["H"],cells=[dict(u=u,v=v,normal_offset=p["depth"]) for u,v in src["uv"]]))
    return [([chart],{}),([main,*children],{})]


def main():
    refits=read(REFIT/"conditions.json.gz")
    rows=[]
    for idx,(kind,panel,name,wall,title,_) in enumerate([*CASES,("depth","holdout","train-arcbridge2-cool-high-right","mask-0","Arcbridge · boundary check regression","")]):
        rel=Path("sky")/name/"charts.json"
        base=ISOLATION/panel/"large-contraction"
        if idx==0:
            before=OLD_ROW/panel/"row-with-material"
            after=RULER/panel/"seam"
            charts=[option(before,rel,wall),option(after,rel,wall)]
            scores=[read_arm(f/"assembly"/LOCATION/name)[0] for f in (before,after)]
        elif idx in (1,2,3,7):
            rec=next(r for r in refits if r["panel"]==panel and r["id"]==name)
            source=PART/panel/"source"
            charts=depth_options(source,rec,rel,wall,guarded=idx in (3,7))
            scores=[rec["score"][arm] for arm in ("source-depth","signed-boundary-guard" if idx in (3,7) else "source-refit-with-small-steps")]
        elif idx==4:
            after=STAIRS/panel
            step_charts=[c for c in read(after/"charts"/rel)["charts"] if c["id"]==wall or c["id"].startswith(wall+"-step-")]
            charts=[option(base,rel,wall),(step_charts,{})]
            scores=[read_arm(f/"assembly"/LOCATION/name)[0] for f in (base,after)]
        else:
            method="outline-fit" if idx==5 else "line-fit"
            before=CROSS/panel if idx==5 else base
            after=SHEAR/panel/method
            charts=[option(before,rel,wall),option(after,rel,wall)]
            scores=[read_arm(f/"assembly"/LOCATION/name)[0] for f in (before,after)]
        spec=(kind,panel,name,wall,title,NOTES[idx] if idx<7 else "Keep A here. The same boundary check that fixes Castle loses nine correct positions and adds nine extras on this view. Which visible rim or connection argues for retaining the depth step?")
        rows.append(make(spec,dict(charts=charts,scores=scores)))
    template=Path(__file__).with_name("floating_grid_signoff.html").read_text(encoding="utf-8")
    template=template.replace("Sizing measures grid cells per full reference face;", "For independently fitted patches, alignment and size are measured on visible reference faces whose centers fall inside a rendered grid. Sizing measures grid cells per full reference face;").replace("floating-grid-signoff-02","floating-grid-signoff-03").replace(
        "floating-grid-signoff-feedback.json","floating-grid-round-two-feedback.json").replace(
        "My proposed adjustment","My assessment of the tested change").replace(
        "My proposed adjustment".lower(),"My assessment of the tested change".lower()).replace(
        "A · Current grid","A · Comparison grid").replace("B · Tested alternative","B · Implemented adjustment")
    (OUT/"review.html").write_text(template.replace("__PAYLOAD__",json.dumps(rows)),encoding="utf-8")
    print([(r["title"],r["options"][1]["metrics"],r["options"][1]["score"]) for r in rows])


if __name__=="__main__":
    main()
