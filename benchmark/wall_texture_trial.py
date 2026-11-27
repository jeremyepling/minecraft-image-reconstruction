"""Re-run the Block Counter and ordinary placer on image-only surface mask changes."""
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark.texture_camera_eval import STABLE
from benchmark.wall_texture_masks import rewrite
from engine import atlas_plane_depth, surface_model, texture_guided_count


def run(image,parts,counts,camera,prior,mode):
    regions,edits=rewrite(image,parts,camera,mode)
    if not edits:return prior,dict(edits=[])
    cells,detail=rebuild(image,counts,camera,regions)
    return cells,dict(edits=edits,**detail)


def rebuild(image,counts,camera,regions):
    by_id={c['id']:c for c in counts['surfaces']}
    created=[p for p in regions if p['id'] not in by_id]
    if created:
        method=texture_guided_count.METHOD if any(c['method']==texture_guided_count.METHOD for c in by_id.values()) else 'masked-repeat'
        # Repeat the source counter's saved configuration on modified masks only.
        atlas_path=STABLE/'runs/26.3/atlas'
        atlas=None
        if 'atlas_phase' in counts:
            from engine.atlas_count import load_atlas
            atlas=load_atlas(atlas_path,'26.3')
        with TemporaryDirectory() as tmp:
            new=surface_model._count_regions(image,created,'26.3',method,atlas,Path(tmp))
        by_id.update((c['id'],c) for c in new['surfaces'])
    next_counts=dict(counts,surfaces=[by_id[p['id']] for p in regions])
    cells,_=atlas_plane_depth.place_ranked(image,'26.3',regions,next_counts,camera,atlas_path=STABLE/'runs/26.3/atlas')
    return cells,dict(recomputed=[dict(id=p['id'],counts=by_id[p['id']]['counts']) for p in created])


def split(image,parts,counts,camera,prior):return run(image,parts,counts,camera,prior,'split')
def join(image,parts,counts,camera,prior):return run(image,parts,counts,camera,prior,'join')
def shade(image,parts,counts,camera,prior):return run(image,parts,counts,camera,prior,'shade')
def prune(image,parts,counts,camera,prior):return run(image,parts,counts,camera,prior,'prune')


def selective(image,parts,counts,camera,prior):
    return run(image,parts,counts,camera,prior,'selective')
