"""Count the truth composition of L-shaped roof proposals without exporting unsupported cells."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark import framed_depth_eval as framed
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.visible_recovery_report import summarize
from engine import visible_roof_rectangle as rectangle
from engine.thin_atlas_faces import StripEvidence
import numpy as np


def score(job):
    split,frame,inputs,control,root = job
    key = frame['id']
    local = set(decoded_pack(checked(root/split/key/'reconstruction.zip',control['pack_sha256'])))
    image,parts,_,camera = framed.photo(split,frame,inputs)
    raw = set()
    note = dict(upper_blocks=0, geometric_candidates=0, edge0=0, edge2=0, corners=0)
    if local and camera.get('P') is not None:
        P = np.asarray(camera['P'],float)
        eye = np.linalg.solve(P[:,:3],-P[:,3])
        upper = {c for c in local if 0 < c[1] < eye[1]-1 and (c[0],c[1]+1,c[2]) not in local}
        geometric,_ = rectangle.rects({0:upper,2:upper},eye,local)
        note.update(upper_blocks=len(upper),geometric_candidates=len(geometric),eye_height=float(eye[1]))
        if geometric:
            atlas = rectangle._atlas('26.3')
            assert atlas is not None and atlas.provenance['atlas_sha256']==rectangle.ATLAS_SHA256
            ctx = StripEvidence(image,parts,camera,local,atlas)
            edges = rectangle.observed_edges(ctx,upper)
            raw,corners = rectangle.rects(edges,eye,local)
            note.update(edge0=len(edges[0]),edge2=len(edges[2]),corners=corners)
    ref_split='old' if split in ('extra','new') else split
    truth,occupied,T,_=framed.references(ref_split,frame,framed.NATIVE_INPUTS.get(split),framed.TRAINING_DATA,
        framed.DEPTH,dict(native=framed.native.BUILDS,old=framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS,
                          fresh=framed.transfer.BUILDS))
    before=registered(local,T)
    measured=placement_scores(before|registered(raw,T),before,dict.fromkeys(truth),occupied)
    score=measured['relative_geometry']
    assert list(measured['baseline_translation'])==control['translation']
    assert score['correct']-score['gained']+score['lost']==control['relative_geometry']['correct']
    return dict(id=key,split=split,source=control['rgb_sha256'],control_pack_sha256=control['pack_sha256'],
                unexported_rectangle_cells=len(raw),**note,**measured)


def main(args):
    meta=read(args.control/'summary.json')
    rows=read(checked(args.control/'cases.json',meta['cases_sha256']))
    control={(r['split'],r['id']):r for r in rows}
    jobs=[(s,f,h,control[s,f['id']],args.control) for s,f,h in framed.cases(framed.SIZES)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result=list(pool.map(score,jobs))
    groups={s:[r for r in result if r['split']==s] for s in framed.SIZES}
    groups['fixed']=[r for r in result if r['split'] in framed.EVALUATION]
    out={s:summarize(rr) for s,rr in groups.items()}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save(args.out,dict(control_sha256=meta['cases_sha256'],adapter_sha256=digest(Path(rectangle.__file__)),
                       groups=out,rows=result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    main(p.parse_args())
