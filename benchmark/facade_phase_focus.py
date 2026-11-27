"""Decode a real per-view export and count restored block positions on each visible reference surface."""
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark import framed_depth_eval as framed
from benchmark.facade_phase_evidence import REVIEW, common
from benchmark.mask_blocks import _write_read,decoded_pack,registered
from benchmark.relative_geometry import _aligned
from benchmark.texture_camera_eval import checked,read,save,digest
from engine.facade_inset_phase import refine
from training.visible_labels import visible_labels


def run():
    s,frame,h,base=next(j for j in common.jobs(['native']) if j[1]['id']==common.FOCUS)
    image,parts,counts,camera=common.ordinary_inputs(s,frame,h,base)
    final,events=refine(image,parts,camera)
    cells,dims=common.build(image,final,counts,camera)
    old=checked(common.CONTROL/s/frame['id']/'reconstruction.zip',base['pack_sha256'])
    rgb=framed.native.rgb_path(frame,framed.native.BUILDS)
    pose=read(checked(rgb.with_suffix('.json'),frame['files']['camera.json']))
    states=read(checked(rgb.with_name(rgb.stem+'-blocks.json'),frame['files']['world-readback.json']))
    labels,_,_=visible_labels(pose,{tuple(b[k] for k in 'xyz'):b for b in states},(0,0,0))
    target,_,transform,_=framed.references(s,frame,None,framed.TRAINING_DATA,framed.DEPTH,dict(native=framed.native.BUILDS))
    original,t0=_aligned(registered(set(decoded_pack(old)),transform),target)
    folder=Path('runs/focus-proof')
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/'reconstruction.zip'
    _write_read(out,cells)
    current,t1=_aligned(registered(set(decoded_pack(out)),transform),target)
    group=defaultdict(set)
    for f in labels['faces']:
        group[f['surface_id']].add(tuple(f['xyz_world']))
    results={sid:dict(visible=len(coords),old=len(coords&original),new=len(coords&current),
                restored=sorted(coords&(current-original)),lost=sorted(coords&(original-current))) for sid,coords in group.items()}
    save(folder/'surfaces.json',dict(image_size=image.size,planes=final))
    save(folder/'dimensions.json',dims)
    zipfile=REVIEW/'focus-export.zip'
    with ZipFile(zipfile,'w',ZIP_DEFLATED) as z:
        for name in ('reconstruction.zip','surfaces.json','dimensions.json'):z.write(folder/name,name)
    with ZipFile(zipfile) as z:assert z.testzip() is None
    save(REVIEW/'focus.json',dict(id=frame['id'],rgb_sha256=frame['files']['rgb.png'],original_pack_sha256=digest(old),
        event=events,readback=results,old_shift=t0,new_shift=t1,source_archive_sha256=digest(zipfile),
        surface_sha256=digest(folder/'surfaces.json'),new_pack_sha256=digest(out),
        restored=sorted((current-original)&target),lost=sorted((original-current)&target),
        extra_added=sorted((current-original)-target),extra_removed=sorted((original-current)-target)))
    print([(s,results[s]['old'],results[s]['new']) for s in (9,14,19)])


if __name__=='__main__':
    run()
