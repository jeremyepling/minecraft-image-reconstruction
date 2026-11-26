"""Answer-assisted diagnostics on development data only; never a production adapter."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from PIL import Image
from benchmark.canopy_depth_choices import DEST
from benchmark.texture_camera_eval import DATA,read,save
from benchmark.visible_depth_transfer import OUT,rgb_path
from engine.atlas_plane_depth import _place
from engine.plane_repeat_evidence import select_repeat
from engine.atlas_plane_depth import frozen_weights


def index():
    result={}
    native={r['id']:r for r in read(OUT/'screen.json')['selected']}
    for path in DEST.glob('*/*.json'):
        row=read(path)
        png=(rgb_path(native[row['id']]) if row['split']=='transfer' else
             DATA/'data'/row['split']/'frames'/row['id']/'rgb.png')
        with Image.open(png) as photo:key=sha256(photo.convert('RGB').tobytes()).hexdigest()
        result[key]=path.relative_to(DEST).as_posix()
    save(DEST/'rgb-index.json',result)


@lru_cache(maxsize=1)
def manifest():
    return read(DEST/'rgb-index.json')


def run(image,version,parts,counts,camera,method):
    key=sha256(image.tobytes()).hexdigest()
    frame=read(DEST/manifest()[key])
    rows={r['id']:r for r in frame['rows']}
    def choose(part,mask,P,eye,rays,axis,baseline,rgb,atlas,scene,weights):
        row=rows[part['id']]
        assert (row['axis'],row['baseline'])==(axis,baseline)
        opts=row['choices']
        if method.startswith('oof'):
            models=read(Path(__file__).parent/'reviews/canopy-plane-chooser-01/development.json')
            return select_repeat(opts,baseline,frozen_weights(),models['folds'][frame['family']],guarded=method=='oof-guarded')
        if method=='narrow':
            opts=[q for q in opts if min(abs(q['level']-baseline),q['scene_delta'])<=6]
        truth=row['truth']
        votes=lambda q:truth.get(str(q['level']),0)
        if not opts:return None
        best=max(opts,key=votes)
        # Abstain when no input samples belong to a visible face on a proposed plane.
        return best['level'] if votes(best) else None
    return _place(image,version,parts,counts,camera,None,remove_occluded=True,choose_plane=choose)


def narrow(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'narrow')


def wide(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'wide')


def family_oof(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'oof')


def family_oof_guarded(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'oof-guarded')


if __name__=='__main__':index()
