"""Verify a proposed wall-panel exchange against shifted atlas texels at adjacent depths."""
from benchmark.local_panel_depth import choose, materialize, trials
from benchmark.texture_camera_eval import STABLE
from engine.camera_atlas_floor import _atlas
from engine.roof_span_texture import measure
from engine.thin_atlas_faces import StripEvidence


def apply(image,parts,camera,prior,scene,option):
    if option is None:
        return prior,{}
    atlas=_atlas('26.3',STABLE/'runs/26.3/atlas')
    if atlas is None:
        return prior,dict(reason='atlas unavailable')
    src=set(option['before'])
    dst=set(option['after'])
    ctx=StripEvidence(image,parts,camera,prior-src,atlas)
    def witnesses(cells):
        joint=ctx.joint(cells)
        result=[]
        for cell in sorted(cells):
            fits=[]
            for a in range(3):
                if cell[a]<=ctx.eye[a]<=cell[a]+1:
                    continue
                row=measure(ctx,cell,a,joint,palette=())
                if row and row['split']>=.62 and row['phase']>=.075 and row['depth']>=.09 and row['color']<=9:
                    fits.append(a)
            result.append(fits)
        return result
    old=witnesses(src)
    new=witnesses(dst)
    # An exposed source tile with a calibrated depth match vetoes the exchange.
    # Every moved block must independently match at least one destination face.
    accept=not any(old) and all(new)
    return (materialize(image,scene,prior,option) if accept else prior),dict(source_axes=old,destination_axes=new)


def recover(image,parts,counts,camera,prior):
    scene,options=trials(image,parts,camera,prior)
    return apply(image,parts,camera,prior,scene,choose(options))
