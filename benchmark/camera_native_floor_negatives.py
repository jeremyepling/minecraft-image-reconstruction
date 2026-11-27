"""Source-level negative controls and true source-only positive. No pose is used."""
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark.camera_native_floor_run import REVIEW, COHORTS, SOURCE, frames, save, solve
from engine.camera_atlas_floor import _atlas


def main():
    frame=next(r for r in frames(SOURCE,COHORTS) if r['id']=='train-pillars-02-marble-high-right')
    path=Path(frame['frame'])/'rgb.png'
    source=np.asarray(Image.open(path).convert('RGB'))
    h,w=source.shape[:2]
    noise=np.random.default_rng(473).normal(0,1.2,(h,w,1))
    color=np.array([126,126,117])
    x,y=np.meshgrid(np.arange(w),np.arange(h))
    # A textured, genuinely vertical plane: depth depends on the horizontal ray.
    # The vertical tile axis has a vertical vanishing point under upright projection.
    denom=1+.32*(x/w-.5)
    atlas=_atlas('26.3')
    tile=atlas.tiles[atlas.labels.index('light_gray_concrete')]
    vertical=tile[np.floor(y/denom/1.6).astype(int)%16,np.floor(x/denom/1.6).astype(int)%16]
    split=source.copy()
    split[:,w//2:]=np.roll(source,37,axis=1)[:,w//2:]
    near=source.copy()
    near[int(.88*h):]=np.roll(source,31,axis=1)[int(.88*h):]
    tint=source.astype(int)+[40,-50,-35]
    fixtures=dict(positive=source,constant=np.broadcast_to(color,source.shape),
                  noise=color+noise,one_axis_stripes=color+2*np.sin(x[:,:,None]*.36),
                  vertical_wall=vertical,split_phase=split,near_far_phase=near,wrong_material=tint)
    outputs=[]
    for name,value in fixtures.items():
        im=Image.fromarray(np.uint8(np.clip(value,0,255)))
        result=solve(im,None)
        expected=(name=='positive')
        outputs.append(dict(fixture=name,expected_acceptance=expected,accepted=result['P'] is not None,
                            reason=result['reason'],evidence=result['evidence']))
        print(name,result['P'] is not None,flush=True)
    save(REVIEW/'negative-controls.json',outputs)
    assert all(r['accepted']==r['expected_acceptance'] for r in outputs)


if __name__=='__main__':main()
