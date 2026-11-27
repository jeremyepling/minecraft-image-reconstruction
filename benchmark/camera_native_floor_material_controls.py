"""Held renderer stress test with real atlas sprites projected onto the same RGB-derived floor."""
import numpy as np
from PIL import Image

from benchmark.camera_native_floor import coords, sample_uv, templates
from benchmark.camera_native_floor_run import COHORTS, REVIEW, RUN, SOURCE, frames, read, save
from engine.camera_atlas_floor import _atlas
from engine.camera_native_floor import predict


def main():
    key = 'train-pillars-02-marble-high-right'
    entry = next(e for e in frames(SOURCE, COHORTS) if e['id'] == key)
    from pathlib import Path
    im = np.asarray(Image.open(Path(entry['frame'])/'rgb.png').convert('RGB'))
    h,w=im.shape[:2]
    camera = read(RUN/'boundary/metric'/(key+'.json'))['camera']
    inv = np.linalg.inv(np.asarray(camera['P'])[:,[0,2,3]])
    y,x=np.mgrid[int(.75*h):h,:w]
    screen=np.array([x.ravel()+.5,y.ravel()+.5,np.ones(x.size)])
    atlas=_atlas('26.3')
    cases=[]
    for name in ('light_gray_concrete','stone','light_gray_wool','polished_andesite'):
        tile=templates(atlas,[atlas.labels.index(name)],16)[0,0]
        pattern=sample_uv(tile,coords(inv,screen,16),'linear').reshape(x.shape)
        normalized=(pattern-pattern.mean())/pattern.std()
        # Equal brightness, chromaticity and modulation contrast force verification
        # to distinguish the native pattern. The geometry and sampler are identical.
        pixels=np.clip(np.rint(np.array([127.,128.,121.])+normalized[:,:,None]*.85),0,255).astype('uint8')
        image=im.copy()
        image[int(.75*h):]=pixels
        result=predict(Image.fromarray(image))
        cases.append(dict(sprite=name,accepted=result['P'] is not None,reason=result['reason'],
                          samplesource=key,evidence=result.get('evidence',{})))
    save(REVIEW/'material-controls.json',cases)
    print([(r['sprite'],r['accepted']) for r in cases])
    assert [r['accepted'] for r in cases]==[True,False,False,False]


if __name__=='__main__':main()
