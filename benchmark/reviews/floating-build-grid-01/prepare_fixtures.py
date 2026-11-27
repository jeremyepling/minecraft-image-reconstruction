
"""Freeze background-swapped, visible-only Java 26.3 test inputs."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import COHORTS, SOURCE, digest, frames, read, save

OUTPUT = SOURCE / 'floating-build-grid-01'
REVISION = 'e3bd220188f224a0a6cf2841c8f495fd0ac9d090'
FRAME_IDS = (
    'test-castle-01-ashlar-high-right',
    'validation-harbor-01-seafoam-low-left',
    'train-viaduct-02-riverstone-high-right',
    'train-pillars-01-marble-low-left',
    'validation-hillside-chapel-1-warm-low-left',
    'test-walled-garden-1-warm-high-right',
)
BACKGROUNDS = ('pavement', 'plain', 'sky')


def cloud_backdrop(size):
    width, height = size
    top = np.array((107, 157, 252), dtype=float)
    bottom = np.array((183, 222, 255), dtype=float)
    weight = np.linspace(0, 1, height)[:, None, None] ** 1.15
    gradient = np.uint8(np.round(top + (bottom - top) * weight))
    image = Image.fromarray(np.repeat(gradient, width, axis=1), 'RGB')
    cloud = Image.new('RGBA', image.size)
    layer = ImageDraw.Draw(cloud)
    for x, y, span in ((.016,.20,.15), (.68,.125,.19), (.81,.39,.17), (.069,.73,.13), (.71,.83,.21)):
        def polygon(points, paint):
            layer.polygon([(round(width*(x + span*u)), round(height*(y + .055*v)))
                            for u,v in points], fill=paint)
        polygon([(0,.32),(.21,.32),(.21,0),(.63,0),(.63,.2),(.81,.2),(.81,.43),(1,.43),
                 (1,.77),(.73,.77),(.73,1),(.26,1),(.26,.76),(0,.76)], (253,254,255,205))
        polygon([(0,.67),(.26,.67),(.26,.85),(.73,.85),(.73,.69),(1,.69),(1,.84),
                 (.73,.84),(.73,1),(.26,1),(.26,.82),(0,.82)], (205,225,255,180))
    return np.array(Image.alpha_composite(image.convert('RGBA'), cloud).convert('RGB'))


def assemble(entry):
    frame = Path(entry['frame'])
    provenance = read(frame / 'frame.json')
    for name in ('rgb.png', 'camera.json', 'labels.json', 'labels.npz'):
        if digest(frame/name) != provenance['files'][name]:
            raise ValueError(f'Invalid frozen input {frame/name}')
    source = frame/'rgb.png'
    photo = np.array(Image.open(source).convert('RGB'))
    with np.load(frame/'labels.npz', allow_pickle=False) as labels:
        visible = labels['face_id'] > 0
    y, x = np.nonzero(visible)
    width, height = photo.shape[1], photo.shape[0]
    clearance = [int(x.min()), int(y.min()), int(width-1-x.max()), int(height-1-y.max())]
    if min(np.asarray(clearance)/[width,height,width,height]) < .02:
        raise ValueError(f'Insufficient framing: {entry["id"]}')
    dest = OUTPUT/'images'/entry['id']
    dest.mkdir(parents=True, exist_ok=True)
    inputs = {'pavement':source}
    for kind, backdrop in (
        ('plain',(97,92,106)),
        ('sky',cloud_backdrop((width,height))),
    ):
        output = np.empty_like(photo)
        output[:] = backdrop
        output[visible] = photo[visible]
        path = dest / f'{kind}.png'
        pending = path.with_suffix('.png.part')
        Image.fromarray(output).save(pending, format='PNG')
        pending.replace(path)
        restored = np.asarray(Image.open(path).convert('RGB'))
        if not np.array_equal(restored[visible],photo[visible]):
            raise ValueError(f'Changed foreground pixels: {path}')
        inputs[kind] = path
    return dict(entry, visible_pixels=int(visible.sum()), clearance_ltrb_px=clearance,
                variants={kind:dict(image=str(inputs[kind]), source_sha256=digest(inputs[kind]))
                          for kind in BACKGROUNDS}, reference_sha256={
                              name:provenance['files'][name] for name in ('camera.json','labels.json','labels.npz')})


def contact_sheet(cases):
    font_dir=Path('C:/Windows/Fonts')
    heading=ImageFont.truetype(str(font_dir/'segoeuib.ttf'),24)
    regular=ImageFont.truetype(str(font_dir/'segoeui.ttf'),17)
    width, height, margin, row = 376, 235, 16, 270
    image=Image.new('RGB',(3*width+4*margin,86+row*len(cases)),'#161e24')
    draw=ImageDraw.Draw(image)
    draw.text((margin,12),'The same structure pixels on three backgrounds',font=heading,fill='#eff3f6')
    for i,label in enumerate(('Original pavement','Plain backdrop','Open sky')):
        draw.text((margin+i*(width+margin),53),label,font=regular,fill='#bed8e7')
    for j,case in enumerate(cases):
        baseline=81+j*row
        draw.text((margin,baseline),case['id'],font=regular,fill='#eff3f6')
        for i, kind in enumerate(BACKGROUNDS):
            x=margin+i*(width+margin)
            preview=Image.open(case['variants'][kind]['image']).convert('RGB')
            preview.thumbnail((width,height),Image.Resampling.LANCZOS)
            image.paste(preview,(x,baseline+23))
    image.save(OUTPUT/'fixture-preview.jpg',quality=90,optimize=True)


def main():
    available={entry['id']:entry for entry in frames(SOURCE,COHORTS)}
    cases=[assemble(available[frame_id]) for frame_id in FRAME_IDS]
    contact_sheet(cases)
    save(OUTPUT/'fixtures.json',dict(revision=REVISION, backgrounds=BACKGROUNDS,
         sky_is_composite=True, prior_visibility_preserved=True,
         description='Original recorded pavement and two native-pixel composites preserving every labeled visible structure pixel. The horizonless open-sky canvas is synthetic.',
         cases=cases))
    print(OUTPUT/'fixtures.json')
    for case in cases:
        print(case['id'],case['visible_pixels'],case['clearance_ltrb_px'])


if __name__ == '__main__':
    main()
