"""Draw the two Courtyard plank faces and the baseline depth error from the frozen exports."""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmark import current_placement_viewer as viewer
from benchmark.camera_sheet import label
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.thin_face_placement_run import ref_inputs
from engine.pillar_seams import native_mask, orientation
from engine.visible_placement import _sample_face
from engine.visible_raster import SAMPLE_STRIDE

KEY = 'legacy-light_gray_concrete-holdout-courtyard-plaster-low-left'
FACES = [(6, 'mask-2', 0, -1), (7, 'mask-0-junction-0-single-457-2', 2, 1)]
OUT = viewer.ROOT / 'courtyard-corner-01' / 'plank-corner.png'


def outline(image, mask, color):
    boundary = mask & ~np.asarray(Image.fromarray(mask).filter(ImageFilter.MinFilter(3)))
    image[boundary] = color


def render(pack):
    job = next(j for j in viewer.suite.jobs(viewer.suite.framed.EVALUATION) if j[1]['id'] == KEY)
    stages = next(r for r in viewer.read(viewer.ROOT / 'facade-inset-round-01/combined.json')['rows'] if r['id'] == KEY)
    image, parts, _, camera = viewer.latest_stages(job, stages)
    _, _, _, _, _, labels, _, _, _, _, _ = viewer.sheet.stages.read_view(viewer.suite.CONTROL, *job)
    _, _, transform, _ = ref_inputs(job)
    _, history, archives = viewer.read_releases([viewer.RELEASE, viewer.ROOT / 'smooth-local-placement-01',
                                                viewer.ROOT / 'smooth-joint-placement-01', viewer.ROOT / 'window-wall-01'])
    cells = decoded_pack(BytesIO(archives[KEY])) if KEY in history else viewer.current_pack(job)[0]
    world = registered(cells, transform)
    masks = {p['id']: native_mask(p, image.size) for p in parts}
    projection, _ = orientation(camera, image.size)
    projection[:2] *= SAMPLE_STRIDE
    inverse = np.linalg.inv(transform)
    wood = {tuple(b['xyz_world']) for b in labels['blocks'] if b['block_id'] == 'minecraft:spruce_planks'}
    assert len(wood) == 35 and len(wood & world) == 19
    if pack is None:
        _, _, latest = viewer.read_releases([viewer.ROOT / 'counted-ground-corner-01'])
        pack = BytesIO(latest[KEY])
    revised = registered(set(decoded_pack(pack)), transform)
    assert wood <= revised and len((world - revised)) == 17 and len(revised - world) == 16
    canvas = Image.new('RGB', (1210, 853), '#f7f6f0')
    pen = ImageDraw.Draw(canvas)
    label(pen, (24, 18), 'Courtyard  /  Why the middle plank rows sat too far forward', 30, bold=True)
    label(pen, (24, 66), 'Both visible walls measure four blocks across and five high. Each orange outline follows a detected face.', 17)
    titles = ['Source image', 'Corner one block forward', 'Corner at the measured depth']
    crop = (772, 390, 1010, 618)
    for panel, heading in enumerate(titles):
        rgb = np.array(image)
        overlaps = []
        for surface_id, name, axis, sign in FACES:
            mask = masks[name]
            region = np.zeros_like(mask)
            if panel:
                # Reference face IDs identify the two counted rectangles for this diagnostic.
                # The grid and camera used to project them come from the ordinary image pipeline.
                for face in labels['faces']:
                    if face['surface_id'] != surface_id:
                        continue
                    pos = np.array(face['xyz_world']) + .5
                    cell = np.floor(inverse[:3, :3] @ pos + inverse[:3, 3]).astype(int)
                    if panel == 1:
                        cell += [-1, 0, 1]
                    box, pixels, *_ = _sample_face(projection, tuple(cell), axis, sign, mask.shape)
                    region[box] |= pixels
                overlaps.append((mask & region).sum() / (mask | region).sum())
                outline(rgb, region, (38, 250, 221))
            outline(rgb, mask, (255, 169, 42))
        left = 24 + 395 * panel
        label(pen, (left, 106), heading, 21, bold=True)
        view = Image.fromarray(rgb).crop(crop).resize((367, 352), Image.Resampling.NEAREST)
        canvas.paste(view, (left, 143))
        note = ('Two detected plank faces' if not panel else
                'Face overlap   ' + '  /  '.join(f'{value:.1%}' for value in overlaps))
        label(pen, (left, 509), note, 18)
    label(pen, (24, 566), 'Orange = detected edge     Teal = projected edge of a complete 4 × 5 face', 18, bold=True)
    label(pen, (24, 604), 'The forward corner can place individual blocks over real plank pixels while missing the shared edge and the outside edges.', 17)
    label(pen, (24, 636), 'The comparison uses the source camera estimate. Reference blocks identify the two rectangles for this diagnostic.', 15)
    for left, title, placed in [(24, 'Previous export', world), (419, 'Revised export', revised)]:
        label(pen, (left, 681), title, 18, bold=True)
        for row, z in enumerate(range(0, -5, -1)):
            for column, x in enumerate(range(1291, 1296)):
                cell = x, 67, z
                color = ('#389579' if cell in wood and cell in placed else '#4e98d5' if cell in wood
                         else '#d26648' if cell in placed else '#e8e7e2')
                a, b = left + 26 * column, 711 + 26 * row
                pen.rectangle((a, b, a+23, b+23), fill=color)
    label(pen, (624, 682), 'A middle row viewed from above', 21, bold=True)
    label(pen, (624, 725), 'Green = correct     Blue = missing     Red = extra', 17)
    label(pen, (624, 760), 'Both grids come from decoded exports.', 17)
    label(pen, (624, 796), 'The two visible faces share the corner block.', 17)
    OUT.parent.mkdir(exist_ok=True, parents=True)
    canvas.save(OUT, optimize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, help='Verified revised Courtyard export')
    render(parser.parse_args().pack)
