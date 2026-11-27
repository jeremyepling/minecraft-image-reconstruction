"""Compare the two decoded Colonnade window-wall exports from the recorded camera."""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

from benchmark import facade_inset_round as suite
from benchmark.camera_sheet import label
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.roof_span_worst_sheet import face_map
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, read
from benchmark.thin_face_placement_run import ref_inputs


KEY = 'inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right'
TARGETS = {(105, 66, -6), (106, 66, -6)}
REVIEWS = suite.ROOT / 'benchmark/reviews'
OUT = REVIEWS / 'window-wall-01/colonnade.png'


def released_pack(folder):
    summary = read(folder / 'summary.json')
    row = next(r for r in summary['changed'] if r['id'] == KEY)
    with ZipFile(checked(folder / 'default-exports.zip', summary['archive_sha256'])) as archive:
        data = archive.read(KEY + '/reconstruction.zip')
    if sha256(data).hexdigest() != row['pack_sha256']:
        raise ValueError('Colonnade export hash differs from the release')
    return decoded_pack(BytesIO(data))


def render(old_pack, new_pack, output):
    job = next(job for job in suite.jobs(suite.framed.EVALUATION) if job[1]['id'] == KEY)
    image, _, _, _, pose, _, _, truth, _, _, _ = read_view(suite.CONTROL, *job)
    _, _, transform, _ = ref_inputs(job)
    before = registered(old_pack, transform)
    after = registered(new_pack, transform)
    if not TARGETS <= truth or TARGETS & before or not TARGETS <= after:
        raise ValueError('Comparison requires the baseline gap and both recovered blocks in the new export')

    crop = (703, 443, 893, 570)
    backdrop = Image.new('RGB', image.size, '#28343a')
    current, _ = face_map(backdrop, pose, before, lambda _: (210, 217, 217))
    revised, _ = face_map(backdrop, pose, after,
                          lambda p: (65, 247, 190) if p in TARGETS else (210, 217, 217))
    board = Image.new('RGB', (1224, 461), '#f6f5f1')
    pen = ImageDraw.Draw(board)
    label(pen, (24, 18), 'Colonnade 01  /  Two blocks below the window', 30, bold=True)
    label(pen, (24, 60), 'The same area, viewed from the recorded camera in all three panels.', 17)
    panels = [
        ('Source image', image, 'The low wall spans the opening.'),
        ('Previous export', current, 'The top row has a two-block gap.'),
        ('New export', revised, 'Green marks the two recovered blocks.'),
    ]
    for i, (title, picture, caption) in enumerate(panels):
        x = 24 + 398 * i
        label(pen, (x, 96), title, 21, bold=True)
        board.paste(picture.crop(crop).resize((380, 254), Image.Resampling.NEAREST), (x, 129))
        label(pen, (x, 395), caption, 17)
    label(pen, (24, 435), 'The center and right panels show reprojected exported blocks. Dark background marks rays that hit no exported geometry.', 16)
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, help='New single-view reconstruction.zip from the ordinary build')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.pack:
        manifest = args.pack.with_name('build-manifest.json')
        if manifest.exists():
            job = next(j for j in suite.jobs(suite.framed.EVALUATION) if j[1]['id'] == KEY)
            if read(manifest)['image_sha256'] != job[1]['files']['rgb.png']:
                parser.error('The supplied build manifest belongs to a different source image')
        new = decoded_pack(args.pack)
    else:
        new = released_pack(OUT.parent)
    render(released_pack(REVIEWS / 'pavement-local-01'), new, args.out)


if __name__ == '__main__':
    main()
