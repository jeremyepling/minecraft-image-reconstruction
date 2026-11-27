"""Draw native source images and aligned recorded-camera export errors."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_recorded_replay import COHORTS, OUTPUT, SOURCE, frames, read
from benchmark.expanded_camera_examples import font
from benchmark.expanded_scores import evaluate_pack, load_reference
from training.visible_labels import visible_labels


EXAMPLES = (
    ('test-castle-01-limestone-low-left', 'Castle 1 · test'),
    ('train-pillars-01-marble-low-left', 'Pillars 1 · train'),
    ('train-market-court-1-warm-low-left', 'Market Court 1 · train'),
)
COLORS = {
    'Correct position and ID': (54, 207, 133),
    'Correct position, other ID': (255, 181, 45),
    'Missed position': (65, 153, 255),
    'Projected extra': (247, 77, 88),
}


def example(entry, output, expected=None):
    folder = Path(entry['frame'])
    reference = load_reference(entry)
    build = output / entry['dataset'] / entry['id']
    if expected is None:
        expected = read(build / 'score.json')
    camera = read(build / 'build-manifest.json')['placement']
    actual, predicted = evaluate_pack(entry, build / 'reconstruction.zip', camera, reference=reference)
    if actual['pack_sha256'] != expected['pack_sha256'] or actual['geometry'] != expected['geometry']:
        raise ValueError(f'Overlay evidence differs from scored pack: {entry["id"]}')
    with Image.open(folder / 'rgb.png') as image:
        photo = image.convert('RGB')
    labels = read(folder / 'labels.json')
    with np.load(folder / 'labels.npz', allow_pickle=False) as data:
        owners, depth = data['face_id'], data['depth_z']
    correct_id, other_id, missed, extra = COLORS.values()
    palette = np.zeros((len(labels['faces']) + 1, 3), np.uint8)
    for face in labels['faces']:
        point = tuple(face['xyz_world'])
        palette[face['id']] = (missed if point not in predicted else
                               correct_id if predicted[point] == reference.truth[point] else other_id)
    pixels = palette[owners]

    # The decoded export contains whole blocks. Draw extra cells only where
    # their front faces would appear from the source view after score alignment.
    cube = dict(block_id='minecraft:smooth_stone', state='Block{minecraft:smooth_stone}')
    reconstructed, reconstructed_ids, reconstructed_depth = visible_labels(
        reference.pose, dict.fromkeys(predicted, cube), (0, 0, 0))
    extra_palette = np.zeros(len(reconstructed['faces']) + 1, bool)
    for face in reconstructed['faces']:
        extra_palette[face['id']] = tuple(face['xyz_world']) not in reference.truth
    foreground_extra = extra_palette[reconstructed_ids] & ((owners == 0) | (reconstructed_depth < depth - .001))
    pixels[foreground_extra] = extra
    painted = (owners > 0) | foreground_extra
    rgb = np.array(photo)
    rgb[painted] = .36 * rgb[painted] + .64 * pixels[painted]
    return photo, Image.fromarray(rgb), expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=OUTPUT)
    parser.add_argument('--out', type=Path, default=OUTPUT / 'examples.jpg')
    args = parser.parse_args()
    entries = {r['id']: r for r in frames(SOURCE, COHORTS)}
    width, height, margin, row = 560, 350, 18, 416
    canvas = Image.new('RGB', (width * 2 + margin * 3, 112 + row * len(EXAMPLES)), '#182228')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), 'Placement with the recorded camera', fill='#f0f5f3', font=font(26))
    draw.text((margin, 51), 'Source screenshot', fill='#f0f5f3', font=font(16))
    draw.text((2 * margin + width, 51), 'Aligned export errors', fill='#f0f5f3', font=font(16))
    for index, (label, color) in enumerate(COLORS.items()):
        x = margin + index * 255
        draw.rounded_rectangle((x, 77, x + 13, 90), 2, fill=color)
        draw.text((x + 20, 75), label, fill='#e5ece8', font=font(15))
    for index, (key, title) in enumerate(EXAMPLES):
        photo, error, score = example(entries[key], args.run)
        y = 105 + row * index
        for column, panel in enumerate((photo, error)):
            x = margin + column * (width + margin)
            canvas.paste(panel.resize((width, height), Image.Resampling.LANCZOS), (x, y))
        strict = score['geometry']['strict']
        typed = score['scores']['strict']
        draw.text((margin, y + height + 3), title, fill='#f0f5f3', font=font(18))
        draw.text((margin, y + height + 27), f'Ordinary export  0 blocks   ·   {key}', fill='#b9c7c4', font=font(15))
        draw.text((2 * margin + width, y + height + 3),
                  f'{strict["correct"]:,} correct  ·  {strict["missed"]:,} missed  ·  {strict["extras"]:,} extra',
                  fill='#f0f5f3', font=font(18))
        draw.text((2 * margin + width, y + height + 27),
                  f'{strict["f1"]:.2%} strict geometry F1  ·  {typed["f1"]:.2%} strict position + ID F1',
                  fill='#b9c7c4', font=font(15))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, quality=90, optimize=True)


if __name__ == '__main__':
    main()
