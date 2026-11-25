"""Attach a visual preference to a run and a source-image rectangle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmark.camera_round import read


def record(run, image_id, box, preferred, note):
    manifest = read(run / 'manifest.json')
    index = read(run / manifest['fixture']['index'])
    frame = next((f for f in index['frames'] if f['id'] == image_id), None)
    if not frame or preferred not in [*manifest['methods'], 'no-preference']:
        raise ValueError('Choose an image and method from this run.')
    pose = read((run / manifest['fixture']['index']).parent / frame['pose'])
    w, h = pose['image_size']
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and note.strip()):
        raise ValueError('Specify an in-frame rectangle and a review note.')
    item = dict(run_id=manifest['run_id'], stage=manifest['stage'], image=image_id, region_xyxy=list(box),
                preferred=preferred, note=note, recorded_at=datetime.now(timezone.utc).isoformat())
    with (run / manifest['artifacts']['visual_feedback']).open('a', encoding='utf8') as out:
        out.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--image', required=True)
    parser.add_argument('--region', nargs=4, required=True, type=int, metavar=('X0','Y0','X1','Y1'))
    parser.add_argument('--preferred', required=True)
    parser.add_argument('--note', required=True)
    args = parser.parse_args()
    record(args.run, args.image, args.region, args.preferred, args.note)
