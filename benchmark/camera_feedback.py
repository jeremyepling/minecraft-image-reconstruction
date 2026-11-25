"""Attach a visual preference to a run and a source-image rectangle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmark.camera_round import read


def record(run, image_id, box, preferred, note, floor=None):
    manifest = read(run / 'manifest.json')
    fixture = manifest['fixture']
    if 'indexes' in fixture:
        if floor not in fixture['indexes']:
            raise ValueError('Specify the floor condition for this paired-suite review.')
        index_path = run / fixture['indexes'][floor]
    else:
        if floor is not None:
            raise ValueError('This historical run has a single fixture index.')
        index_path = run / fixture['index']
    index = read(index_path)
    frame = next((f for f in index['frames'] if f['id'] == image_id), None)
    if not frame or preferred not in [*manifest['methods'], 'no-preference']:
        raise ValueError('Choose an image and method from this run.')
    pose = read(index_path.parent / frame['pose'])
    w, h = pose['image_size']
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and note.strip()):
        raise ValueError('Specify an in-frame rectangle and a review note.')
    item = dict(run_id=manifest['run_id'], stage=manifest['stage'], image=image_id, region_xyxy=list(box),
                preferred=preferred, note=note, recorded_at=datetime.now(timezone.utc).isoformat())
    if floor is not None:
        item.update(floor=floor, suite=manifest['suite'], image_sha256=frame['image_sha256'])
    with (run / manifest['artifacts']['visual_feedback']).open('a', encoding='utf8') as out:
        out.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--image', required=True)
    parser.add_argument('--floor', help='Floor condition of the reviewed image in a paired run')
    parser.add_argument('--region', nargs=4, required=True, type=int, metavar=('X0','Y0','X1','Y1'))
    parser.add_argument('--preferred', required=True)
    parser.add_argument('--note', required=True)
    args = parser.parse_args()
    record(args.run, args.image, args.region, args.preferred, args.note, args.floor)
