"""Reserve the remaining supported shapes and run the ordinary upstream build once."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from benchmark import visible_depth_transfer as transfer
from benchmark.texture_camera_eval import STABLE, checked, read, save


OUT = STABLE / 'runs/canopy-placement/confirmation'
SCREEN_SHA = '691708e501e0647365f15793d78f5001b59a9b35815b0052ad6586d0eba5a866'


def reserve(destination):
    screen = read(checked(transfer.OUT/'screen.json', SCREEN_SHA))
    used = {r['shape'] for r in screen['selected']}
    unique = {}
    for row in sorted(screen['screened'], key=lambda r: r['files']['rgb.png']):
        if row['camera']['P'] is not None and row['shape'] not in used:
            unique.setdefault(row['shape'], row)
    assert len(unique) == 8
    destination.mkdir(parents=True, exist_ok=True)
    save(destination/'screen.json', dict(source_sha256=SCREEN_SHA, selected=list(unique.values())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('reserve','build'))
    p.add_argument('--inputs', type=Path, default=OUT)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    if a.stage == 'reserve':
        reserve(a.inputs)
        return
    selected = read(a.inputs/'screen.json')['selected']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(transfer.build, [(r, a.inputs) for r in selected]))


if __name__=='__main__':
    main()
