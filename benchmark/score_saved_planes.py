"""Compare surface outlines with the recorded Minecraft geometry without launching the game."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from time import perf_counter

from benchmark.camera_round import ROOT, digest, read
from benchmark.surface_metrics import score_planes
from runtime.mode import image_only


ROUND = ROOT / 'benchmark/reviews/normals-01'
METHOD = 'normal-regions'


def select_case(run, cases, image, build_manifest):
    if image is not None:
        fingerprint = digest(image)
        if build_manifest is not None and fingerprint != build_manifest['image_sha256']:
            raise ValueError('The supplied screenshot differs from the screenshot in the build manifest.')
    elif build_manifest is not None:
        fingerprint = build_manifest['image_sha256']
    else:
        raise ValueError('Supply --image FROZEN.png to identify a prediction without a build manifest.')

    candidates = [row for row in cases if row['sha256'] == fingerprint]
    if not candidates:
        raise ValueError('This screenshot is outside the frozen round.')
    if len(candidates) == 1:
        return candidates[0]
    if image is not None:
        for row in candidates:
            if (run / row['source']).resolve() == image.resolve():
                return row
    paths = '\n'.join(str((run / row['source']).resolve()) for row in candidates)
    raise ValueError(f'These fixture screenshots have identical pixels. Select one with --image:\n{paths}')


def candidate_manifest(surfaces):
    file = surfaces.with_name('build-manifest.json')
    if not file.is_file():
        return None
    manifest = read(file)
    named = manifest.get('artifacts', {}).get('surfaces', manifest.get('artifact'))
    if named != surfaces.name or manifest.get('through') not in ('surfaces', 'counts'):
        raise ValueError('The adjacent build manifest does not identify this surface prediction.')
    if manifest.get('version') != '26.3' or not manifest.get('image_sha256'):
        raise ValueError('The build manifest must identify the source screenshot and Minecraft Java 26.3.')
    return manifest


def score_case(run, row, planes=None, *, use_masks=False):
    from benchmark.surface_reference import recorded, review_case

    archived = planes is None
    if archived:
        with gzip.open(run / row['prediction'], 'rt', encoding='utf-8') as source:
            planes = json.load(source)[METHOD]['model']['planes']
    field = 'support' if use_masks else 'quad'
    if not isinstance(planes, list) or any(not isinstance(p, dict) or field not in p for p in planes):
        raise ValueError(f'Expected a list of planes, each with {field}.')
    _, reference = recorded(review_case(run, row))
    if use_masks:
        from benchmark.mask_metrics import measure
        return measure(planes, reference, use_mask=True)[0]
    score = score_planes(reference, planes)
    if archived:
        saved = row['methods'][METHOD]['outlines']
        if any(score[k] != saved[k] for k in ('matched', 'eligible', 'proposed')):
            raise ValueError(f"Saved counts differ from the replay on {row['key']}.")
    return score


def print_results(rows):
    groups = defaultdict(list)
    for row, score in rows:
        groups[row['layout']].append(score)
    print('Building    Images   Surfaces per image   Found / counted across images   Extra outlines')
    for layout, group in sorted(groups.items()):
        counts = [score['eligible'] for score in group]
        low, high = min(counts), max(counts)
        per_image = str(low) if low == high else f'{low}-{high}'
        matched = sum(score['matched'] for score in group)
        proposed = sum(score['proposed'] for score in group)
        found = f'{matched} / {sum(counts)}'
        print(f'{layout.title():<12}{len(group):>5}{per_image:>21}{found:>32}{proposed - matched:>17}')


@image_only()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Score a single frozen screenshot')
    parser.add_argument('--surfaces', type=Path, help='Score your own surfaces.json for a single screenshot')
    args = parser.parse_args()
    started = perf_counter()
    cases = read(ROUND / 'cases.json')
    try:
        if args.image is not None or args.surfaces is not None:
            manifest = candidate_manifest(args.surfaces) if args.surfaces is not None else None
            row = select_case(ROUND, cases, args.image, manifest)
            planes = None
            use_masks = False
            if args.surfaces is not None:
                model = read(args.surfaces)
                if not isinstance(model, dict) or not isinstance(model.get('planes'), list):
                    raise ValueError('The candidate JSON must contain a planes list.')
                planes = model['planes']
                use_masks = model.get('footprint') == 'support'
            result = score_case(ROUND, row, planes, use_masks=use_masks)
            print(f"Screenshot  {row['key']}")
            print(f"Found {result['matched']} of {result['eligible']} counted surfaces. "
                  f"Missed {result['eligible'] - result['matched']}. "
                  f"Extra outlines {result['proposed'] - result['matched']}.")
        else:
            print_results([(row, score_case(ROUND, row)) for row in cases])
            print(f"All {len(cases)} images reproduced their saved B counts.")
        print(f'Offline comparison took {perf_counter() - started:.2f} seconds.')
    except (ValueError, OSError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
