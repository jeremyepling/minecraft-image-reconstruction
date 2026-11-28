"""Materialize a source challenger in the portable Floating replay layout."""
from argparse import ArgumentParser
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, save
from benchmark.floating_expanded_run import score_export, summarize
from benchmark.floating_selective_replay import assemble_arm
from benchmark.floating_stage_report import read
from benchmark.floating_wall_depth_partition import parts, source_parts
from benchmark.mask_blocks import decoded_pack
from benchmark.mobile_surface_evidence import build_expanded


def materialize(base, outputs, fields, mode):
    cv2.setNumThreads(1)
    root = outputs / ('mobile-' + mode)
    root.mkdir(exist_ok=True)
    case_list = base / 'fixtures.json'
    shutil.copyfile(case_list, root / 'fixtures.json')
    records = []
    for entry in read(case_list)['cases']:
        name = entry['id']
        old = base / 'original' / name
        folder = root / 'original' / name
        comparison = read(outputs / name / (mode + '.json'))
        measured = comparison['arms'][mode]
        assert comparison['rgb_sha256'] == entry['rgb_sha256']
        if not folder.exists():
            handoff = 'source-handoff'
            if measured['unchanged']:
                # Only the self-contained native replay receipts are needed by the review.
                folder.mkdir(parents=True)
                for p in old.iterdir():
                    if p.is_file():
                        shutil.copyfile(p, folder / p.name)
            else:
                packet = read(old / 'charts.json')
                source = Path(entry['variants']['original']['image'])
                assert digest(source) == comparison['rgb_sha256']
                rgb = np.asarray(Image.open(source).convert('RGB'))
                planes = read(old / handoff / 'surfaces.json')['planes']
                original = read(old / 'case.json')
                target = next(c for c in packet['charts'] if c['id'] == comparison['target'])
                ids, _ = source_parts(target, planes, rgb, fields / name / 'fields', original, mode)
                charts, siblings, counts = parts(packet['charts'], target, ids)
                assert {str(k):v for k,v in counts.items()} == comparison['partition_cells']
                folder.parent.mkdir(parents=True, exist_ok=True)
                assembled = assemble_arm(dict(packet, charts=charts), planes, rgb, siblings, 2.5,
                                         folder, save_trace=True)
                score, aligned = score_export(entry, folder / 'reconstruction.zip', assembled)
                assert score['geometry'] == measured['score']['geometry'] and score['scores'] == measured['score']['scores']
                assert decoded_pack(folder / 'reconstruction.zip') == decoded_pack(outputs/name/mode/'reconstruction.zip')
                save(folder / 'charts.json', dict(packet, charts=charts))
                save(folder / 'aligned-placements.json.gz', [[*key, value] for key, value in sorted(aligned.items())])
                stages = dict(original['stages'], split_border_charts=len(charts),
                              oriented_charts=len(read(folder / 'oriented-charts.json')['charts']),
                              selected_charts=len(assembled['selected']), recorded_joins=len(assembled['joins']))
                record = dict(original, charts_sha256=digest(folder / 'charts.json'),
                              oriented_charts_sha256=digest(folder / 'oriented-charts.json'),
                              assembly_sha256=digest(folder / 'assembly.json'), stages=stages, score=score)
                save(folder / 'case.json', record)
            (folder / handoff).mkdir()
            for filename in ('surfaces.json', 'counts.json', 'build-manifest.json'):
                shutil.copyfile(old / handoff / filename, folder / handoff / filename)
            if not measured['unchanged']:
                build_expanded(folder, source, record, assembled)
        record = read(folder / 'case.json')
        evidence = read(folder / 'surfaces.json')
        assert digest(folder / 'reconstruction.zip') == record['score']['pack_sha256'] == evidence['provenance']['arms']['Candidate']['packSha256']
        assert digest(folder / 'charts.json') == record['charts_sha256'] == evidence['provenance']['arms']['Candidate']['chartSha256']
        assert record['score']['scores'] == measured['score']['scores'] and record['score']['geometry'] == measured['score']['geometry']
        records.append(record)
    summary = read(base / 'summary.json')
    summary['baseline_revision'] = summary['revision']
    revision = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    summary.update(method=mode, revision=revision, all_arms=summarize(records),
                   by_split={k:summarize([r for r in records if r['split'] == k]) for k in ('train','validation','test')},
                   by_lighting={'noon':summarize(records)})
    summary.pop('predictor_sha256', None)
    save(root / 'summary.json', summary)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('base', 'output', 'fields'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=['source', 'source-appearance'], required=True)
    args = parser.parse_args()
    materialize(args.base, args.output, args.fields, args.mode)
