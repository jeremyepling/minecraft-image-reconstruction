"""Render the six row-transfer decisions alongside their paired exported geometry."""
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read
from benchmark.expanded_scores import load_reference
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_round import SHARED
from benchmark.surface_sample_review import main as render_samples


PRIOR = WORKTREES / 'relative-face-assembly/runs/relative-face-stage-review-01'
CANDIDATE = ROOT / 'runs/shared-corner-rows-01/sky'
OUTPUT = ROOT / 'runs/shared-corner-viewer-01'


def metrics(*variants):
    result = []
    for label, group, priority in [
        ('Strict placement F1', 'geometry', 'strict'), ('Facade placement F1', 'geometry', 'facade'),
        ('Position + material F1', 'scores', 'strict')]:
        result.append([label, *[f'{100 * s[group][priority]["f1"]:.2f}%' for s in variants]])
    for key, label in [('correct', 'Correct blocks'), ('missed', 'Missed blocks'), ('extras', 'Extra blocks')]:
        result.append([label, *[s['geometry']['strict'][key] for s in variants]])
    return result


def format_seam(seam):
    decision = seam['decision']
    accepted = decision in ('accepted', 'accept', 'transferred')
    desc = f"{seam['donor']} → {seam['receiver']}. {seam.get('reason', '')}"
    if seam.get('donor_rows') is not None:
        desc += f" Donor height estimate: {seam['donor_rows']} rows."
    if seam.get('receiver_rows') is not None:
        desc += f" Receiver estimate: {seam['receiver_rows']} rows."
    return {**seam, 'description': desc, 'accepted': accepted}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = runpy.run_path(str(PRIOR / 'render_review.py'))
    stages = json.loads((PRIOR / 'data.js').read_text(encoding='utf-8').split('=', 1)[1].strip().rstrip(';'))
    fixtures = {c['id']: c for c in read(SHARED / 'fixtures.json')['cases']}
    audits = {c['id']: c for c in read(ROOT / 'runs/shared-corner-audit-01/visible-edges.json')}
    cases = []
    for stage in stages:
        ident = stage['id']
        folder = OUTPUT / ident
        folder.mkdir(exist_ok=True)
        run = CANDIDATE / ident
        case = fixtures[ident]
        ref = load_reference(case)
        receipt = read(run / 'score.json')
        build = read(run / 'placements.json')
        assert receipt['source_sha256'] == digest(PRIOR / stage['source'])
        assert receipt['pack_sha256'] == digest(run / 'reconstruction.zip')
        assert decoded_pack(run / 'reconstruction.zip') == {tuple(p): m for p, m in build['local']}
        aligned = {tuple(p): m for p, m in build['aligned']}
        after = receipt['score']['fixed']
        assert len(aligned.keys() & ref.truth.keys()) == after['geometry']['strict']['correct']
        base = WORKTREES / 'relative-face-assembly/runs/relative-face-assembly-01/atlas-dense/sky' / ident
        before = read(base / 'score.json')['score']['fixed']
        donor = read(CANDIDATE.parent / 'donor-only' / 'sky' / ident / 'score.json')['score']['fixed']
        sparse_run = ROOT / 'runs/shared-corner-audit-01/receiver-only/sky' / ident
        sparse = read(sparse_run / 'score.json')['score']['fixed'] if sparse_run.exists() else before
        images = {}
        for key, src in [('seams', stage['source']), ('masks', stage['assets']['masks']),
                         ('before', stage['methods']['image']['assets']['placed'])]:
            target = folder / Path(src).name
            shutil.copyfile(PRIOR / src, target)
            images[key] = f'{ident}/{target.name}'
        frame = Path(case['frame'])
        shutil.copyfile(run / 'source-overlay.jpg', folder / 'grids.jpg')
        images['grids'] = f'{ident}/grids.jpg'
        with np.load(frame / 'labels.npz', allow_pickle=False) as native:
            images['after'] = previous['placements'](
                Image.open(PRIOR / stage['source']).convert('RGB'), native['face_id'], native['depth_z'],
                read(frame / 'labels.json'), ref, aligned, folder, 'row-transfer')
            images['sparse'] = images['before']
            if sparse_run.exists():
                sparse_build = read(sparse_run / 'placements.json')
                assert decoded_pack(sparse_run / 'reconstruction.zip') == {tuple(p): m for p, m in sparse_build['local']}
                images['sparse'] = previous['placements'](Image.open(PRIOR / stage['source']).convert('RGB'),
                    native['face_id'], native['depth_z'], read(frame / 'labels.json'), ref,
                    {tuple(p): m for p, m in sparse_build['aligned']}, folder, 'original-donor')
        seams = [format_seam(s) for s in read(run / 'seams.json')['seams']]
        chains = audits[ident]['chains']
        for seam in seams:
            matched = [a for a in chains if set(a['pair']) == {seam['donor'], seam['receiver']}]
            if matched:
                longest = max(a['count'] for a in matched)
                seam['truth'] = f'Recorded geometry: the longest verified edge at this mask pair has {longest} visible unit segments.'
        segments = []
        for chain in chains:
            for j, segment in enumerate(chain['boundaries']):
                segments.append(dict(endpoints=[segment['low'], segment['high']],
                    label=' / '.join(chain['pair']) + f" · {chain['count']}" if j == 0 else ''))
        images['reference'] = images['masks']
        accepted = sum(s['accepted'] for s in seams)
        cases.append(dict(title=stage['title'], size=stage['size'], images=images, seams=seams,
                          summary=f'{accepted} accepted transfers from {len(seams)} reported candidates.',
                          placementLegend='Green = correct position and material. Amber = correct position, different material. Blue = missed block. Pink = visible extra.',
                          metrics=metrics(before, sparse, donor, after), references=segments))
    (OUTPUT / 'corner-data.js').write_text('const cornerCases=' + json.dumps(cases) + ';', encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('shared_rows_viewer.html'), OUTPUT / 'index.html')
    render_samples()


if __name__ == '__main__':
    main()
