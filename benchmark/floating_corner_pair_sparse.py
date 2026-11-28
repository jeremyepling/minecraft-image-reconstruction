"""Probe the pair rule on sparse connected-atlas inputs through its ordinary wall-only assembler."""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import expanded_scores, floating_cloud_replay, relative_face_round
from benchmark.expanded_baseline import digest, read, save, ROOT
from benchmark.floating_corner_overwrite import linked, sources
from benchmark.floating_corner_pair_repair import OUT, REVIEW, alternatives, select
from benchmark.floating_bridges_report import summarize
from benchmark.floating_extra_budget import LOCATION

O2 = 'test-twinspire1-earth-low-left'


def run(panel, case):
    src, ordinary, _, _ = sources(panel)
    name = case['id']
    rel = Path('sky') / name
    image = Path(case['variants']['sky']['image'])
    source = digest(image)
    rgb = np.array(Image.open(image).convert('RGB'))
    current_file = ordinary/rel/'charts.json'
    before_file, shared_file, seam_file = linked(name,panel,src)
    packet, before, shared, seams = [read(f) for f in (current_file,before_file,shared_file,seam_file)]
    assert source == case['variants']['sky']['source_sha256']
    assert all(doc['source_sha256']==source for doc in (packet,before,shared,seams))
    assert shared['baseline_chart_sha256'] == digest(before_file)
    dense, active, corner = [{c['id']:c for c in doc['charts']} for doc in (before,packet,shared)]
    def unchanged(cid):
        if cid not in dense or not dense[cid]['cells'] or cid not in active:
            return False
        a,b = (np.asarray(cs[cid]['H']) for cs in (active,corner))
        return np.allclose(a/a[2,2],b/b[2,2])
    eligible = [s for s in seams['seams'] if s['decision']=='accepted'
                and all(unchanged(s[k]) for k in ('donor_id','receiver_id'))]
    shapes = read(src/rel/'source-handoff/surfaces.json')['planes']
    counts = {c['id']:c for c in read(src/rel/'source-handoff/counts.json')['surfaces']}
    options, proposals, main, edges = alternatives(packet,dense,eligible,shapes,counts,rgb,include_horizontal=False)
    choice = select(options,main,edges)
    folder = OUT/'sparse'/panel
    def checked(path):
        doc = read(path)
        if case['dataset']=='visible-builds-v2' and path==Path(case['frame'])/'frame.json' and 'framing' not in doc:
            audit = read(ROOT/'training/framing-audit-v2.json')['frames']
            proof = next(r for r in audit if r['id']==name)
            assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
            doc = dict(doc,framing=proof)
        return doc
    arms = {}
    for arm in ('keep',*proposals):
        charts = ordinary
        if arm!='keep':
            charts = folder/arm/'charts'
            save(charts/rel/'charts.json',dict(packet,charts=proposals[arm]))
        mode = (charts,'charts.json',False,1,2.5,.5,False)
        target = folder/arm/'assembly'
        with patch.object(floating_cloud_replay,'RUN',src), patch.object(relative_face_round,'ASSEMBLY',target), \
                patch.dict(relative_face_round.MODES,{'connected-atlas':mode}), patch.object(expanded_scores,'read',checked):
            relative_face_round.evaluate((case,'sky','connected-atlas'))
        actual = read(target/LOCATION/name/'score.json')
        if arm=='keep':
            assert main == set(actual['selected_origins'])
        else:
            assert next(set(o['selected']) for o in options if o['arm']==arm) == set(actual['selected_origins'])
        arms[arm] = target
    paired = summarize([case],arms,LOCATION)['rows'][0]
    return dict(paired,panel=panel,chosen=choice,options=options,selected_before=sorted(main),
                hashes={k:digest(f) for k,f in [('chart',current_file),('seams',seam_file),('dense',before_file)]},
                source_sha256=source,pack_sha256={a:digest(t/LOCATION/name/'reconstruction.zip') for a,t in arms.items()})


if __name__ == '__main__':
    cv2.setNumThreads(1)
    cases = [('sky',c) for c in sources('sky')[3]]
    cases.append(('original',next(c for c in sources('original')[3] if c['id']==O2)))
    rows = [run(panel,case) for panel,case in cases]
    save(REVIEW/'sparse.json.gz',rows)
    for r in rows:
        print(r['id'],r['chosen'],r['score']['keep']['geometry']['strict'],r['paired'][r['chosen']])
