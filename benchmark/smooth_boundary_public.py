"""Check public CLI masks, saved source crops and the optional placement run against frozen inputs."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark import facade_inset_round as suite
from benchmark.eave_patch_release import dimensions
from benchmark.edge_vote_outline_eval import baseline_pack
from benchmark.mask_blocks import decoded_pack
from benchmark.shared_face_error_budget import current_stages, LATEST
from benchmark.smooth_boundary_interior import OUT
from benchmark.texture_camera_eval import digest, read, save
from engine import smooth_atlas_outline, surface_perimeter
from engine.pillar_seams import native_mask


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    args=parser.parse_args()
    cases={r['source_sha256']:r for r in read(OUT/'interior.json')['rows']}
    jobs={j[1]['id']:j for j in suite.jobs(suite.framed.EVALUATION)}
    expected={r['id']:r for r in read(LATEST)['rows']}
    checks=[]
    for directory in args.runs:
        manifest=read(directory/'build-manifest.json')
        source=manifest['image_sha256']
        row=cases[source]
        job=jobs[row['id']]
        image,before,counts,_=current_stages(job,expected[row['id']])
        lookup={p['id']:p for p in before}
        patches=read(directory/'surfaces.json')['planes']
        settings=manifest['surface_finder']['boundary_settings']['smooth_atlas_outline']
        events=settings['events']
        measured=smooth_atlas_outline.measurement_parts(patches)
        footprint=lambda es:sorted((e['surface'],e['box'],e['pixels']) for e in es)
        assert footprint(events)==footprint(row['radii']['2']['events'])
        for part,original in zip(patches,measured):
            if not part.get('measured_smooth_support'):
                continue
            assert original['support']==lookup[part['id']]['support']
            assert part['outline']==surface_perimeter.outline(part['support'])
            mask=native_mask(part,image.size)
            old=native_mask(original,image.size)
            assert not np.any(old & ~mask)
            assert (mask&~old).sum()==sum(e['pixels'] for e in events if e['surface']==part['id'])
            x,y,r,b=part['source_crop']['xyxy']
            with Image.open(directory/part['source_crop']['path']) as crop:
                assert np.array_equal(np.asarray(crop.getchannel('A'),bool),mask[y:b,x:r])
        check=dict(id=row['id'],through=manifest['through'],source_sha256=source,events=events,
                   source_crop_alpha_and_saved_outlines_agree=True)
        if manifest['through']=='placement':
            assert dimensions(counts)==dimensions(read(directory/'counts.json'))
            prior_hash,prior=baseline_pack(job)
            result=decoded_pack(directory/'reconstruction.zip')
            assert result==dict.fromkeys(prior,'minecraft:smooth_stone')
            check.update(block_count=len(result), placement_sha256=digest(directory/'reconstruction.zip'),
                         frozen_sha256=prior_hash,dimensions_and_export_agree=True)
        checks.append(check)
    save(OUT/'ordinary.json',checks)


if __name__=='__main__':
    main()
