"""Challenge only newly completed envelope-refit sizes with the frozen vanilla atlas test."""
import argparse
from pathlib import Path

from PIL import Image

from benchmark.camera_round import ROOT,digest,read,revision,save
from benchmark.normal_round import groups
from benchmark.portico_atlas import SETTINGS,atlas,predict
from benchmark.texture_count_round import SOURCE,summarize
from benchmark.wall_join_test import checked,read_gzip
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface


REFIT=ROOT/'benchmark/reviews/repeat-rectify-01'


def challenge(sample, fitted):
    withheld=dict(fitted,counts=[None,None],phase=[None,None],evidence=dict(fitted['evidence']))
    proposals=[]
    for span,options in zip(fitted['counts'],fitted['evidence']['alternatives']):
        # Reserve one of the three original slots for the image-derived refit pitch.
        # This also lets an aspect-transfer pitch enter the challenge as a hypothesis.
        ranked=sorted(options,key=lambda o:o.get('agreement',0)+o.get('native_support',0),reverse=True)
        alternatives=[o for o in ranked if abs(span-o['count'])>1e-6]
        proposals.append([dict(count=span),*alternatives[:SETTINGS['axis_candidates']-1]])
    withheld['evidence']['alternatives']=proposals
    return predict(sample,withheld)


def run(out):
    manifest=read(REFIT/'manifest.json')
    src=read(checked(SOURCE/'manifest.json',manifest['source_manifest_sha256']))
    refs={r['key']:r for r in read(checked(SOURCE/'cases.json',src['artifacts']['cases.json']))}
    rows=read(checked(REFIT/'cases.json',manifest['artifacts']['cases.json']))
    cases={c.floor+'/'+c.id:c for grp in groups()[0].values() for c in grp}
    tested=[]
    for row in rows:
        key=row['key']
        meta=refs[key]
        original=read_gzip(checked(SOURCE/meta['predictions'],src['artifacts'][meta['predictions']]))['current']
        parts={p['id']:p for p in original}
        archive=read_gzip(checked(REFIT/row['predictions'],manifest['artifacts'][row['predictions']]))
        with Image.open(checked(cases[key].path('image'),row['sha256'])) as opened:
            photo=opened.convert('RGB')
        if [q['proposal'] for q in archive]!=[r['proposal'] for r in row['surfaces']]:
            raise ValueError('Changed proposal order')
        for p,score in zip(archive,row['surfaces']):
            old,refit=p['methods']['current'],p['methods']['rectify']
            score['guess']['confirmed']=score['guess']['current']
            if None not in old['counts'] or 'quad' not in refit:
                continue
            part=parts[p['proposal']]
            sample=sample_surface(photo,dict(part,quad=refit['quad']))
            fitted=count(sample)
            if fitted['counts']!=refit['counts']:
                raise ValueError('Could not reproduce texture counts on the saved fitted chart')
            verified=challenge(sample,fitted)
            atlas_size=dimensions(part,refit['quad'],photo.size,verified['counts'])['blocks']
            accepted=atlas_size==score['guess']['rectify']
            if accepted:
                score['guess']['confirmed']=atlas_size
            # The recorded reference and actual dimensions are reporting data only.
            tested.append(dict(key=key,sha256=row['sha256'],proposal=p['proposal'],baseline=score['guess']['current'],
                               refit=score['guess']['rectify'],atlas_size=atlas_size,confirmed=accepted,
                               reference=score.get('reference'),actual=score.get('actual'),
                               evidence=verified['evidence']['atlas_check']))
    names=['current','rectify','confirmed']
    scores={group:summarize([r for r in rows if group in ('all',r['cohort'])],names)
            for group in ('all','development','previously_reviewed_regression')}
    save(out,dict(scope='Only previously unresolved proposals yielding a new complete refit. Frozen atlas settings, '
                        'same fitted crop, spans withheld. Challenge full, half and double pitch. Require identical final size.',
                  code=revision(),source_manifest_sha256=digest(REFIT/'manifest.json'),settings=SETTINGS,
                  atlas=atlas().provenance,scores=scores,tested=tested))
    print(scores['all'])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    run(cli.parse_args().out)
