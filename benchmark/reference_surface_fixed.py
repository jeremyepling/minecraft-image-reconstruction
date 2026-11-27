"""Pool the paired original/true-mask replays for all 104 fixed source images."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import gzip
import json

from benchmark import reference_surface_feedback as oracle
from benchmark.reference_surface_summary import compare


def summarize(folder,out):
    release,_=oracle.releases()
    checked_cases=[]
    maps={}
    jobs=list(oracle.SUITE.jobs(oracle.SUITE.framed.EVALUATION))
    if len(jobs)!=104:
        raise ValueError('Expected all 104 fixed source images')
    for split, frame, _, _ in sorted(jobs,key=lambda j:j[1]['id']):
        key=frame['id']
        checked=release[key]
        pair={m:json.loads((folder/(key+'--'+m+'.json')).read_bytes()) for m in ('baseline','mask')}
        maps[key]={m:r['points'] for m,r in pair.items()}
        for m,row in pair.items():
            if (row['id'],row['mode'],row['source_sha256'])!=(key,m,frame['files']['rgb.png']) or row['source_sha256']!=checked['source_hash']:
                raise ValueError(f'Invalid source binding for {key} {m}')
        ctl,mask=pair['baseline'],pair['mask']
        case=dict(id=key,split=split,layout=checked['layout'],shape_sha256=checked['shape_sha256'],
                  source_sha256=ctl['source_sha256'],
                  released_pack_sha256=checked['pack_sha256'],reproduced_pack=ctl['pack_sha256']==checked['pack_sha256'],
                  published_geometry=checked['geometry'],published_scores=checked['result'],
                  changes=compare(ctl,mask))
        for m,row in pair.items():
            case[m]={k:row[k] for k in ('geometry','scores','regions','pack_sha256','translation','no_overlap_donor')}
        checked_cases.append(case)
    summary={}
    for m in ('baseline','mask'):
        group=[c[m] for c in checked_cases]
        geo=oracle.typed.pooled_scores([c['geometry'] for c in group])
        joint=oracle.typed.pooled_scores([{k:c['scores'][k] for k in ('strict','facade')} for c in group])
        exact=sum(c['scores']['material']['correct'] for c in group)
        positioned=geo['strict']['correct']
        summary[m]=dict(views=len(group),geometry=geo,typed=joint,
                        material=dict(correct=exact,positions=positioned,accuracy=exact/positioned),
                        changes=dict(sum((Counter(c['changes']) for c in checked_cases),Counter())) if m=='mask' else {},
                        improved=sum(c[m]['geometry']['strict']['f1']>c['baseline']['geometry']['strict']['f1'] for c in checked_cases),
                        worse=sum(c[m]['geometry']['strict']['f1']<c['baseline']['geometry']['strict']['f1'] for c in checked_cases),
                        regions=sum(r['regions'] for r in group))
    shapes = {}
    for shape in sorted({c['shape_sha256'] for c in checked_cases}):
        group = [c for c in checked_cases if c['shape_sha256'] == shape]
        shapes[shape] = dict(name=group[0]['layout'],views=len(group),
                             baseline=oracle.typed.pooled_scores([c['baseline']['geometry'] for c in group]),
                             mask=oracle.typed.pooled_scores([c['mask']['geometry'] for c in group]),
                             changes=dict(sum((Counter(c['changes']) for c in group), Counter())))
    summary['shapes'] = shapes
    sites=gzip.compress(json.dumps(maps,separators=(',',':')).encode(),mtime=0)
    (out/'fixed-scored-sites.json.gz').write_bytes(sites)
    summary.update(cases=checked_cases,published_pack_matches=sum(c['reproduced_pack'] for c in checked_cases),
                   sites_sha256=sha256(sites).hexdigest())
    (out/'fixed-confirmation.json').write_text(json.dumps(summary,separators=(',',':'))+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in',dest='folder',required=True,type=Path)
    p.add_argument('--out',default=oracle.ROOT,type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    summarize(a.folder,a.out)

