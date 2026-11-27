"""Collect section candidate ablations, occupied-wall exchanges and exported benchmark geometry."""
import argparse
from collections import Counter
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile,ZIP_DEFLATED

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_results import collect
from benchmark.texture_camera_eval import read,save,digest


def run(runs,out):
    baseline=runs/'control'
    baseline.mkdir(exist_ok=True)
    for name in ('summary.json','cases.json'):
        copyfile(runs.parent.parent/'wall-section-round'/'runs/control'/name,baseline/name)
    methods=['operating','transferred','anchored']
    collect(runs,methods,out,'control','028819a')
    oracle=read(runs/'oracle/oracle.json')
    o_index={r['id']:r for r in oracle['frames']}
    named=read(Path('benchmark/reviews/facade-placement-01/worst/ranking.json'))['layouts']
    names={r['representative']['id']:r['name'] for r in named}
    rows={m:{r['id']:r for r in read(runs/m/'cases.json')} for m in methods}
    checks={}
    stage={}
    ablation={}
    for row in oracle['frames']:
        i,s=row['id'],row['split']
        doc=read(runs/'oracle'/s/(i+'.json'))
        shell={}
        for n,allow_depth,allow_origin in [('base',False,False),('origin',False,True),('depth',True,False),('joint',True,True)]:
            shell[n]=sum(max([o['positive'] for o in g if o['positive']==len(o['cells']) and
                              (allow_depth or o['features'][14]==0) and (allow_origin or not any(o['origin']))]+[0])
                         for g in doc['sections'])
        assert shell['joint']==row['pure_chooser_without_joint_visibility']
        for split in [s]+(['fixed'] if s in framed.EVALUATION else []):
            ablation.setdefault(split, Counter()).update(shell)
        checks[i]=dict(split=s,before_final_render=shell,clean_actual=row['actual_oracle_unique_correct'],
                       candidate_positive=row['positive'],candidate_total=row['candidates'])
        if i in names:
            stage[names[i]]=dict(id=i,**checks[i],measured={
                m:{k:rows[m][i]['relative_geometry'][k] for k in ('gained','lost','extra_added','hidden_added')}
                for m in methods})
    save(out/'oracle.json',dict(control=oracle['control'],hash=oracle['control_cases_sha256'],
                                groups=oracle['groups'],ablations=ablation,frames=checks,named=stage))
    ex=read(runs/'exchanges.json')
    result={}
    for row in ex['rows']:
        for key in [row['split']]+(['fixed'] if row['split'] in framed.EVALUATION else []):
            counts=result.setdefault(key,{'oracle':Counter(),'method':Counter(),'eligible_arcs':0,'matches_before_render':0})
            for m in ('oracle','method'):
                counts[m].update(row[m])
            counts['eligible_arcs']+=row['edges']
            counts['matches_before_render']+=row['pairs']
    save(out/'occupied.json',dict(control=ex['control'],groups=result,rows=ex['rows']))
    camera={}
    with ZipFile(out/'comparisons.zip','a',ZIP_DEFLATED) as z:
        for panel in ('native','panel','elevated'):
            here=runs/(panel+'-transferred')
            summary=read(here/'summary.json')
            camera[panel]=summary['groups']['all']
            for file in ('summary.json','cases.json'):
                z.write(here/file,'camera/'+panel+'/'+file)
    results=read(out/'results.json')
    results['archive_sha256']=digest(out/'comparisons.zip')
    results['camera_challenges_transferred']=camera
    with ZipFile(out/'comparisons.zip') as z:
        assert z.testzip() is None
    save(out/'results.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',default='runs',type=Path)
    p.add_argument('--out',default='benchmark/reviews/wall-section-grid-01',type=Path)
    a=p.parse_args()
    run(a.runs.resolve(),a.out)
