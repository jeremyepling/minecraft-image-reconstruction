"""Train a one-split placement chooser and measure best-case selection on the same saved builds."""
import argparse
from collections import Counter
from pathlib import Path

from benchmark.mask_blocks import decoded_pack
from benchmark.nearby_thin_comparison import EVALUATION
from benchmark.texture_camera_eval import checked, digest, read, save

FEATURES = ('added', 'removed', 'added_fraction', 'touch_fraction', 'height_change')
STEPS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))


def predictions(folders):
    result, manifests = {}, {}
    for folder in folders:
        meta=read(folder/'summary.json')
        path=checked(folder/'cases.json',meta['cases_sha256'])
        manifests[folder.name]=digest(path)
        for row in read(path):
            key=(row.get('split','transfer'),row['id'])
            if key in result:
                raise ValueError(f'Duplicate frame {key}')
            root=folder/key[0] if 'split' in row else folder
            result[key]=(row,root/key[1]/'reconstruction.zip')
    return result,manifests


def describe(before,after):
    added=after-before
    touch=sum(any((x+dx,y+dy,z+dz) in before for dx,dy,dz in STEPS) for x,y,z in added)
    tallest=lambda cells:max((p[1] for p in cells),default=0)
    return dict(added=len(added),removed=len(before-after),added_fraction=len(added)/max(1,len(before)),
                touch_fraction=touch/max(1,len(added)),height_change=abs(tallest(after)-tallest(before)))


def score(rows, choose):
    correct=placed=target=hidden=0
    choices=Counter()
    for row in rows:
        name=choose(row)
        m=row['stats'][name]
        choices[name]+=1
        correct+=m['correct']
        target+=m['target']
        placed+=m['placed']
        hidden+=m['hidden_extras']
    return dict(frames=len(rows),correct=correct,target=target,extras=placed-correct,hidden_extras=hidden,
                f1=2*correct/(target+placed),choices=dict(choices))


def route(rule,row):
    name=rule['method']
    if 'feature' not in rule:
        return name
    value=row['features'][name][rule['feature']]
    take=value<=rule['threshold'] if rule['direction']=='at_most' else value>rule['threshold']
    return name if take else 'default'


def fit(rows,names):
    # Prefer the current default or an unconditional order when a split makes no difference.
    rules=[dict(method=name) for name in names]
    for name in names[1:]:
        for feature in FEATURES:
            values=sorted({row['features'][name][feature] for row in rows})
            for threshold in values[:-1]:
                for direction in ('at_most','above'):
                    rule=dict(method=name,feature=feature,threshold=threshold,direction=direction)
                    used={r['family'] for r in rows if r['different'][name] and route(rule,r)==name}
                    if len(used)>=2:
                        rules.append(rule)
    return max(rules,key=lambda rule:score(rows,lambda r:route(rule,r))['f1'])


def oracle(rows,names):
    # Optimize pooled F1, which weights frames by their predicted and true block counts.
    ratio=0.
    for _ in range(30):
        def best(row):
            def gain(name):
                m=row['stats'][name]
                return 2*m['correct']-ratio*m['placed']
            return max(names,key=gain)
        totals=score(rows,best)
        if abs(totals['f1']-ratio)<1.e-12:
            return totals
        ratio=totals['f1']
    raise RuntimeError('Pooled F1 oracle did not converge')


def run(methods,out):
    names=[name for name,_ in methods]
    assert names[0]=='default' and len(set(names))==len(names)
    inputs,hashes={},{}
    for name,paths in methods:
        inputs[name],hashes[name]=predictions(paths)
    baseline=inputs['default']
    assert all(rows.keys()==baseline.keys() for rows in inputs.values())
    rows=[]
    for key,(base,basefile) in baseline.items():
        before=set(decoded_pack(checked(basefile,base['pack_sha256'])))
        features,stats,different={},{},{}
        for name,table in inputs.items():
            row,path=table[key]
            for field in ('rgb_sha256','reference_hashes'):
                assert base[field]==row[field],(key,field)
            if 'input_hashes' in base:
                assert base['input_hashes']==row['input_hashes']
            after=before if row['pack_sha256']==base['pack_sha256'] else set(decoded_pack(checked(path,row['pack_sha256'])))
            features[name]=describe(before,after)
            different[name]=before!=after
            stats[name]=row['relative_geometry']
        rows.append(dict(id=key[1],split=key[0],family=base.get('family'),features=features,stats=stats,different=different))
    train=[r for r in rows if r['split']=='train']
    assert len(train)==72
    families=sorted({r['family'] for r in train})
    assert len(families)==6
    rules={f:fit([r for r in train if r['family']!=f],names) for f in families}
    rule=fit(train,names)
    groups={'train':train,'fixed':[r for r in rows if r['split'] in EVALUATION]}
    for split in ('extra','new','transfer'):
        selected=[r for r in rows if r['split']==split]
        if selected:
            groups[split]=selected
    assert len(groups['fixed'])==104
    report={}
    for group,rr in groups.items():
        report[group]=dict(default=score(rr,lambda r:'default'),trained=score(rr,lambda r:route(rule,r)),
                           oracle=oracle(rr,names),each={n:score(rr,lambda r:n) for n in names})
    report['train']['family_held_out']=score(train,lambda r:route(rules[r['family']],r))
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(methods=names,case_manifests=hashes,rule=rule,held_out_rules=rules,groups=report,
                  switches=[dict(id=r['id'],split=r['split'],method=route(rule,r)) for r in rows
                            if r['different'][route(rule,r)]]))
    print(rule,{g:[round(v[k]['f1'],5) for k in ['default','trained','oracle']] for g,v in report.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',nargs='+',action='append',required=True,metavar='METHOD_AND_RUN_FOLDERS')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    run([(name,list(map(Path,paths))) for name,*paths in args.method],args.out)
