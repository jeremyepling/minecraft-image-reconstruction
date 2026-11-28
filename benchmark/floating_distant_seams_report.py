"""Summarize decoded exports, fingerprints and the whole-export oracle across frozen arms."""
import argparse
from collections import Counter
from benchmark.floating_distant_seams import GROUPS,REVIEW,OUT,data,base,locations,read,save,digest
from benchmark.floating_one_face_scores import best,FIELDS

ARMS = ('horizontal','source_one','distant')


def aggregate(rows, choices):
    scores = [r['score'][arm] for r,arm in zip(rows,choices,strict=True)]
    result = {}
    for stage,kind in FIELDS:
        s = {k:sum(r[stage][kind][k] for r in scores) for k in ('correct','missed','extras')}
        s['f1'] = 2*s['correct']/(2*s['correct']+s['missed']+s['extras'])
        result[stage+'_'+kind] = s
    for k in ('assigned','fallback'):
        result[k] = sum(r['scores'][k] for r in scores)
    return result


def make(groups):
    panels = {}
    index = {}
    audit = {}
    for group in groups:
        rows = read(REVIEW/(group+'.json.gz'))['rows']
        combined = {r['id']:r for r in read(REVIEW/(group+'-combined.json.gz'))['rows']}
        for row in rows:
            other = combined[row['id']]
            assert other['score']['horizontal'] == row['score']['horizontal']
            for arm in ('contraction','combined'):
                row['score'][arm] = other['score'][arm]
        panels[group] = rows
        audit[group] = [c for r in read(REVIEW/(group+'-audits.json.gz')) for c in r['charts']]
        index[group] = {}
        baseline = base(group) if group!='holdout5' else data(group)[0].parent/'horizontal-cells/assembly'
        for row in rows:
            n = row['id']
            item = {}
            for key,root in zip((*ARMS,'contraction','combined'),(baseline,locations(group)/'assembly',OUT/group/'assembly',OUT/group/'contraction/assembly',OUT/group/'combined/assembly')):
                folder = root/'connected-atlas/sky'/n
                scored = read(folder/'score.json')
                assert scored['pack_sha256'] == digest(folder/'reconstruction.zip')
                item[key] = {k:scored.get(k) for k in ('pack_sha256','reference_sha256','source_sha256','chart_sha256','yaw')}
            for k in ('source_sha256','reference_sha256'):
                assert all(a[k]==item['horizontal'][k] for a in item.values())
            index[group][n] = item
    save(REVIEW/'manifest.json.gz',index)
    if all(g in groups for g in GROUPS):
        panels['previous86'] = [row for g in GROUPS for row in panels[g]]
    if 'previous86' in panels:
        for ds in sorted({r['dataset'] for r in panels['previous86']}):
            panels['previous86/'+ds] = [r for r in panels['previous86'] if r['dataset']==ds]
    summary = {}
    for panel,rows in panels.items():
        choices = best([r['score'] for r in rows],ARMS[:2])
        arms = {a:aggregate(rows,[a]*len(rows)) for a in (*ARMS,'contraction','combined')}
        arms['two-export-oracle'] = aggregate(rows,choices)
        summary[panel] = dict(views=len(rows),arms=arms,oracle_picks=Counter(choices))
        if panel in audit:
            summary[panel].update(offers=len(audit[panel]),accepted=sum(r['accepted'] for r in audit[panel]))
        print(panel,len(rows),[(k,round(s['geometry_strict']['f1']*100,2)) for k,s in arms.items()])
    save(REVIEW/'summary.json',summary)
    lines = ['# Decoded placement results','','All F1 scores are percentages. Exact IDs use the geometry-selected alignment. The oracle selects one complete horizontal or source-one export per image to maximize panel-wide strict position F1.','','| Panel | N | Method | Strict | Facade | Strict + ID | Facade + ID | Correct / extra / missed | ID accuracy on correct | Assigned / fallback |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for panel,s in summary.items():
        for arm,a in s['arms'].items():
            vals = [f"{100*a[stage+'_'+kind]['f1']:.2f}" for stage,kind in FIELDS]
            g = a['geometry_strict']
            lines.append('| '+' | '.join([panel,str(s['views']),arm,*vals,
                ' / '.join(str(g[k]) for k in ('correct','extras','missed')),
                f"{100*a['scores_strict']['correct']/max(1,g['correct']):.2f}",
                f"{a['assigned']} / {a['fallback']}"])+' |')
    (REVIEW/'scores.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('groups',nargs='*',default=GROUPS)
    make(parser.parse_args().groups)


