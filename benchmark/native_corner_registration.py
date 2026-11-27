"""Register native face indices to a previously joined corner receiver using shared observations."""
from collections import Counter
from itertools import permutations

import numpy as np

from benchmark.expanded_baseline import digest, read, save
from benchmark.floating_cloud_replay import check_handoff, source_handoff
from benchmark.native_surface_counter import BASE, OUTPUT, ordinary, project


def correspondence(old, new, matches):
    inverse = np.linalg.inv(old['H'])
    center = np.median([[hit['u']+.5,hit['v']+.5] for hit in matches],axis=0)
    projected = project(inverse,project(np.asarray(new['H']),center+np.vstack(([0,0],np.eye(2)))))
    tangent = projected[1:]-projected[0]
    normed = tangent/np.linalg.norm(tangent,axis=1,keepdims=True)
    axes = max(permutations((0,1)),key=lambda p:sum(abs(normed[i,j]) for i,j in enumerate(p)))
    if min(abs(normed[i,j]) for i,j in enumerate(axes))<.8:
        return None, []
    matrix = np.zeros((2,2),int)
    for i,j in enumerate(axes):
        matrix[i,j] = int(np.sign(normed[i,j]))
    originals = {(cell['u'],cell['v']) for cell in old['cells']}
    votes = []
    for hit in matches:
        native = np.array([hit['u'],hit['v']])
        pos = project(inverse,project(np.asarray(new['H']),native+.5))
        mapped = np.floor(pos).astype(int)
        if tuple(mapped) in originals:
            votes.append(dict(native=native.tolist(),mapped=mapped.tolist(),
                              offset=(mapped-native@matrix).tolist(),
                              residual=(pos-mapped-.5).round(4).tolist()))
    return matrix, votes


def register(old, new, matches, strict):
    evidence = dict(decision='rejected',reason='insufficient shared face centers')
    matrix,votes = correspondence(old,new,matches)
    evidence['votes'] = votes
    if len(votes)<5:
        return old,evidence
    # Split before choosing the offset so the held-out rows test registration.
    votes.sort(key=lambda v:(v['mapped'][1],v['mapped'][0]))
    train,holdout = votes[::2],votes[1::2]
    counts = Counter(tuple(v['offset']) for v in train)
    offset,frequency = counts.most_common(1)[0]
    agreement = [v for v in votes if tuple(v['offset'])==offset]
    mapped = {tuple(v['mapped']) for v in agreement}
    if (frequency < .8*len(train) or len(agreement)<.8*len(votes) or len(mapped)!=len(agreement)
        or any(tuple(v['offset'])!=offset or max(abs(t) for t in v['residual'])>.35 for v in holdout)):
        return old,dict(evidence,reason='shared centers do not validate one integer offset')
    nx,ny = (len({p[i] for p in mapped}) for i in (0,1))
    evidence.update(offset=list(offset),matrix=matrix.tolist(),columns=nx,rows=ny,train=len(train),held_out=len(holdout))
    if ny<3 or (strict and nx<2):
        return old,dict(evidence,reason='insufficient independent rows or columns')
    cells={(c['u'],c['v']):c for c in old['cells']}
    added = updated = 0
    for m in matches:
        u,v = map(int,np.array([m['u'],m['v']])@matrix+offset)
        current = cells.get((u,v))
        if current is not None and not current.get('fallback'):
            continue
        cells[u,v] = dict(current or {},u=u,v=v,confidence=m['correlation'],
                          block_id=m['block_id'],fallback=False,native_quad=m['quad'])
        added += current is None
        updated += current is not None
    return dict(old,cells=list(cells.values())),dict(evidence,decision='accepted',
        reason='integer offset reproduced on held-out shared face centers',added=added,updated=updated)


def predict(case):
    name=case['id']
    baseline = BASE/name/'charts.json'
    base = read(baseline)
    seams = read(BASE/name/'seams.json')
    evidence_file = OUTPUT/'sky'/name/'native-evidence.json'
    native = read(evidence_file)
    native_charts = read(OUTPUT/'sky'/name/'charts.json')
    handoff = source_handoff(name)
    sha = case['variants']['sky']['source_sha256']
    for packet in (base, seams, native, native_charts):
        if packet['source_sha256'] != sha:
            raise ValueError(f'Screenshot hashes disagree for {name}')
        check_handoff(packet, handoff, sha)
    if native['shared_chart_sha256'] != digest(baseline):
        raise ValueError(f'Native evidence was measured against a different shared chart for {name}')
    receivers = {s['receiver_id'] for s in seams['seams'] if s['decision']=='accepted'}
    rows = {r['id']:r for r in native['surfaces'] if r['decision']=='accepted'}
    new = {c['id']:c for c in native_charts['charts']}
    for mode,strict in [('registered-two-columns',True),('registered-seam',False)]:
        charts, evidence = [],[]
        for chart in base['charts']:
            sid=chart['id']
            if sid in receivers and sid in rows:
                chart,record=register(chart,new[sid],rows[sid]['matches'],strict)
                evidence.append(dict(id=sid,**record))
            charts.append(chart)
        path=OUTPUT/mode/'sky'/name
        save(path/'charts.json',dict(base,charts=charts,native_evidence_sha256=digest(evidence_file),
             shared_chart_sha256=native['shared_chart_sha256']))
        save(path/'registration.json',evidence)
        if evidence:
            print(mode,name,evidence[0]['decision'],evidence[0]['reason'],flush=True)


if __name__=='__main__':
    for case in read(ordinary.SHARED/'fixtures.json')['cases']:
        predict(case)
