"""Validate matched Floating diagnostics and present their downstream effect."""
from collections import Counter
import html
import json


METRICS = {
    'geometry_strict': 'Strict placement',
    'geometry_facade': 'Facade placement',
    'material_strict': 'Strict position + material',
    'material_facade': 'Facade position + material',
}
COUNTS = ('correct', 'missed', 'extras')
EARLY = {
    'surface-mask': ('Surface Finder', 'Exact native visibility masks',
                     'Replace the surface groups at native pixel centers. Normal and depth estimates still come from the source models.'),
    'surface-plane': ('Surface and depth separation', 'Exact native masks and measured plane fields',
                      'Also supply the recorded plane facing and median optical inverse depth. Integer offsets between surfaces still have to be recovered.'),
    'wall-grid': ('Block Counter', 'Exact visible full-block wall grids',
                  'Supply correctly projected wall cells and count. Estimate materials from the source and infer the joins and heading as usual.'),
}


def aggregate(scores):
    result = {}
    for metric in METRICS:
        counts = Counter()
        for score in scores:
            counts.update({key: score[metric][key] for key in COUNTS})
        result[metric] = dict(counts, f1=200 * counts['correct'] / max(1, sum(counts.values()) + counts['correct']))
    return result


def from_archive(archive, comparison_sha, frames, names, early=None):
    if archive['comparison_sha256'] != comparison_sha:
        raise ValueError('Perfect Stage archive belongs to a different comparison. Replay it for these conditions.')
    arm = archive['base']
    if arm not in names:
        raise ValueError(f'Perfect Stage archive was anchored to an unreported method: {arm}')
    frame_map = {frame['id']: frame for frame in frames if frame['dataset'] != 'decorative-builds-v1'}

    def make_row(title, kind, change, cases, cohort):
        identifiers = [case['id'] for case in cases]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError(f'Expected distinct diagnostic frame IDs: {title}')
        for case in cases:
            frame = frame_map.get(case['id'])
            base = frame['arms'][arm] if frame else None
            if base is None or case['source_sha256'] != frame['source_sha256'] or (
                    case['reference_sha256'] != frame['reference_sha256'] or
                    case['base_pack_sha256'] != base['pack_sha256']):
                raise ValueError(f'Perfect Stage frame does not match the scored source, labels and export: {case["id"]}')
            if any(set(case[side]) != set(METRICS) for side in ('before', 'after')):
                raise ValueError(f'Perfect Stage requires all four scored metrics: {case["id"]}')
            for metric, counts in case['before'].items():
                stored = metric.replace('material_', 'scores_')
                if any(counts[k] != base['scores'][stored][k] for k in COUNTS):
                    raise ValueError(f'Perfect Stage anchor could not reproduce the matched score: {case["id"]}')
        return dict(stage=title, kind=kind, change=change, cohort=cohort, tested=len(cases),
                    coverage=sorted({case['split'] for case in cases}),
                    before=aggregate([case['before'] for case in cases]),
                    after=aggregate([case['after'] for case in cases]), cases=cases)

    rows = []
    test_ids = {case['id'] for case in archive['cases'] if case['split'] == 'test'}
    all_ids = {case['id'] for case in archive['cases']}
    for key, cases in (early or {}).items():
        if any(case['stage'] != key or case['config']['comparison_arm'] != arm or
               case['config']['comparison_sha256'] != case['base']['pack_sha256'] for case in cases):
            raise ValueError(f'Mismatched intervention or comparison arm: {key}')
        adapted = [
            dict(id=case['id'], split=case['split'], source_sha256=case['config']['source_sha256'],
                 reference_sha256=case['config']['references'], base_pack_sha256=case['base']['pack_sha256'],
                 output_pack_sha256=case['output']['pack_sha256'],
                 before={metric.replace('scores_', 'material_'): result for metric, result in case['base']['metrics'].items()},
                 after={metric.replace('scores_', 'material_'): result for metric, result in case['output']['metrics'].items()})
            for case in cases
        ]
        if len({case['id'] for case in adapted}) != len(adapted):
            raise ValueError(f'Expected distinct diagnostic frame IDs: {key}')
        identifiers = {case['id'] for case in adapted}
        if identifiers == all_ids:
            rows.append(make_row(*EARLY[key], adapted, 'Archived comparison'))
        matched = [case for case in adapted if case['id'] in test_ids]
        if test_ids and {case['id'] for case in matched} == test_ids and test_ids != all_ids:
            rows.append(make_row(*EARLY[key], matched, 'Held-out test split'))
        elif identifiers != all_ids:
            rows.append(make_row(*EARLY[key], adapted, f'{EARLY[key][0]} diagnostic subset'))
    for key, (stage, kind, change) in archive['stages'].items():
        cases = [
            dict(id=case['id'], split=case['split'], source_sha256=case['inputs']['source_sha256'],
                 reference_sha256=case['inputs']['reference_sha256'],
                 base_pack_sha256=case['results']['baseline']['pack_sha256'],
                 output_pack_sha256=case['results'][key]['pack_sha256'],
                 before=case['results']['baseline']['scores'], after=case['results'][key]['scores'])
            for case in archive['cases']
        ]
        rows.append(make_row(stage, kind, change, cases, 'Archived comparison'))
        if test_ids and test_ids != all_ids and early:
            rows.append(make_row(stage, kind, change, [case for case in cases if case['id'] in test_ids],
                                 'Held-out test split'))
    rows.sort(key=lambda row: row['cohort'] != 'Archived comparison')
    return dict(anchor=names[arm], rows=rows)


def markdown(data):
    report = [
        '### Perfect from this stage', '',
        f"Source-only anchor is **{data['anchor']}** on each row’s exact views. "
        'The reference is supplied at the named handoff. Each row is an independent experiment.', '',
        '[Open the interactive Perfect Stage comparison](perfect-stage.html).', '',
        '| Handoff and reference input | Tested | Strict placement F1 | Facade placement F1 | Strict position + material F1 | Facade position + material F1 | Correct / missed / extra |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in data['rows']:
        cells = []
        for metric in METRICS:
            before, after = row['before'][metric]['f1'], row['after'][metric]['f1']
            cells.append(f'{before:.2f}% → {after:.2f}% ({after - before:+.2f} pp)')
        counts = ' / '.join(f"{row['after']['geometry_strict'][key]:,}" for key in COUNTS)
        report.append(f"| {row['stage']} · {row['kind']} | {row['tested']} ({row['cohort']}) | " +
                      ' | '.join(cells) + f' | {counts} |')
    return report + ['', 'The heading row chooses among existing possibilities. The writer row verifies the export path. '
                         'Geometry and material gains across rows share some of the same errors.', '']


def viewer(data, cohort):
    labels = json.dumps(METRICS)
    title = html.escape(cohort or 'Visible Builds')
    template = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Floating · Perfect Stage</title>
<style>
*{box-sizing:border-box}body{max-width:1030px;margin:0 auto;padding:32px 20px;color:#eff4fc;background:#101922;font:16px/1.52 system-ui,sans-serif}
h1,h2,h3,p{margin:0 0 .5em}h1{font-size:clamp(2rem,6vw,3.1rem);line-height:1.1}
.intro,.scope,.detail,small{color:#b8cadc}.chooser{margin:1.25rem 0}.chooser label{font-weight:650}
select{font:inherit;padding:.45rem .65rem;color:inherit;background:#1d3043;border:1px solid #68819a;border-radius:5px}
.cards{display:grid;gap:12px}.card{background:#192736;border:1px solid #38516a;border-radius:8px;padding:16px}
.header,.pair{display:flex;justify-content:space-between;flex-wrap:wrap;gap:.3rem 1rem}.delta{color:#ffc97c;font-weight:700}
.pair{margin-top:.65rem}.bar{height:8px;background:#2d455b;border-radius:3px;overflow:hidden}
.bar i{height:100%;display:block;background:#75c5ff}.bar.reference i{background:#ffc97c}
.table{max-height:290px;overflow:auto;margin-top:.45em}table{border-collapse:collapse;width:100%;text-align:left}
td,th{padding:.3em .6em;border-bottom:1px solid #38516a}th{position:sticky;top:0;background:#192736}summary{cursor:pointer;margin-top:.7em}
@media(max-width:600px){body{padding:23px 14px}.table{font-size:.83em}}
</style>
<main>
<p class="scope">COHORT</p>
<h1>Perfect from this stage</h1>
<p class="intro">How much would the final export improve if this handoff had the recorded visible answer? Each row reruns or checks the stages stated on that row.</p>
<p class="detail" id="anchor"></p>
<div class="chooser">
<label for="scope">Views </label><select id="scope"></select>
<label for="metric">Final score </label><select id="metric"></select>
</div>
<div class="cards" id="cards"></div>
</main>
<script id="measurements" type="application/json">PAYLOAD</script>
<script>
const data=JSON.parse(document.querySelector('#measurements').textContent);
const labels=METRICS;
const menu=document.querySelector('#metric');
const scope=document.querySelector('#scope');
const cards=document.querySelector('#cards');
const node=(tag,s,parent,cls)=>{
  const el=document.createElement(tag);
  el.textContent=s;
  if(cls)el.className=cls;
  parent.append(el);
  return el;
};
const fmt=v=>v.toFixed(2)+'%';
const delta=v=>(v>0?'+':'')+v.toFixed(2)+' pp';
const counts=s=>[s.correct,s.missed,s.extras].map(n=>n.toLocaleString()).join(' / ');
document.querySelector('#anchor').textContent='Source-only anchor: '+data.anchor
  +'. Compare within a row. Gains overlap across rows.';
Object.entries(labels).forEach(([key,label])=>menu.add(new Option(label,key)));
const cohorts=new Map(data.rows.map(row=>[row.cohort,row.tested]));
for(const [label,count] of cohorts)scope.add(new Option(label+' ('+count+')',label));
function draw(){
  const metric=menu.value;
  const opened=new Set(Array.from(cards.querySelectorAll('details[open]'),el=>el.dataset.stage));
  cards.replaceChildren();
  for(const row of data.rows){
    if(row.cohort!==scope.value)continue;
    const card=node('section','',cards,'card');
    const head=node('div','',card,'header');
    node('h2',row.stage,head);
    node('span',delta(row.after[metric].f1-row.before[metric].f1),head,'delta');
    node('p',row.kind+' · '+row.tested+' matched views ('+row.coverage.join(', ')+')',card,'scope');
    node('p',row.change,card);
    for(const [name,key,cls] of [['Source-only anchor','before',''],['With reference at this handoff','after','reference']]){
      const score=row[key][metric], pair=node('div','',card,'pair');
      node('span',name,pair);
      node('strong',fmt(score.f1),pair);
      const graph=node('div','',card,'bar '+cls);
      graph.setAttribute('aria-hidden','true');
      node('i','',graph).style.width=score.f1+'%';
    }
    node('small','Correct / missed / extra with reference: '+counts(row.after[metric]),card);
    const detail=node('details','',card);
    detail.dataset.stage=row.stage;
    detail.open=opened.has(row.stage);
    node('summary','Per-view scores',detail);
    const table=node('table','',node('div','',detail,'table'));
    const header=node('tr','',node('thead','',table));
    for(const col of ['View','Source-only','With reference','Change','Correct / missed / extra'])node('th',col,header);
    const body=node('tbody','',table);
    const gain=c=>c.after[metric].f1-c.before[metric].f1;
    for(const c of [...row.cases].sort((a,b)=>gain(b)-gain(a))){
      const entry=node('tr','',body);
      for(const value of [c.id,fmt(c.before[metric].f1),fmt(c.after[metric].f1),delta(gain(c)),counts(c.after[metric])])
        node('td',value,entry);
    }
  }
}
menu.addEventListener('change',draw);
scope.addEventListener('change',draw);
draw();
</script>
</html>
"""
    visual = dict(anchor=data['anchor'], rows=[
        dict(row, cases=[dict(id=case['id'], before=aggregate([case['before']]), after=aggregate([case['after']]))
                         for case in row['cases']]) for row in data['rows']
    ])
    payload = json.dumps(visual, ensure_ascii=False, separators=(',', ':')).replace('<', r'\u003c')
    return template.replace('COHORT', title).replace('METRICS', labels).replace('PAYLOAD', payload)
