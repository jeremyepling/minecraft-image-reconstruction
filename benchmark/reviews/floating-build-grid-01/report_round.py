
"""Collect the paired placement receipts and render a local review."""
from html import escape
from pathlib import Path

from benchmark.expanded_baseline import read, save, digest

from benchmark.expanded_baseline import STABLE

RUN = STABLE/'runs/floating-build-grid-01'
TREES = STABLE.with_name(STABLE.name+'-worktrees')
EXACT = TREES/'floating-build-floor-bypass/runs/floating-build-grid-01-verified'
GRID = TREES/'floating-build-local-grid/runs/floating-build-grid-01'
GRID_NATIVE = GRID.with_name('floating-build-grid-native-01')
METHODS = {
    'default':'Ordinary default',
    'local':'Build-local grid from image',
    'local-native':'Build-local grid with native wall fallback',
    'exact-floor-base':'Recorded projection, floor on',
    'exact-floating-base':'Recorded projection, floor off',
    'exact-floor-roof':'Recorded projection, roof origin, floor on',
    'exact-floating-roof':'Recorded projection, roof origin, floor off',
    'exact-distance-sky':'Recorded projection and anchor depth, floor off',
}
VARIANTS = dict(pavement='Pavement', plain='Plain backdrop', sky='Open sky')
COHORTS = ['all','visible-builds-v3','decorative-builds-v1']
STYLE = """
:root{color-scheme:dark;--base:#151b20;--table:#1d272e;--line:#384a54;--mute:#b8c8ce;--ink:#eff4f7;--accent:#8fe0df}
*{box-sizing:border-box}body{margin:0;background:var(--base);color:var(--ink);font:16px system-ui,sans-serif}
main{padding:26px clamp(14px,3vw,55px) 50px;max-width:1600px;margin:auto}h1{font-size:clamp(26px,5vw,42px);margin:0}
h2{margin-top:26px;margin-bottom:6px}p{color:var(--mute);line-height:1.48;max-width:80ch;margin:8px 0}
a{color:var(--accent)}small{color:var(--mute)}.stock{margin-top:18px}.table{overflow-x:auto;}
table{border-collapse:collapse;width:100%;background:var(--table);margin:12px 0;text-align:right;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}
select,button{font:inherit;background:var(--table);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:5px 10px}
label{display:inline-flex;align-items:center;gap:8px;margin:10px 12px 0 0}
.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:14px 0}figure{margin:0;background:var(--table);border:1px solid var(--line)}
figure img{display:block;width:100%;height:auto}figcaption{padding:7px 10px}details{margin-top:13px}summary{cursor:pointer}
@media(max-width:700px){.images{grid-template-columns:1fr}}
"""


def score_path(method,variant,case):
    ident,dataset=case['id'],case['dataset']
    if method=='default':
        return RUN/'baseline'/variant/dataset/ident/'score.json'
    if method=='local':
        return GRID/variant/ident/'case.json'
    if method=='local-native':
        return GRID_NATIVE/variant/ident/'case.json'
    return EXACT/method/variant/ident/'score.json'


def evaluation(method,variant,case):
    path=score_path(method,variant,case)
    if not path.is_file():
        return None
    document=read(path)
    row=document['score'] if method in ('local','local-native') else document
    source=case['variants'][variant]['source_sha256']
    if row['source_sha256'] != source:
        raise ValueError(f'Score source does not match the frozen fixture: {path}')
    if row['reference_sha256'] != case['reference_sha256']:
        raise ValueError(f'Score reference does not match the frozen fixture: {path}')
    if digest(path.parent/'reconstruction.zip') != row['pack_sha256']:
        raise ValueError(f'Decoded pack reference mismatch: {path}')
    output=dict(row, method=method, variant=variant,
                receipt=path.as_uri(), image=Path(case['variants'][variant]['image']).as_uri())
    if method in ('local','local-native'):
        output['localEvidence']=document['evidence']
        overlay=path.parent/'source.jpg'
        if overlay.is_file():
            output['overlay']=overlay.as_uri()
    return output


def pooled(rows):
    hits, misses, extras = (sum(part[name] for part in rows)
                            for name in ('correct','missed','extras'))
    denom=2*hits+misses+extras
    return dict(correct=hits,missed=misses,extras=extras,f1=2*hits/denom if denom else 0)


def summary(rows):
    result={}
    for cohort in COHORTS:
        result[cohort]={}
        for method in METHODS:
            tally={}
            for variant in VARIANTS:
                sample=[r for r in rows if r['method']==method and r['variant']==variant
                        and (cohort=='all' or r['dataset']==cohort)]
                if not sample:
                    continue
                tally[variant]=dict(n=len(sample),
                    camera_failures=sum(not s['camera_usable'] for s in sample),
                    local_only=sum(not s['camera_usable'] and bool(s['exported']) for s in sample),
                    empty_with_camera=sum(s['camera_usable'] and not s['exported'] for s in sample),
                    empty=sum(not s['exported'] for s in sample),
                    geometry={mode:pooled([s['geometry'][mode] for s in sample])
                              for mode in ('strict','facade')},
                    typed={mode:pooled([s['scores'][mode] for s in sample])
                           for mode in ('strict','facade')})
            result[cohort][method]=tally
    return result


CONTENTS = """
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Floating builds · Minecraft reconstruction</title>
<style>STYLE</style>
<main>
<h1>Can a build establish its own grid?</h1>
<p>The pavement view is the original capture. The plain and sky versions retain identical visible build pixels and replace their surroundings. The sky composite carries over the original visibility, including any surfaces occluded in that capture.</p>
<label>Cohort <select id="cohort"><option value="all">All matched views</option><option>visible-builds-v3</option><option>decorative-builds-v1</option></select></label>
<section id="pooled"></section>
<p>Placement F1 counts correct, missed and extra visible positions after one whole-build integer translation. Position + material requires the block ID at the same position. The recorded-projection arms use reference camera data and share each view's baseline surface and count handoff. The roof-origin arms check invariance when the local coordinate origin is moved.</p>
<h2>Inspect a matched view</h2>
<label>Structure <select id="case"></select></label>
<label>Image <select id="image-mode"><option value="source">Source</option><option value="ruled">Build-local grid evidence</option></select></label>
<div id="images" class="images"></div>
<section id="detail"></section>
<details class="stock"><summary>Scoring artifacts and fixture provenance</summary>
<p><a href="fixtures.json">Frozen inputs and SHA-256 hashes</a> · <a href="paired-scores.json">Individual and pooled scores</a> · <a href="fixture-preview.jpg">Source image sheet</a></p>
</details>
</main>
<script src="report-data.js"></script>
<script>
const D=window.floatingRound;
const el=id=>document.getElementById(id);
const pct=n=>(n*100).toFixed(2)+'%';
const count=n=>n.toLocaleString();
const method=(id)=>D.methods[id];
const variants=Object.entries(D.variants);
function node(tag,text){
 const e=document.createElement(tag);
 if(text!==undefined)e.textContent=text;
 return e;
}
function table(target,headers,rows){
 target.replaceChildren();
 const wrapper=node('div');
 wrapper.className='table';
 const t=node('table'),thead=node('thead'),head=node('tr'),body=node('tbody');
 headers.forEach(h=>head.append(node('th',h)));
 thead.append(head);t.append(thead,body);
 for(const row of rows){
  const item=node('tr');
  for(const value of row)item.append(node('td',value));
  body.append(item);
 }
 wrapper.append(t);target.append(wrapper);
}
function showSummary(){
 const rows=[];
 const cohort=D.summary[el('cohort').value];
 for(const [key,name] of Object.entries(D.methods)){
  for(const [bg,background] of variants){
   const data=cohort[key][bg];
   if(!data)continue;
   const g=data.geometry.strict;
   rows.push([name,background,data.n,data.camera_failures,data.local_only,data.empty_with_camera,
     pct(g.f1),pct(data.geometry.facade.f1),pct(data.typed.strict.f1),
     [g.correct,g.missed,g.extras].map(count).join(' / ')]);
  }
 }
 table(el('pooled'),['Method','Background','Views','No full projection','Local export only','Camera, empty build',
  'Strict placement','Facade placement','Position + material','Correct / missed / extra'],rows);
}
function inspect(){
 const target=D.cases.find(x=>x.id===el('case').value);
 const imageHost=el('images');
 imageHost.replaceChildren();
 const rows=[];
 for(const [id,label] of variants){
  const local=D.rows.find(x=>x.method==='local-native'&&x.variant===id&&x.id===target.id)||D.rows.find(x=>x.method==='local'&&x.variant===id&&x.id===target.id);
  const src=el('image-mode').value==='ruled'?(local?.overlay||target.images[id]):target.images[id];
  const link=node('a');link.href=src;link.target='_blank';link.rel='noopener';
  const image=node('img');image.src=src;image.alt=target.id+' · '+label;
  link.append(image);
  const card=node('figure');card.append(link,node('figcaption',label));imageHost.append(card);
  for(const r of D.rows.filter(x=>x.variant===id&&x.id===target.id)){
   const g=r.geometry.strict;
   const camera=r.camera_usable?(r.exported?'Supported':'Supported, empty'):(r.exported?'Local grid only':'Unresolved');
   const status=r.localEvidence?.status||r.camera_reason||'';
   rows.push([method(r.method),label,camera,r.exported,pct(g.f1),pct(r.geometry.facade.f1),
              [g.correct,g.missed,g.extras].map(count).join(' / '), status]);
  }
 }
 table(el('detail'),['Method','Background','Projection','Exported','Strict placement','Facade placement','Correct / missed / extra','Decision'],rows);
}
for(const c of D.cases){const option=node('option',c.id);option.value=c.id;el('case').append(option);}
el('cohort').addEventListener('change',showSummary);
el('case').addEventListener('change',inspect);
el('image-mode').addEventListener('change',inspect);
showSummary();inspect();
</script>
"""


def main():
    import json
    fixtures=read(RUN/'fixtures.json')
    rows=[]
    for case in fixtures['cases']:
        for variant in VARIANTS:
            for method in METHODS:
                result=evaluation(method,variant,case)
                if result is not None:
                    rows.append(result)
    report=dict(methods=METHODS,variants=VARIANTS,rows=rows,
                cases=[dict(id=c['id'],dataset=c['dataset'],split=c['split'],
                    images={v:Path(x['image']).as_uri() for v,x in c['variants'].items()}) for c in fixtures['cases']],
                summary=summary(rows))
    save(RUN/'paired-scores.json',report)
    (RUN/'report-data.js').write_text('window.floatingRound = '+json.dumps(report,ensure_ascii=True)+';\n',encoding='utf-8')
    (RUN/'report.html').write_text(CONTENTS.replace('STYLE',STYLE).strip()+'\n',encoding='utf-8')
    print(f'{len(rows)} decoded pack receipts collected from {len(fixtures["cases"])} source viewpoints')


if __name__=='__main__':
    main()
