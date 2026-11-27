"""Write the offline viewer of ranked structures with release-specific stage images."""
import json


def write_viewer(folder, views, asset_dir, revision, gap):
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lowest-scoring structures · Minecraft reconstruction</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#131e23;color:#f3f4ee;font:17px/1.5 "Segoe UI",Arial,sans-serif}
main{max-width:1460px;margin:auto;padding:16px clamp(12px,3vw,44px)}h1{font-size:clamp(22px,3vw,36px);margin:0}
.controls,nav{display:flex;align-items:center;gap:8px 14px;flex-wrap:wrap}label{display:inline-flex;align-items:center;gap:8px}
.controls{margin:12px 0}button,select{font:inherit;color:inherit;background:#293941;padding:5px 10px;border:1px solid #8c9da0;border-radius:4px}
button[aria-pressed=true]{color:#132623;background:#ccf1df}button{cursor:pointer}.frame{display:grid;background:#101818}
.frame img,.frame svg{width:100%;grid-area:1/1;height:auto}.frame svg{pointer-events:none}p{margin:6px 0}a{color:#bcdfd3}
summary{cursor:pointer;font-weight:600}details{padding:8px 0}button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #e6bd79}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%;margin:8px 0}th,td{padding:5px 10px;text-align:left;border-bottom:1px solid #40525b}
.scroll{overflow-x:auto}td select{max-width:260px}tr.reached{background:#27433b}#pathDescription,#themeLine{color:#d8e4e1}#themeLine{max-width:105ch}
</style>
<main><h1>Lowest-scoring structures</h1><p>Ten distinct structures from the 104 fixed source images, ranked by their lowest strict placement F1. Evaluated default <code>__REV__</code>.</p>
<details open><summary>Which fixes would reach 95% facade-priority F1?</summary>
<p>__PATHHEAD__</p><label>Compare <select id="unit"><option value="layouts">Fewest layouts</option><option value="images">Fewest individual source images</option><option value="displayed">Ten strict-ranked layouts in viewer order</option></select></label>
<p id="pathDescription"></p><div class="scroll"><table><thead><tr><th>Structure</th><th>Views</th><th>Required misses</th><th>Extras</th><th>Cumulative F1</th><th>Most frequent miss types</th><th>Inspect</th></tr></thead><tbody id="pathRows"></tbody></table></div>
<p><strong>Common placement problems.</strong> __THEMES__</p></details>
<details><summary>How much of the facade-priority gap comes from the ten displayed views?</summary><p>__GAP__</p><p>Each required miss and each extra gets an equal share of the gap. A missed optional roof sliver contributes zero.</p></details>
<div id="smoothExamples" class="controls" aria-label="Smooth wall examples">Compare smooth walls</div>
<div class="controls"><label>Source image <select id="pick"></select></label>
<label>Overlay opacity <input id="mix" type="range" min="0" max="100" value="100"><output id="pct"></output></label></div>
<nav aria-label="Pipeline stage"></nav><p id="legend"></p>
<div class="frame"><img id="base" width="1440" height="900" alt=""><img id="overlay" width="1440" height="900" alt=""><svg id="warnings" viewBox="0 0 1440 900" aria-hidden="true"></svg></div>
<p id="issueReview"></p>
<p id="measures" aria-live="polite"></p><p id="contribution"></p><p id="themeLine"></p><footer><a href="__ASSETS__/stage-map.jpg">Compare the ten strict-ranked structures stage by stage</a> · <a href="../pavement-local-01/openings.jpg">Colonnade 01 before and after</a></footer></main>
<script>
const views=__DATA__;
const paths=__PATHS__;
const stages=[
['source','Source image','Full input screenshot.'],
['outlines','Surface Finder','Orange traces detected surface boundaries. Cyan traces the surface followed in the stage map.'],
['issues','Outline issues','Pink dots mark repeated short, sharp turns on a detected outline. The shape is flagged for inspection.'],
['smooth','Smooth wall edges','Teal marks pixels added to the visible outline. Amber marks proposed pixels that the atlas, background or boundary check withheld.'],
['grid','Block Counter','Yellow marks the selected grid and orange marks other measured grids. A line requires an estimated offset on that axis.'],
['reference','Visible reference','Purple marks blocks with a visible face in this source image.'],
['build','Projected build','Teal shows the exported blocks projected onto the source image.'],
['required','Required misses','Blue marks required visible positions that were missed. Gold marks optional horizontal slivers that were missed.'],
['errors','All placement errors','Green marks correct positions, blue marks all missed positions and red marks extras. The frontmost colored face is shown.']];
const el=id=>document.getElementById(id),pick=el('pick'),nav=document.querySelector('nav'),mix=el('mix');
const params=new URLSearchParams(location.hash.slice(1));
let stage=Math.max(0,stages.findIndex(s=>s[0]===params.get('stage')));
function shortName(v){return v.id.replace(/^.*?--(?:confirm--)?/,'').replace(/^legacy-light_gray_concrete-/,'')}
const groups=['Ten lowest strict F1','Other source images in the layout comparisons'].map(label=>{const g=document.createElement('optgroup');g.label=label;pick.append(g);return g});
views.forEach((v,i)=>groups[Number(i>=10)].append(new Option((i<10?(i+1)+'. '+v.name:shortName(v))+' · '+v.f1,i)));
pick.selectedIndex=Math.max(0,views.findIndex(v=>v.id===params.get('view')));
stages.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s[1];b.onclick=()=>{stage=i;draw()};nav.append(b)});
for(const [name,key] of [['Shopfront','legacy-light_gray_concrete-dev-shopfront-plaster-low-left'],
 ['Registry','legacy-light_gray_concrete-holdout-registry-plaster-high-right'],
 ['Portico','legacy-light_gray_concrete-holdout-portico-plaster-high-right'],
 ['Courtyard','legacy-smooth_stone-holdout-courtyard-plaster-high-right']]){
 const index=views.findIndex(v=>v.id===key);if(index<0)continue;
 const b=document.createElement('button');b.textContent=name;
 b.onclick=()=>{pick.selectedIndex=index;stage=stages.findIndex(s=>s[0]==='smooth');draw()};
 el('smoothExamples').append(b);
}
function draw(){
 const v=views[Number(pick.value)],s=stages[stage],prefix='__ASSETS__/'+v.prefix+'-';
 el('base').src=prefix+'source.jpg';el('base').alt=v.name+' source image';
 el('overlay').src=prefix+(s[0]==='issues'?'outlines':s[0]==='smooth'?'source':s[0])+'.jpg';el('overlay').alt=s[1]+' for '+v.name;
 const warnings=el('warnings'), review=el('issueReview');warnings.replaceChildren();review.replaceChildren();
 if(s[0]==='issues'){
  const issues=v.issues||[];
  for(const item of issues)for(const [x,y] of item.native_turn_xy){
   const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
   for(const [k,val] of Object.entries({cx:x,cy:y,r:3.5,fill:'#ff5aaf',stroke:'#381d3c'}))dot.setAttribute(k,val);
   warnings.append(dot);
  }
  warnings.style.opacity=mix.value/100;
  review.textContent=issues.length?'Flagged masks: '+issues.map(i=>i.surface+' ('+i.native_turn_xy.length+' turns)').join(', ')+'. ':'No repeated-turn warning on this source image. A straight cutoff can still miss a wall. ';
 }
 if(s[0]==='smooth'){
  if(v.smooth){
   for(const [data,color] of [[v.smooth.path,'#ffaf52'],[v.smooth.acceptedPath,'#34dfce']]){
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d',data);path.setAttribute('stroke',color);path.setAttribute('fill','none');warnings.append(path);
   }
   warnings.style.opacity=.75*mix.value/100;
   review.textContent=v.smooth.accepted.toLocaleString()+' source pixels added to visible outlines. '+v.smooth.pixels.toLocaleString()+' proposed before checks. ';
  }else{review.textContent='This comparison covers Shopfront, Registry, Portico, and Courtyard. ';}
 }
 if(['issues','outlines','smooth'].includes(s[0])&&['Registry','Shopfront'].includes(v.name)){
  const link=document.createElement('a');link.href='../shopfront-outline-review-01/'+(v.name==='Registry'?'registry-right.jpg':'plaster-gap.jpg');
  link.textContent=v.name==='Registry'?'Inspect the missing upper-right plaster wall in the high-right source image':'Inspect the missing plaster patch in the low-left source image';
  review.append(link);
 }
 if(v.id==='inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right'){
  const link=document.createElement('a');link.href='colonnade-01-window-wall.png';
  link.textContent='Trace the two missing blocks below the window';
  review.append(link);
 }
 mix.disabled=stage===0;el('overlay').style.opacity=mix.value/100;el('pct').value=mix.value+'%';
 [...nav.children].forEach((b,i)=>b.setAttribute('aria-pressed',String(i===stage)));el('legend').textContent=s[2];
 el('measures').textContent=v.name+' · '+v.f1+' strict F1 · '+v.facade+' facade-priority F1 · '+v.correct+' correct · '+v.required+' required misses · '+v.optional+' optional misses · '+v.extras+' extras';
 el('contribution').textContent='This source image accounts for '+v.gap.points.toFixed(3)+' points ('+v.gap.share.toFixed(2)+'%) of the facade-priority gap. All '+v.layoutGap.views+' fixed view(s) of this layout account for '+v.layoutGap.points.toFixed(3)+' points.';
 const t=v.themes;
 el('themeLine').textContent='Required misses in this image: '+t.uncovered+' mostly uncovered, '+t.position+' covered by a face on the correct axis at the wrong position, '+t.direction+' covered by a face on the wrong axis, '+t.ties+' tied. Of the uncovered blocks, '+t.count+' lack a complete block count and '+t.origin+' lack a grid origin on a correctly oriented mask. '+t.background+' extra blocks project mostly onto background.';
}
function common(t){
 const parts=[[t.count,'incomplete block count'],[t.origin,'missing grid origin'],[t.noMask,'missing surface coverage'],[t.maskAxis,'mask facing the wrong axis'],[t.position,'correct face axis at the wrong position'],[t.direction,'wrong projected face axis']];
 return parts.sort((a,b)=>b[0]-a[0]).filter(a=>a[0]>0).slice(0,2).map(a=>a[0]+' · '+a[1]).join(' / ') || 'No required misses';
}
function showPath(){
 const mode=el('unit').value,rows=paths[mode], reached=rows.findIndex(r=>r.score>=95);
 el('pathDescription').textContent='Correcting every required miss and removing every extra in the first '+(reached+1)+' '+(mode==='images'?'images':'layouts')+' would reach '+rows[reached].score.toFixed(2)+'%. All other views stay at their measured scores.';
 const tbody=el('pathRows');tbody.replaceChildren();
 for(const [i,r] of rows.entries()){
  const tr=tbody.insertRow();if(i===reached)tr.className='reached';
  for(const value of [(i+1)+'. '+r.name,r.ids.length,r.missed,r.extras,r.score.toFixed(2)+'%',common(r.themes)])tr.insertCell().textContent=value;
  const cell=tr.insertCell(),menu=document.createElement('select');menu.setAttribute('aria-label','Source views for '+r.name);
  for(const key of r.ids){const v=views.find(v=>v.id===key);menu.add(new Option(shortName(v),key))}
  if(r.ids.length>1)cell.append(menu);else if(mode==='images')cell.append(document.createTextNode(shortName(views.find(v=>v.id===r.ids[0]))+' '));
  const b=document.createElement('button');b.textContent='View';b.onclick=()=>{pick.selectedIndex=views.findIndex(v=>v.id===menu.value);stage=stages.findIndex(s=>s[0]==='errors');draw();pick.scrollIntoView({block:'start'});pick.focus({preventScroll:true})};cell.append(b);
 }
}
el('unit').onchange=showPath;showPath();pick.onchange=draw;mix.oninput=draw;draw();
</script></html>"""
    picked, layouts = gap['selected'], gap['layouts']
    summary = (f"The {picked['views']} displayed source images account for {picked['points']:.2f} of the "
               f"{gap['total']:.2f} percentage points to 100% ({picked['share']:.2f}%). "
               f"All {layouts['views']} fixed views of those layouts together account for "
               f"{layouts['points']:.2f} points ({layouts['share']:.2f}%). Perfecting just the ten displayed images "
               f"would bring the fixed suite to {gap['displayedCeiling']:.2f}%.")
    paths=gap['paths']
    headline=(f"Starting from {gap['baseline']:.2f}%, the smallest sufficient set is {len(paths['layouts'])} layouts "
              f"across {sum(len(r['ids']) for r in paths['layouts'])} fixed views, or {len(paths['images'])} individual images chosen across the suite. "
              "These are perfect-fix scenarios. The table shows the pooled score after each successive fix.")
    page=page.replace('__REV__',revision).replace('__ASSETS__',asset_dir).replace('__GAP__',summary)
    page=page.replace('__PATHHEAD__',headline).replace('__PATHS__',json.dumps(paths).replace('<',r'\u003c'))
    t=gap['themes']
    themes=(f"Across all 104 views, the projected build leaves {t['uncovered']} required blocks mostly uncovered. "
            f"Of these, {t['count']+t['origin']} sit in a correctly oriented surface with incomplete counts or grid offsets. "
            f"At another {t['position']} misses a face points along the right axis at the wrong coordinate. "
            f"At {t['direction']} the visible face points along the wrong axis. {t['ties']} are tied.")
    page=page.replace('__THEMES__',themes)
    (folder/'viewer.html').write_text(page.replace('__DATA__',json.dumps(views).replace('<',r'\u003c')),encoding='utf-8')
