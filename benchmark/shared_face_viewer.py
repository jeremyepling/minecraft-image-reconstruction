"""Write an offline source-image viewer for the five pre-experiment stage maps."""
import json


def write_viewer(folder, views):
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Where placement fails</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#131e23;color:#f3f4ee;font:17px/1.5 "Segoe UI",Arial,sans-serif}
main{max-width:1460px;margin:auto;padding:16px clamp(12px,3vw,44px)}h1{font-size:clamp(22px,3vw,36px);margin:0}
.controls,nav{display:flex;align-items:center;gap:8px 14px;flex-wrap:wrap}label{display:inline-flex;align-items:center;gap:8px}
.controls{margin:12px 0}button,select{font:inherit;color:inherit;background:#293941;padding:5px 10px;border:1px solid #8c9da0;border-radius:4px}
button[aria-pressed=true]{color:#132623;background:#ccf1df}button{cursor:pointer}.frame{display:grid;background:#101818}
.frame img{width:100%;grid-area:1/1;height:auto}p{margin:6px 0}footer a{color:#bcdfd3}
</style>
<main><h1>Where placement fails</h1><div class="controls"><label>Source image <select id="pick"></select></label>
<label>Overlay opacity <input id="mix" type="range" min="0" max="100" value="100"><output id="pct"></output></label></div>
<nav aria-label="Pipeline stage"></nav><p id="legend"></p>
<div class="frame"><img id="base" width="1440" height="900" alt=""><img id="overlay" width="1440" height="900" alt=""></div>
<p id="measures" aria-live="polite"></p><footer><a href="stage-map.jpg">Compare every stage for all five buildings</a></footer></main>
<script>
const views=__DATA__;
const stages=[
['source','Source image','Full input screenshot.'],
['outlines','Surface Finder','Orange traces detected surface boundaries. Cyan traces the surface followed in the stage map.'],
['grid','Block Counter','Yellow marks the selected grid and orange marks other measured grids. A line requires an estimated offset on that axis.'],
['build','Projected build','Teal shows the exported blocks projected onto the source image.'],
['required','Required misses','Blue marks required visible positions that were missed. Gold marks optional horizontal slivers that were missed.'],
['errors','All placement errors','Green marks correct positions, blue marks all missed positions and red marks extras. The frontmost colored face is shown.']];
const el=id=>document.getElementById(id),pick=el('pick'),nav=document.querySelector('nav'),mix=el('mix');
let stage=0;
views.forEach((v,i)=>pick.add(new Option(v.name,i)));
stages.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s[1];b.onclick=()=>{stage=i;draw()};nav.append(b)});
function draw(){
 const v=views[Number(pick.value)],s=stages[stage],prefix='overlays/'+v.prefix+'-';
 el('base').src=prefix+'source.jpg';el('base').alt=v.name+' source image';
 el('overlay').src=prefix+s[0]+'.jpg';el('overlay').alt=s[1]+' for '+v.name;
 mix.disabled=stage===0;el('overlay').style.opacity=mix.value/100;el('pct').value=mix.value+'%';
 [...nav.children].forEach((b,i)=>b.setAttribute('aria-pressed',String(i===stage)));el('legend').textContent=s[2];
 el('measures').textContent=v.name+' · '+v.f1+' strict F1 · '+v.correct+' correct · '+v.required+' required misses · '+v.optional+' optional misses · '+v.extras+' extras';
}
pick.onchange=draw;mix.oninput=draw;draw();
</script></html>"""
    (folder/'viewer.html').write_text(page.replace('__DATA__',json.dumps(views).replace('<',r'\u003c')),encoding='utf-8')
