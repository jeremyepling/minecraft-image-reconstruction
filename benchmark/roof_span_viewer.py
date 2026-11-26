"""Build an offline full-frame overlay viewer for the ten ranked layouts."""
import argparse
import json
from pathlib import Path

from PIL import Image

from benchmark.texture_camera_eval import read

PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lowest-scoring structures · Reconstruction overlays</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#12232b;color:#f4f6ef;font:16px/1.45 "Segoe UI",sans-serif}main{max-width:1560px;margin:auto;padding:24px clamp(15px,3vw,54px)}
h1{font-size:clamp(1.65rem,3vw,2.5rem);margin:0}p{margin:.4em 0 1em;color:#c7d4d5}header,.controls,nav{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px}header{justify-content:space-between}header label, .blend{display:flex;align-items:center;gap:8px}
select,button{font:inherit;color:inherit;background:#253941;border:1px solid #587079;border-radius:5px;padding:8px 12px}button{cursor:pointer}button[aria-pressed=true]{background:#d9ede4;color:#182c29;border-color:#d9ede4}select:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid #ffca67}
.controls{justify-content:space-between;margin:.7em 0}.frame{display:grid;background:#0b1519;border:1px solid #49616b;border-radius:6px;overflow:hidden}
.frame img{grid-area:1/1;width:100%;height:auto;display:block}figcaption{margin:8px 0;color:#d5dfdf}figure{margin:0}
footer{margin:14px 0}a{color:#afe0d1}#metrics{color:#f0f4ed;font-variant-numeric:tabular-nums}
@media(max-width:650px){main{padding:14px}nav{gap:6px}button{padding:5px 9px}.controls{align-items:flex-start}}
</style>
<main><header><h1>Lowest-scoring structures</h1><label>Structure <select id="layout"></select></label></header>
<p>Ranked by geometry F1 across all fixed views of each layout. The overlay uses its lowest-scoring photograph.</p>
<div class="controls"><nav aria-label="Pipeline stage"></nav><label class="blend">Overlay opacity <input id="opacity" type="range" min="0" max="100" value="100"><output id="pct">100%</output></label></div>
<figure><div class="frame"><img id="source" width="1440" height="900" alt=""><img id="overlay" width="1440" height="900" alt=""></div><figcaption id="caption"></figcaption></figure>
<div id="metrics" aria-live="polite"></div><footer><a id="sheet" href="">Open all stages for this structure</a> · <a href="comparison.jpg">Open the ten-structure stage map</a></footer></main>
<script>
const layouts=__LAYOUTS__;
const stages=[
 ['source','Source','Original Minecraft screenshot.'],
 ['outlines','Surface Finder','Detected surface boundaries on the original screenshot.'],
 ['grid','Block Counter','The measured grid is projected back into each supported image region.'],
 ['build','Reconstruction','The decoded build projected from the source camera.'],
 ['truth','Visible reference','Recorded geometry visible in this photograph.'],
 ['errors','Placement errors','Green marks correct blocks, red marks extras, and blue marks missed visible blocks after one shared translation.']
];
const el=id=>document.getElementById(id);
const chooser=el('layout'),nav=document.querySelector('nav');
let stage=0;
layouts.forEach((r,i)=>chooser.add(new Option((i+1)+'. '+r.name,i)));
stages.forEach(([key,name],i)=>{const b=document.createElement('button');b.textContent=name;b.onclick=()=>{stage=i;render()};nav.append(b)});
function render(){
 const r=layouts[Number(chooser.value)],prefix='overlays/'+String(r.rank).padStart(2,'0')+'-';
 el('source').src=prefix+'source.jpg';el('source').alt=r.name+' source photograph';
 el('overlay').src=prefix+stages[stage][0]+'.jpg';el('overlay').alt=r.name+' '+stages[stage][1];
 el('opacity').disabled=stage===0;el('overlay').style.opacity=Number(el('opacity').value)/100;
 el('pct').textContent=el('opacity').value+'%';[...nav.children].forEach((b,i)=>b.setAttribute('aria-pressed',String(i===stage)));
 el('sheet').href=r.sheet;el('caption').textContent=stages[stage][2];
 const v=r.view;
 el('metrics').textContent='Layout F1 '+(100*r.f1).toFixed(2)+'% across '+r.frames+' '+(r.frames===1?'view':'views')+'  ·  Shown view '+(100*v.f1).toFixed(2)+'%  ·  '+v.correct+' of '+v.target+' visible blocks  ·  '+v.extras+' extra';
}
chooser.onchange=render;el('opacity').oninput=render;render();
</script></html>
"""


def build(folder):
    layouts=[]
    for r in read(folder/'ranking.json')['layouts']:
        source=folder/'overlays'/f"{r['rank']:02}-source.jpg"
        with Image.open(source) as im:
            native_size=im.size
        for stage in ('source','outlines','grid','build','truth','errors'):
            path=folder/'overlays'/f"{r['rank']:02}-{stage}.jpg"
            with Image.open(path) as im:
                assert im.size==native_size,path
        assert (folder/r['sheet']).is_file()
        layouts.append(dict(rank=r['rank'],name=r['name'],f1=r['f1'],frames=r['frames'],
                            view=r['representative'],sheet=r['sheet']))
    assert len(layouts)==10 and (folder/'comparison.jpg').is_file()
    data=json.dumps(layouts,separators=(',',':')).replace('<',r'\u003c')
    (folder/'viewer.html').write_text(PAGE.replace('__LAYOUTS__',data),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    build(p.parse_args().folder)
