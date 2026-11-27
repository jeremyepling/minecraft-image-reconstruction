"""Save a local viewer for the frozen first-hit probes on the reported structures."""
import json

import numpy as np
from PIL import Image


def layers(folder,key,source,details,old,new,has_old,has_new):
    folder.mkdir(parents=True,exist_ok=True)
    photo=np.asarray(source)[::2,::2,:3].copy()
    h,w=photo.shape[:2]
    Image.fromarray(photo).save(folder/(key+'-source.jpg'),quality=93)
    crop=[]
    for name,mask in details:
        y,x=np.nonzero(mask.reshape(h,w))
        crop.append([int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)])
    for i,(_,active) in enumerate(details):
        for stage,exact,fg in [('before',old,has_old),('after',new,has_new)]:
            overlay=photo.copy()
            field=overlay.reshape(-1,3)
            for use,color in [(active&exact,(49,212,145)),
                              (active&~exact&fg,(244,96,84)),(active&~exact&~fg,(82,147,249))]:
                field[use]=np.rint(.42*field[use]+.58*np.array(color))
            suffix='-'+str(i) if i else ''
            Image.fromarray(overlay).save(folder/(key+'-'+stage+suffix+'.jpg'),quality=93)
    return dict(crops=crop,width=w,height=h)


def page(folder,rows):
    data=[dict(id=r['id'],name=r['name'],probes=r['probes'],view=r['view']) for r in rows]
    html="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Visible placement checks</title>
<style>
*{box-sizing:border-box}body{font:16px/1.45 'Segoe UI',sans-serif;background:#14262e;color:#f3f6f5;margin:0}main{max-width:1150px;margin:auto;padding:20px}h1{margin:0;font-size:1.8rem}p{margin:.6rem 0;color:#dce5e4}nav{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:16px 0}button,select{font:inherit;background:#243f4d;color:white;border:1px solid #8ca6ab;border-radius:5px;padding:5px 9px}canvas{display:block;max-width:100%;max-height:65vh;background:#0a1319;margin:auto;image-rendering:pixelated}canvas[hidden]{display:none}figure{margin:12px 0}figcaption{margin:8px 0;color:#d9ebe5}output{font-weight:600}</style>
<main><h1>Visible placement checks</h1><p>Compare the first block seen along each sampled camera ray in the reported region.</p>
<nav><label>Structure <select id="build"></select></label><label>Region <select id="probe"></select></label>
<label>Layer <select id="stage"><option value="source">Source</option><option value="before">Previous default</option><option value="after" selected>New default</option></select></label>
<label><input id="zoom" type="checkbox" checked> Zoom to region</label></nav>
<figure><div style="display:flex;align-items:center;justify-content:center;gap:18px;flex-wrap:wrap"><canvas id="context" width="280" aria-label="Selected region on the source screenshot" style="max-width:38%;margin:0"></canvas><canvas id="plot" style="margin:0"></canvas></div><figcaption>Green marks the correct first block or correctly clear background. Red marks a misplaced foreground block. Blue marks a missing foreground block.</figcaption></figure><output id="counts" aria-live="polite"></output>
</main><script>
const frames=DATA;
const el=id=>document.getElementById(id), build=el('build'), probe=el('probe');
frames.forEach((row,i)=>build.add(new Option(row.name,i)));
function pick(){probe.replaceChildren();frames[build.value].probes.forEach((p,i)=>probe.add(new Option(p.region,i)));draw();}
let version=0;
function draw(){const token=++version,r=frames[build.value],i=Number(probe.value),p=r.probes[i],v=r.view;
el('counts').textContent=p.before+' → '+p.after+' of '+p.samples+' sampled rays correct with the new default.';
const img=new Image();img.onload=()=>{if(token!==version)return;
const [l,t,rr,b]=v.crops[i];const margin=Math.max(6,Math.min(55,Math.round(Math.max(rr-l,b-t)*.12)));
const origin=el('zoom').checked?[Math.max(0,l-margin),Math.max(0,t-margin),Math.min(v.width,rr+margin),Math.min(v.height,b+margin)]:[0,0,v.width,v.height];
const [x,y,x1,y1]=origin,w=x1-x,h=y1-y,canvas=el('plot'),scale=Math.min(7,1000/w,580/h);
canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));
const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(img,x,y,w,h,0,0,canvas.width,canvas.height);
if(!el('zoom').checked){ctx.strokeStyle='#f8cf62';ctx.lineWidth=2;ctx.strokeRect((l-x)*scale,(t-y)*scale,(rr-l)*scale,(b-t)*scale);}
};const stage=el('stage').value;img.src='layers/'+r.id+'-'+stage+(i&&stage!=='source'?'-'+i:'')+'.jpg';
const box=el('context');box.hidden=!el('zoom').checked;
if(!box.hidden){const src=new Image();src.onload=()=>{if(token!==version)return;
const k=box.width/v.width;box.height=Math.round(v.height*k);
const ctx=box.getContext('2d');ctx.drawImage(src,0,0,box.width,box.height);
const [l,t,right,bottom]=v.crops[i];ctx.strokeStyle='#ffd05d';ctx.lineWidth=2;ctx.strokeRect(l*k,t*k,Math.max(3,(right-l)*k),Math.max(3,(bottom-t)*k));};src.src='layers/'+r.id+'-source.jpg';}}
build.onchange=pick;probe.onchange=draw;el('stage').onchange=draw;el('zoom').onchange=draw;pick();
</script></html>"""
    (folder/'viewer.html').write_text(html.replace('DATA;',json.dumps(data)+';'),encoding='utf-8')
