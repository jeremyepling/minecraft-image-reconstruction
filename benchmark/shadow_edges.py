"""Compare image-only lighting and texture seam evidence against Java 26.3 readbacks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from benchmark.textures import extract_split
from engine.illumination import edge_evidence,MATERIAL_THRESHOLD


def collect(root,split,examples=None):
    rows,_=extract_split(root,['normal','small','warm-shader'],split)
    views=defaultdict(dict)
    for r in rows:
        if r['crop']=='known':
            views[r['view'],r['condition']][r['xyz']]=r
    result=[]
    for (view,condition),coords in views.items():
        pairs=[]
        for (x,y,z),a in coords.items():
            for neighbor in ((x+1,y,z),(x,y+1,z)):
                if neighbor in coords:
                    pairs.append(('adjacent',a,coords[neighbor]))
            # Supplemental paired-light controls. Separated physically, same
            # camera, same material, and at least two measured sky-light levels apart.
            options=[b for b in coords.values() if b['texture']==a['texture'] and b['sky']>=a['sky']+2]
            if options:
                b=min(options,key=lambda b:sum((v-w)**2 for v,w in zip(a['xyz'],b['xyz'])))
                pairs.append(('paired_light',a,b))
        scores=edge_evidence(np.stack([a['tile'] for _,a,b in pairs]),np.stack([b['tile'] for _,a,b in pairs]))
        for j,(kind,a,b) in enumerate(pairs):
            if examples is not None and condition=='normal' and a['angle']=='front' and kind=='adjacent':
                # Choose two legible illustrations of the evaluated classes.
                # Class labels are available only to the scorer and renderer.
                same=a['texture']==b['texture']
                score=scores['material_score'][j]
                suitable=(same and a['sky']!=b['sky'] and score<MATERIAL_THRESHOLD) or (
                    not same and a['sky']==b['sky'] and score>=MATERIAL_THRESHOLD)
                if suitable:
                    examples.append((same,a,b,float(score)))
            result.append(dict(kind=kind,condition=condition,angle=a['angle'],different=a['texture']!=b['texture'],
                               light_difference=abs(a['sky']-b['sky']),view=view,
                               observable=bool(scores['observable'][j]),model=str(scores['shading_model'][j]),
                               luminance=float(scores['luminance_change'][j]),texture=float(scores['material_score'][j])))
    return result


def metrics(rows,method,threshold):
    actual=np.array([r['different'] for r in rows],dtype=bool)
    proposed=np.array([r[method]>=threshold for r in rows],dtype=bool)
    hit=int((actual&proposed).sum()); keep=int((~actual&~proposed).sum())
    positive=int(actual.sum());negative=len(actual)-positive
    return dict(different=positive,correct_seams=hit,same=negative,correct_same=keep,
                balanced_accuracy=.5*(hit/positive+keep/negative) if positive and negative else None)


def draw_examples(records,out):
    same=[r for r in records if r[0]]
    different=[r for r in records if not r[0]]
    # Strong brightness difference for the shadow, comparable brightness for
    # the material seam. Both panels contain unedited in-game source swatches.
    def contrast(r):
        return abs(float(r[1]['tile'].mean())-float(r[2]['tile'].mean()))
    shade=max(same,key=contrast)
    material=min(different,key=lambda r:(max(r[1]['tile'].mean(),r[2]['tile'].mean())>110,contrast(r)))
    im=Image.new('RGB',(875,460),'#192229')
    d=ImageDraw.Draw(im)
    def font(size,bold=False):
        for name in (('arialbd.ttf','DejaVuSans-Bold.ttf') if bold else ('arial.ttf','DejaVuSans.ttf')):
            try: return ImageFont.truetype(name,size)
            except OSError: pass
        return ImageFont.load_default(size=size)
    def text(x,y,msg,size=16,bold=False,color='#e4ebef'):
        d.text((x,y),msg,font=font(size,bold),fill=color)
    text(18,10,'Shadow or a change of material?',24,True)
    text(18,45,'Adjacent front faces in Minecraft Java 26.3 at clear noon. Texture score above 0.505 marks a material seam.',15)
    for x,label in ((18,'Face A'),(138,'Face B'),(258,'B relit to A')):
        text(x,77,label,16,True)
    for row,(is_same,a,b,_) in enumerate((shade,material)):
        if a['tile'].mean()<b['tile'].mean():
            a,b=b,a
        A,B=a['tile'],b['tile']
        s=edge_evidence(A[None],B[None],return_gain=True)
        relit=np.clip((B.astype(float)+.025*255)/s['gain_map'][0,:,:,None]-.025*255,0,255).astype('uint8')
        y=106+row*160
        for x,swatch in ((18,A),(138,B),(258,relit)):
            im.paste(Image.fromarray(swatch).resize((112,112),Image.Resampling.NEAREST),(x,y))
        text(389,y,'Same block under different light' if is_same else 'Different blocks under the same skylight',19,True)
        name=lambda r:r['texture'].replace('_',' ')
        text(389,y+34,f"A  {name(a)}   /   sky light {a['sky']}")
        text(389,y+58,f"B  {name(b)}   /   sky light {b['sky']}")
        text(389,y+89,f"Material score {s['material_score'][0]:.3f}     Texture agreement {s['texture_agreement'][0]:.2f}",15,color='#97d6bf')
    text(18,428,'A and B are rectified screenshot crops. Relighting is a pixel estimate. Block identity and sky light come from the game.',14)
    out.parent.mkdir(exist_ok=True,parents=True)
    im.save(out)


def run(root,out,examples=None):
    dev=collect(root,'dev')
    records=[] if examples else None
    test=collect(root,'test',records)
    if examples:
        draw_examples(records,examples)
    report=dict(release='26.3',task='Boundary classification with supplied rectified neighboring face patches',
                unit='Pair observations repeated across angles and render conditions',methods={})
    same_shaded=[r for r in test if r['kind']=='adjacent' and not r['different'] and r['light_difference']]
    report['lighting_models_on_same_material_with_changed_skylight']={
        m:sum(r['model']==m for r in same_shaded) for m in ('uniform','smooth','step')}
    report['adjacent_pairs_with_observable_texture_or_hue']=sum(r['observable'] for r in test if r['kind']=='adjacent')
    report['limits']=[
        'World readback establishes block identity and skylight. The fitted spatial lighting shape has no physical shadow-mask ground truth.',
        'Known camera-projected neighboring faces are inputs. This experiment does not detect seams in an unlabeled full frame.',
        'Distinct neutral flat materials can be inseparable from changes in illumination when neither side has texture.']
    for method in ('luminance','texture'):
        rows=[r for r in dev if r['kind']=='adjacent']
        thresholds=np.quantile([r[method] for r in rows],np.linspace(.01,.99,125))
        threshold=float(max(thresholds,key=lambda t:metrics(rows,method,t)['balanced_accuracy']))
        scored=lambda rows:metrics(rows,method,threshold)
        adjacent=[r for r in test if r['kind']=='adjacent']
        report['methods'][method]=dict(threshold_from_dev=threshold,dev=scored(rows),test=scored(adjacent),
            test_changed_skylight=scored([r for r in adjacent if r['light_difference']]),
            test_paired_light=scored([r for r in test if r['kind']=='paired_light']),
            by_condition={k:scored([r for r in adjacent if r['condition']==k]) for k in sorted({r['condition'] for r in adjacent})},
            by_angle={k:scored([r for r in adjacent if r['angle']==k]) for k in sorted({r['angle'] for r in adjacent})})
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({m:{k:v[k] for k in ('threshold_from_dev','test','test_paired_light','test_changed_skylight')}
                      for m,v in report['methods'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--examples',type=Path,help='Write an illustrated pair from each boundary class')
    args=p.parse_args()
    run(args.fixture,args.out,args.examples)
