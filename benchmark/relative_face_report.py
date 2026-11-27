
"""Check the decoded camera-free experiments and build a review image."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import digest, read, save
from benchmark.relative_face_round import MODES, OUTPUT, SHARED

CORE = ('atlas','atlas-dense','atlas-dense-depth','count-exploratory','count-depth-veto',
        'exact-edges','exact-edges-coplanar')
LABELS = {
    'atlas':'Verified atlas sheets',
    'atlas-dense':'Denser atlas sheets',
    'atlas-dense-depth':'Denser atlas sheets + depth check',
    'count-exploratory':'Unverified counter grids',
    'count-depth-veto':'Unverified counter grids + depth check',
    'exact-edges':'Exact grids + shared edges',
    'exact-edges-coplanar':'Exact grids + shared edges + same-plane alignment',
}


def pooled(metrics):
    counts={key:sum(r[key] for r in metrics) for key in ('correct','missed','extras')}
    c,m,e=counts.values()
    return dict(counts,f1=2*c/(2*c+m+e) if (c or m or e) else 0.)


def collect(fixtures):
    rows={}
    for method in CORE:
        group=[]
        for case in fixtures:
            ref=case['reference_sha256']
            sha=case['variants']['sky']['source_sha256']
            folder=OUTPUT/method/'sky'/case['id']
            row=read(folder/'score.json')
            tree,name,*_ = MODES[method]
            original=tree/'sky'/case['id']/name
            if row['reference_sha256']!=ref or row['source_sha256']!=sha:
                raise ValueError(f'Mismatched fixture receipt: {folder}')
            if row['chart_sha256']!=digest(original) or row['pack_sha256']!=digest(folder/'reconstruction.zip'):
                raise ValueError(f'Changed chart input or decoded pack: {folder}')
            group.append(row)
        rows[method]=group
    return rows


def totals(rows):
    result={}
    for method,arm in rows.items():
        result[method]=cohorts={}
        for dataset in ('all','visible-builds-v3','decorative-builds-v1'):
            keep=[r for r in arm if dataset=='all' or r['dataset']==dataset]
            fixed=[r['score']['fixed'] for r in keep if r['score']['fixed'] is not None]
            cohorts[dataset]=dict(
                frames=len(keep),fixed_frames=len(fixed),charted_views=sum(r['charts_seen']>0 for r in keep),
                emitted_views=sum(r['score']['exported']>0 for r in keep),
                joined_views=sum(r['kept_charts']>1 for r in keep),
                edges=sum(sum(j['kind']=='edge' for j in r['joins']) for r in keep),
                coplanar=sum(sum(j['kind']=='coplanar' for j in r['joins']) for r in keep),
                depth_vetoes=sum(r['depth_vetoes'] for r in keep),
                geometry={mode:pooled([r['geometry'][mode] for r in fixed]) for mode in ('strict','facade')},
                material={mode:pooled([r['scores'][mode] for r in fixed]) for mode in ('strict','facade')})
    return result


def example_sheet(fixtures, rows):
    width,height,gap=440,275,15
    page=Image.new('RGB',(3*width+4*gap,756),'#161e24')
    d=ImageDraw.Draw(page)
    fonts=Path('C:/Windows/Fonts')
    headline=ImageFont.truetype(str(fonts/'segoeuib.ttf'),24)
    normal=ImageFont.truetype(str(fonts/'segoeui.ttf'),17)
    note=ImageFont.truetype(str(fonts/'segoeui.ttf'),15)
    d.text((gap,10),'One grid from visible faces, with no camera distance or ground',font=headline,fill='#f2f7f9')
    d.text((gap,42),'Green  placed on the same grid      Blue  measured but unplaced      Gold  matched shared edge',
           font=note,fill='#bacfdc')
    for i,label in enumerate(('Source image','Image-only atlas measurements','Diagnostic with exact face grids')):
        d.text((gap+i*(width+gap),68),label,font=normal,fill='#f2f7f9')
    chosen=('test-castle-01-ashlar-high-right','train-viaduct-02-riverstone-high-right')
    for j,ident in enumerate(chosen):
        case=next(c for c in fixtures if c['id']==ident)
        panels=[Path(case['variants']['sky']['image']),
                *[OUTPUT/m/'sky'/ident/'source-overlay.jpg' for m in ('atlas-dense','exact-edges-coplanar')]]
        y=119+j*335
        d.text((gap,y-24),ident,font=normal,fill='#f2f7f9')
        for i,path in enumerate(panels):
            x=gap+i*(width+gap)
            image=Image.open(path).convert('RGB')
            image.thumbnail((width,height),Image.Resampling.LANCZOS)
            page.paste(image,(x,y))
            if i:
                mode=('atlas-dense','exact-edges-coplanar')[i-1]
                item=next(r for r in rows[mode] if r['id']==ident)
                scores=item['score']['fixed']['geometry']['strict']
                d.text((x,y+282),f"{scores['correct']:,} / {item['score']['visible_truth']:,} correct visible blocks",
                       font=note,fill='#d3eaf1')
    page.save(OUTPUT/'examples.jpg',quality=90,optimize=True)


def main():
    fixtures=read(SHARED/'fixtures.json')['cases']
    rows=collect(fixtures)
    data=totals(rows)
    save(OUTPUT/'summary.json',dict(methods=LABELS,results=data,cases=rows))
    example_sheet(fixtures,rows)
    for method,summary in data.items():
        print(method,summary['all'])
    print(OUTPUT/'examples.jpg')


if __name__=='__main__':
    main()
