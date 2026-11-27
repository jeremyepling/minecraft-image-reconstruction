"""Bundle same-frame source, scoring, export comparison and representative source overlays."""
from collections import Counter
from PIL import Image,ImageDraw
from benchmark.camera_silhouette_round import RUN,REVIEW,CTRL,SOURCE,ROOT,read,save,digest


def pooled(rows):
    out={}
    for span in ('geometry','scores'):
        for mode in ('strict','facade'):
            c={k:sum(row[span][mode][k] for row in rows) for k in ('correct','missed','extras')}
            d=2*c['correct']+c['missed']+c['extras']
            out[span+'_'+mode]={**c,'f1':2*c['correct']/d if d else 0}
    return out


def sheets(records):
    views=[('train-viaduct-02-travertine-rear-oblique','1','Viaduct, one-block rises'),
           ('test-castle-03-ashlar-low-left','2','Castle, two-block rises'),
           ('train-shrine-01-ochre-high-right','unmatched','Shrine, sprite agreement only'),
           ('validation-harbor-03-kiln-rear-oblique','mixed-depth-or-axis','Harbor, mixed turns')]
    sheet=Image.new('RGB',(1000,680),'white')
    for i,(key,target,label) in enumerate(views):
        r=next(r for r in records if r['id']==key)
        pred,score=r['source'],r['score']
        img=Image.open(SOURCE/'visible-builds-v3/data'/r['split']/'frames'/key/'rgb.png').convert('RGB')
        d=ImageDraw.Draw(img)
        for p in pred['polygons']:
            d.line([tuple(x) for x in p+[p[0]]],(100,220,220),width=2)
        chosen=[m['trace'] for m in pred['metric'] if m['sprite_count']]
        for j,(trace,kind) in enumerate(zip(pred['traces'],score['source'])):
            if j in chosen or (not chosen and kind==target):
                coords=[tuple(x) for x in trace['xy']]
                d.line(coords,(255,195,0),width=5)
                for x,y in coords:
                    d.ellipse((x-4,y-4,x+4,y+4),fill='red')
        img.thumbnail((500,315))
        x=(i%2)*500
        y=(i//2)*340
        sheet.paste(img,(x,y))
        ImageDraw.Draw(sheet).text((x+4,y+316),label,fill='black')
    sheet.save(REVIEW/'source-examples.jpg',quality=90)


def main():
    allrows=read(CTRL/'independent-matched-cases.json.gz')
    records=[]
    chosen=[]
    for r in allrows:
        if r['camera_usable']:
            chosen.append(r)
            continue
        d=RUN/r['id']
        source=read(d/'source.json.gz')
        assert not source['candidate'], 'Accepted cameras require source-policy replay'
        forced=read(d/'forced.json.gz')
        opportunities=[f['case'] for f in forced if f['method']=='source-ground']
        best=max([r]+opportunities,key=lambda c:c['geometry']['strict']['f1'])
        chosen.append(best)
        records.append(dict(id=r['id'],split=r['split'],source=source,score=read(d/'score.json.gz'),
                            forced=forced,best_ground_pack=best['pack_sha256']))
    summary={}
    for group in ('train','validation','test'):
        rows=[r for r in records if r['split']==group]
        ps=[r['source'] for r in rows]
        ms=[m for p in ps for m in p['metric']]
        summary[group]=dict(frames=len(rows),orientation_frames=sum(p['orientation'] for p in ps),
            trace_frames=sum(bool(p['traces']) for p in ps),trace_count=sum(len(p['traces']) for p in ps),
            metric_frames=sum(bool(p['metric']) for p in ps),metric_options=len(ms),
            source_turns=Counter(t for r in rows for t in r['score']['source']),
            known_orientation_turns=Counter(t for r in rows for t in r['score']['pose_orientation']),
            sprite_options=sum(bool(m['sprite_count']) for m in ms),direct_probe_singles=sum(m['direct_cells']==1 for m in ms),
            ground_options=sum(m['contact'] for m in ms),accepted=sum(bool(p['candidate']) for p in ps))
    summary['incumbent']={d:pooled([r for r in allrows if r['dataset']==d]) for d in sorted({r['dataset'] for r in allrows})}
    summary['forced_source_ground_chooser_v3']=pooled([r for r in chosen if r['dataset']=='visible-builds-v3'])
    summary['export_count']=Counter(f['method'] for r in records for f in r['forced'])
    summary['correct_height_pairs']=[dict(id=r['id'],arm=f['method'],
        strict=f['case']['geometry']['strict'],strict_ids=f['case']['scores']['strict'],
        relative_px=f.get('native',{}).get('median_check_px'),registered_px=f['case']['camera_validation']['median_reprojection_px'],
        relative_height_error=f['case']['camera_validation']['relative_scale_error'])
        for r in records for f in r['forced'] if f['method']!='source-ground']
    save(REVIEW/'cases.json.gz',records)
    save(REVIEW/'summary.json',summary)
    save(REVIEW/'manifest.json',dict(freeze='493350c',source_base='62611ee',frames=digest(CTRL/'frames.json'),
         controls=digest(CTRL/'independent-matched-cases.json.gz'),per_frame_hashes='cases.json.gz: source.hashes',
         incumbent_pack_hashes=[(r['id'],r['pack_sha256']) for r in allrows],records=digest(REVIEW/'cases.json.gz')))
    sheets(records)


if __name__=='__main__':
    main()

