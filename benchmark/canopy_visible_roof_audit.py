"""Report recovered visible roof-edge, roof-top, underside and support cells by view."""
from collections import Counter,defaultdict

from benchmark.texture_camera_eval import ROOT,checked,read,save
from benchmark.visible_depth_transfer import OUT,rgb_path
from training.visible_labels import visible_labels


def main():
    outputs={'ordinary':OUT/'results/production','learned':ROOT/'runs/place-native-transfer',
             'guarded':ROOT/'runs/guarded-native-transfer','narrow_oracle':ROOT/'runs/oracle-narrow',
             'wide_oracle':ROOT/'runs/oracle-wide'}
    results=[]
    for f in read(OUT/'screen.json')['selected']:
        if 'canopy-' not in f['id']:continue
        png=rgb_path(f)
        pose=read(checked(png.with_suffix('.json'),f['files']['camera.json']))
        raw=read(checked(png.with_name(f['source_id']+'-blocks.json'),f['files']['world-readback.json']))
        states={tuple(b[k] for k in 'xyz'):b for b in raw}
        truth,_,_=visible_labels(pose,states,(0,0,0))
        top=max(b['xyz_world'][1] for b in truth['blocks'])
        groups=defaultdict(set)
        for face in truth['faces']:
            pos=tuple(face['xyz_world'])
            if pos[1]<top:category='support'
            elif face['direction']=='down':category='underside'
            elif face['direction']=='up':category='roof_top'
            else:category='roof_edge'
            groups[category].add(pos)
        for method,folder in outputs.items():
            cells={tuple(x) for x in read(folder/f['id']/'cells.json')['world']}
            for name in ('roof_edge','underside','roof_top','support'):
                results.append(dict(id=f['id'],category=name,method=method,correct=len(cells&groups[name]),
                                    target=len(groups[name])))
    save(ROOT/'benchmark/reviews/canopy-plane-chooser-01/roof-cells.json',results)
    for m in outputs:
        bucket=defaultdict(Counter)
        for row in results:
            if row['method']==m:bucket[row['category']].update({k:row[k] for k in ('correct','target')})
        print(m,dict(bucket))


if __name__=='__main__':main()
