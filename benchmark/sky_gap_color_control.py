"""Check sky-colored gaps against rendered material colors and a separate 72-view building set."""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.normal_round import groups
from benchmark.sky_round import occupied, reference_masks
from benchmark.wall_join_test import checked, read_gzip
from engine.sky_filter import sky_masks


def pack(rgb):
    r, g, b = rgb.astype(np.int32).T
    return (r << 16) | (g << 8) | b


def rgb(value):
    return [int(value >> 16), int((value >> 8) & 255), int(value & 255)]


def disputed_colors():
    review = ROOT/'benchmark/reviews/sky-01'
    decisions = ROOT/'benchmark/reviews/sky-ambiguity-01/index.json'
    hashes = read(review/'manifest.json')['artifacts']
    rows = {r['key']:r for r in read(checked(review/'cases.json',hashes['cases.json']))}
    captures = {c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    records, colors = [], Counter()
    for item in read(decisions)['examples'][:4]:
        key = item['frame']
        case, row = captures[key], rows[key]
        with Image.open(checked(case.path('image'), item['sha256'])) as source:
            image = source.convert('RGB')
        path = row['predictions']
        parts = read_gzip(checked(review/path, hashes[path]))['sky']
        _, sky, _ = reference_masks(case)
        disputed = occupied(parts,image.size) & sky
        l,t,r,b = item['box']
        selected = np.asarray(image)[t:b,l:r][disputed[t:b,l:r]]
        assert len(selected) == item['marked']
        hist = Counter(map(int,pack(selected)))
        colors.update(hist)
        records.append(dict(id=item['id'],frame=key,marked=len(selected),source_sha256=item['sha256'],
                            top_rgb=[dict(rgb=rgb(v),pixels=n) for v,n in hist.most_common(5)]))
    return colors, records, digest(decisions)


def score(archive, out):
    from engine.sky_gap import gap_mask
    targets, examples, selection_hash = disputed_colors()
    shard = read(ROOT/'training/visible-builds-v1.json')['shards']['train']
    checked(archive, shard['sha256'])
    interior_colors = defaultdict(Counter)
    face_pixels, inside_pixels, base_hits, added_hits, added_interior = (Counter() for _ in range(5))
    rows = []
    with ZipFile(archive) as zipped:
        for line in zipped.read('frames.jsonl').splitlines():
            key = json.loads(line)['directory']
            with Image.open(io.BytesIO(zipped.read(key+'/rgb.png'))) as src:
                image = src.convert('RGB')
            blue, cloud = sky_masks(image)
            additions = gap_mask(image, blue, cloud) & ~(blue | cloud)
            labels = json.loads(zipped.read(key+'/labels.json'))
            with np.load(io.BytesIO(zipped.read(key+'/labels.npz')), allow_pickle=False) as fields:
                ids = fields['face_id']
            blocks = {r['id']:r['block_id'] for r in labels['blocks']}
            mats = sorted(set(blocks.values()))
            lut = np.zeros(max(f['id'] for f in labels['faces'])+1,int)
            for face in labels['faces']:
                lut[face['id']] = 1 + mats.index(blocks[face['block']])
            materials = lut[ids]
            inside = ids>0
            # Check material colors only inside a single unit face, avoiding label/raster edge disagreements.
            pad = np.pad(ids,1)
            h,w=ids.shape
            for dy in range(3):
                for dx in range(3):
                    inside &= ids==pad[dy:dy+h,dx:dx+w]
            photo = np.asarray(image)
            for i,mat in enumerate(mats,1):
                material = materials==i
                clean = material & inside
                codes, counts = np.unique(pack(photo[clean]),return_counts=True)
                interior_colors[mat].update(dict(zip(map(int,codes), map(int,counts))))
                face_pixels[mat]+=int(material.sum())
                inside_pixels[mat]+=int(clean.sum())
                base_hits[mat]+=int((material & (blue|cloud)).sum())
                added_hits[mat]+=int((material & additions).sum())
                added_interior[mat]+=int((clean & additions).sum())
            rows.append(dict(frame=key,added_pixels=int(additions.sum()),
                             added_on_building=int((additions & (ids>0)).sum())))
    use = {}
    for mat, hist in interior_colors.items():
        use[mat] = np.array([rgb(v) for v in hist],int)
    nearest = []
    for value, n in targets.most_common():
        color = rgb(value)
        options=[]
        for mat, pixels in use.items():
            dist = np.abs(pixels-color).max(axis=1)
            best = dist.argmin()
            options.append((int(dist[best]),mat,pixels[best].tolist()))
        best = min(x[0] for x in options)
        nearest.append(dict(rgb=color, marked_pixels=n, closest_max_channel_delta=best,
                            closest=[dict(block=mat,rgb=c) for d,mat,c in options if d==best]))
    stats=dict(building_pixels=sum(face_pixels.values()),interior_pixels=sum(inside_pixels.values()),
               distinct_materials=len(inside_pixels), old_building_hits=sum(base_hits.values()),
               added_building_hits=sum(added_hits.values()),added_interior_hits=sum(added_interior.values()),
               marked_pixels=sum(targets.values()),marked_with_exact_material_color=sum(x['marked_pixels'] for x in nearest if x['closest_max_channel_delta']==0))
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(code=revision(), shard=shard, test='L-infinity RGB distance to visible unit-face interiors on the training shard',
                  selection_sha256=selection_hash, examples=examples,summary=stats,colors=nearest,
                  by_material={m:dict(pixels=face_pixels[m],interior_pixels=inside_pixels[m],old_hits=base_hits[m],
                                      added_hits=added_hits[m],added_interior_hits=added_interior[m]) for m in sorted(face_pixels)},frames=rows))
    print(stats)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    score(args.archive,args.out)
