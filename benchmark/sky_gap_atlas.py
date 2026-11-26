"""Compare reviewed opening colors to unshaded vanilla full-block sprite colors."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.normal_round import groups
from benchmark.sky_round import occupied, reference_masks
from benchmark.wall_join_test import checked, read_gzip
from engine.atlas_count import load_atlas


def run(out):
    reviews = ROOT/'benchmark/reviews'
    sky = reviews/'sky-01'
    hashes = read(sky/'manifest.json')['artifacts']
    index = reviews/'sky-ambiguity-01/index.json'
    feedback = reviews/'sky-ambiguity-01/feedback.jsonl'
    confirmed = {json.loads(line)['example'] for line in feedback.read_text().splitlines()}
    captures = {c.floor+'/'+c.id: c for g in groups()[0].values() for c in g}
    colors = []
    for card in read(index)['examples']:
        if card['id'] not in confirmed:
            continue
        case = captures[card['frame']]
        path = 'predictions/'+card['frame']+'.json.gz'
        parts = read_gzip(checked(sky/path, hashes[path]))['sky']
        with Image.open(checked(case.path('image'), card['sha256'])) as im:
            mask = occupied(parts, im.size) & reference_masks(case)[1]
            l,t,r,b = card['box']
            pixels = np.array(im.convert('RGB'))[t:b,l:r][mask[t:b,l:r]]
            if len(pixels) != card['marked']:
                raise ValueError('Reviewed region changed')
            colors.append(pixels)
    palette, freq = np.unique(np.concatenate(colors), axis=0, return_counts=True)
    atlas = load_atlas()
    tiles = atlas.tiles.astype(np.int16)
    nearest = []
    exact_colors = exact_pixels = 0
    for k in np.argsort(-freq):
        rgb = palette[k]
        delta = np.abs(tiles-rgb).max(axis=3).min(axis=(1,2))
        if delta.min() == 0:
            exact_colors += 1
            exact_pixels += int(freq[k])
        if len(nearest)<3:
            i = int(delta.argmin())
            j = int(np.abs(atlas.means-rgb).max(axis=1).argmin())
            nearest.append(dict(rgb=rgb.tolist(), observations=int(freq[k]), nearest_texel_block=atlas.ids[i][0],
                                max_channel_difference=int(delta[i]), closest_mean_block=atlas.ids[j][0],
                                closest_mean_rgb=atlas.means[j].tolist()))
    out.parent.mkdir(exist_ok=True,parents=True)
    save(out, dict(index_sha256=digest(index),feedback_sha256=digest(feedback), atlas=atlas.provenance,
                   distinct_reviewed_colors=len(palette), reviewed_pixels=int(sum(freq)), exact_texel_colors=exact_colors,
                   pixels_with_exact_texel_color=exact_pixels, most_common=nearest,
                   scope='Unshaded opaque untinted full-block atlas sprites. Rendered material colors depend on lighting and sampling.'))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    run(cli.parse_args().out)
