"""Find atlas motifs without assuming the cell contains exactly one complete block."""
import json

import numpy as np
from PIL import Image

from engine.geometry import bilinear

LUMA = np.array([.213, .715, .072], np.float32)
OFFSETS = np.array([(x, y) for y in range(-3, 4) for x in range(-3, 4)], np.float32)
PITCH = np.geomspace(.8, 6, 19)


def normalize(values):
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(1., np.linalg.norm(centered, axis=-1, keepdims=True))


def bank(atlas_dir, allowed):
    data = json.loads((atlas_dir / 'atlas.json').read_text())
    names, tiles = [], []
    with Image.open(atlas_dir / data['atlas']) as atlas:
        for entry in data['textures']:
            name = entry['id'].split('/')[-1]
            if name not in allowed or entry['source_size'] != [16, 16]:
                continue
            x, y, w, h = entry['rect']
            tile = atlas.crop((x, y, x+w, y+h)).resize((16, 16), Image.Resampling.BOX)
            if tile.getchannel('A').getextrema()[0] == 255:
                tiles.append(np.asarray(tile.convert('RGB')))
                names.append(name)
    if not tiles:
        return None
    tiles = np.asarray(tiles, np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(tiles @ LUMA, (7, 7), axis=(1, 2))
    return np.asarray(names), tiles.mean(axis=(1, 2)), normalize(windows.reshape(len(names), 100, 49))


def proposals(rgb, cells, library):
    """Score a texture hypothesis by agreement across several nearby probes."""
    names, colors, motifs = library
    corners = np.array([c['quad'] for c in cells])
    center = corners.mean(axis=1)
    right = (corners[:, 1] - corners[:, 0] + corners[:, 2] - corners[:, 3]) / 2
    down = (corners[:, 3] - corners[:, 0] + corners[:, 2] - corners[:, 1]) / 2
    origins = center[:, None] + np.array([0,-.23,.23,-.21,.21])[None,:,None]*right[:,None] + np.array([0,-.19,.19,.2,-.2])[None,:,None]*down[:,None]
    right /= np.maximum(.01, np.linalg.norm(right, axis=1, keepdims=True))
    down /= np.maximum(.01, np.linalg.norm(down, axis=1, keepdims=True))
    lattice = OFFSETS[:,0,None] * right[:,None] + OFFSETS[:,1,None] * down[:,None]
    # Shortlist by sampled surface color with a bounded shared RGB gain.
    samples = bilinear(rgb, origins).mean(axis=1)
    gain = np.clip((samples @ LUMA)[:,None] / np.maximum(1, colors @ LUMA), .42, 1.15)
    dist = np.linalg.norm(samples[:,None] - gain[:,:,None]*colors, axis=2) + 6*abs(np.log(gain))
    top = np.argsort(dist, axis=1)[:,:64]
    result = []
    for i in range(len(cells)):
        coords = origins[i,:,None,None,:] + PITCH[None,:,None,None]*lattice[i,None,None]
        patches = (bilinear(rgb, coords) @ LUMA).astype(np.float32)
        # Discount straight outlines. Distinctive textures vary across both axes.
        native = patches.reshape(5,len(PITCH),7,7)
        dx = np.diff(native,axis=3)[:,:,:-1,:]
        dy = np.diff(native,axis=2)[:,:,:,:-1]
        xx, yy, xy = ((a*b).mean(axis=(2,3)) for a,b in ((dx,dx),(dy,dy),(dx,dy)))
        weak = (xx+yy-np.sqrt((xx-yy)**2+4*xy**2))/2
        weight = np.clip(weak/9,0,1)
        features = normalize(patches)
        scores = (features.reshape(-1,49) @ motifs[top[i]].reshape(-1,49).T).reshape(
            len(origins[i]), len(PITCH), len(top[i]), 100)
        # A repeat seen at several locations and the same scale is stronger
        # evidence than the best alignment of any single small patch.
        by_scale = scores.max(axis=3)*weight[:,:,None]
        strength = np.sort(by_scale,axis=0)[-3:].mean(axis=0).max(axis=0)
        # A bright, low-contrast sprite can have a coincidentally similar
        # patch. Require its overall color to be plausible in the same light.
        strength -= .003*dist[i,top[i]]
        order = np.argsort(-strength)
        best, second = order[:2]
        result.append((str(names[top[i,best]]),float(strength[best]),float(strength[best]-strength[second])))
    return result


def priors(rgb, cells, library):
    """Propagate replicated, distinctive motifs to nearby similar-looking cells."""
    if library is None or len(library[0]) < 2:
        return {}
    matches = proposals(rgb, cells, library)
    centers = np.asarray([np.mean(c['quad'], axis=0) for c in cells])
    quads = np.asarray([c['quad'] for c in cells])
    support = {}
    extent = np.linalg.norm(quads[:,2]-quads[:,0],axis=1)
    # Use several small samples so a single bright texel cannot drive propagation.
    uv = np.array([(x,y) for x in (-.3,0,.3) for y in (-.3,0,.3)])
    probes = (centers[:,None]+uv[:,0,None]*(quads[:,1]-quads[:,0])[:,None]
              + uv[:,1,None]*(quads[:,3]-quads[:,0])[:,None])
    colors = np.median(bilinear(rgb,probes),axis=1)
    luminosity = colors @ LUMA
    chroma = colors-luminosity[:,None]
    labels = np.array([n for n,_,_ in matches])
    certainty = np.array([s if m>.025 else 0 for _,s,m in matches])
    anchors = np.flatnonzero(certainty>.87)
    if not len(anchors):
        return support
    for i in range(len(cells)):
        distance = np.linalg.norm(centers[anchors]-centers[i],axis=1)
        light = (luminosity[anchors]-luminosity[i])/16
        tint = np.linalg.norm(chroma[anchors]-chroma[i],axis=1)/6
        weight = (certainty[anchors]-.87)*np.exp(-.5*(light**2+tint**2+
                  (distance/np.maximum(16,3*np.minimum(extent[i],extent[anchors])))**2))
        evidence = {}
        for label,w in zip(labels[anchors],weight):
            evidence[label] = evidence.get(label,0.) + w
        winner = max(evidence,key=evidence.get)
        confidence = evidence[winner]
        if confidence < .085 or confidence < .7*sum(evidence.values()):
            continue
        support[i] = (winner, confidence/(confidence+.085))
    return support
