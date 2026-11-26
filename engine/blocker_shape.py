"""Match visible Minecraft face markings using aligned directional contrast and color."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine.atlas import digest
from engine.geometry import face
from engine.surface_model import canonical


def norm_rows(values):
    flat = values.reshape(len(values), -1)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-4)


def features(tiles):
    gray = tiles @ np.array([.213, .715, .072], np.float32)
    coarse = gray.reshape(-1, 8, 2, 8, 2).mean((2, 4))
    # Signed derivatives preserve which side of a crack or flake is lighter.
    dx = np.diff(coarse, axis=2)
    dy = np.diff(coarse, axis=1)
    edges = norm_rows(np.concatenate([dx.reshape(len(dx), -1), dy.reshape(len(dy), -1)], axis=1))
    spatial = norm_rows(coarse - coarse.mean((1, 2), keepdims=True))
    # Average parallel responses along a line, preserving the location of bands and rims.
    lines = norm_rows(np.concatenate([dx.mean(1), dy.mean(2)], axis=1))
    return spatial, edges, lines


class ShapeMatcher:
    def __init__(self, atlas_path, *, catalog=None):
        atlas_path = Path(atlas_path)
        meta = json.loads((atlas_path/'atlas.json').read_bytes())
        index = catalog if catalog is not None else json.loads((atlas_path/'placement-index.json').read_bytes())
        if (meta['version'] != index['version'] or meta['atlas_sha256'] != index['atlas_sha256']
                or digest(atlas_path/meta['atlas']) != meta['atlas_sha256']):
            raise ValueError('Atlas and placement index must match')
        self.version = meta['version']
        self.mapping = defaultdict(set)
        for side in index['choices'].values():
            for name, ids in side.items():
                self.mapping[name].update(ids)
        tiles, names = [], []
        self.glass = []
        with Image.open(atlas_path/meta['atlas']) as sheet:
            for texture in meta['textures']:
                name = texture['id'].split('/')[-1]
                if name not in self.mapping:
                    continue
                x,y,w,h = texture['rect']
                rgba = np.asarray(sheet.crop((x,y,x+w,y+h)).convert('RGBA').resize((16,16),Image.Resampling.BOX))
                if rgba[:,:,3].min() == 255:
                    tiles.append(rgba[:,:,:3])
                    names.append(name)
                elif name == 'glass' or name.endswith('_stained_glass'):
                    for tile in self.orientations(rgba[None]):
                        alpha = tile[:,:,3]
                        mask = alpha >= int(alpha.min()) + 32
                        if mask.any():
                            self.glass.append((name, tile[:,:,:3].astype(np.float32), alpha.astype(np.float32)/255, mask))
        tiles = np.asarray(tiles, np.float32)
        self.names = np.array(names)
        self.mean = tiles.mean((1,2))
        self.bank = features(self.orientations(tiles))

    @staticmethod
    def orientations(tiles):
        return np.concatenate([np.rot90(t, k, axes=(1,2)) for t in (tiles,tiles[:,:,::-1]) for k in range(4)])

    def rank(self, rgb, quad):
        return self.rank_patch(face(rgb,canonical(quad),32))

    def rank_patch(self, patch):
        small = np.asarray(Image.fromarray(np.uint8(np.clip(patch,0,255))).resize((16,16),Image.Resampling.BOX),np.float32)
        # Compete under a single small translation, shared by every descriptor.
        offsets = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
        padded = np.pad(small, ((1,1),(1,1),(0,0)), mode='edge')
        probes = np.stack([padded[1+y:17+y,1+x:17+x] for x,y in offsets])
        query = features(probes)
        similarities = sum(weight*(ref@q.T) for weight,ref,q in zip((.55,.32,.13), self.bank, query))
        similarities[:,1:] -= .035
        structure = similarities.reshape(8,len(self.names),-1).max((0,2))
        mean = small.mean((0,1))
        gain = np.clip((self.mean@mean)/(np.square(self.mean).sum(1)+1e-4),.28,1.5)
        color = np.linalg.norm(self.mean*gain[:,None]-mean,axis=1)/65 + .22*np.abs(np.log(gain/.8))
        score = structure - .34*color
        names = self.names
        # Transparent regions have no reference color. Compare contrast only at
        # visible marks, relative to the immediately surrounding rendered pixels.
        glass_scores = {}
        gray = small.mean(2)
        neighbors = np.pad(gray, 1, mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(neighbors, (3, 3))
        mark_contrast = gray - np.median(windows, axis=(2, 3))
        # Four-neighborhood differences reveal whether both diagonal flecks and
        # the rim align. Compare each mark with its adjacent unmarked pixels so
        # arbitrary background texture cannot earn a template match.
        local = np.pad(small, ((1,1),(1,1),(0,0)), mode='edge')
        backgrounds = {}
        for name, tile, alpha, mask in self.glass:
            interior = mask.copy()
            interior[[0,-1],:] = False
            interior[:,[0,-1]] = False
            rim = mask & ~interior
            if interior.sum()<3 or not rim.any():
                continue
            mask_key = mask.tobytes()
            if mask_key not in backgrounds:
                flags = np.pad(mask,1,constant_values=True)
                adjacent = [(i,j) for i in range(3) for j in range(3) if i!=1 or j!=1]
                bg_sum = np.zeros_like(small)
                n = np.zeros_like(gray)
                for i,j in adjacent:
                    valid = ~flags[i:i+16,j:j+16]
                    bg_sum += local[i:i+16,j:j+16]*valid[:,:,None]
                    n += valid
                bg = bg_sum/np.maximum(n,1)[:,:,None]
                backgrounds[mask_key] = bg
            bg = backgrounds[mask_key]
            # Clear and tinted glass both have two levels of opacity. Invert
            # the base tint from the measured neighbor before predicting marks.
            a0 = float(alpha.min())
            blend = (alpha[mask]-a0)/(1-a0)
            src = tile[mask]*alpha[mask,None] - np.median(tile[~mask],axis=0)*a0*(1-blend[:,None])
            obs = small[mask]-bg[mask]*(1-blend[:,None])
            gain = np.clip(np.sum(src*obs)/max(1,np.sum(src*src)),.28,1.5)
            err = np.linalg.norm(src*gain-obs,axis=1)
            magnitude = np.abs(small-bg).mean(2)
            interior_strength = np.quantile(magnitude[interior],.25)
            rim_strength = np.median(magnitude[rim])
            if min(interior_strength,rim_strength)<5:
                continue
            # Strong evidence needs several marks at their predicted locations.
            noise = max(3,float(np.median(abs(mark_contrast[~mask]))))
            distinction = min(interior_strength,rim_strength)/noise
            if distinction<2:
                continue
            signal = min(1., distinction/6)
            residual = np.median(err[interior[mask]])*.5 + np.median(err[rim[mask]])*.5
            glass_score = .65 + .42*signal - residual/80
            if glass_score>glass_scores.get(name,(-np.inf,0,0))[0]:
                glass_scores[name] = (glass_score,float(signal),float(residual))
        if glass_scores:
            names=np.concatenate([names,np.asarray(list(glass_scores))])
            score=np.concatenate([score,np.array([v[0] for v in glass_scores.values()])])
        order = np.argsort(-score,kind='stable')[:5]
        textures = names[order].tolist()
        ids = [min(self.mapping[t], key=lambda b:(b!=t,b)) for t in textures]
        winner=order[0]
        if winner<len(self.names):
            shape_agreement=float(structure[winner])
            color_penalty=float(.34*color[winner])
        else:
            _,shape_agreement,error=glass_scores[textures[0]]
            color_penalty=float(error/80)
        margin=float(score[order[0]]-score[order[1]])
        return dict(shape_agreement=shape_agreement,color_penalty=color_penalty,score_margin=margin,
                    ambiguous=bool(margin<.04 or len(self.mapping[textures[0]])>1),texture=textures[0],block_id=ids[0],ranked_ids=ids,ranked_textures=textures,scores=score[order].tolist())
