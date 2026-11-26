"""Require one independently detected neighboring surface to explain a candidate tail."""
import numpy as np

from engine import attached_strip, strip_evidence
from engine.entrance_regions import _components
from engine.surface_perimeter import _native_alpha


METHOD = 'neighbor-supported-strip'
SETTINGS = dict(neighbor_radius_at_900px=3, length_sections=4, minimum_section_fraction=.75,
                minimum_near_fraction=.90, maximum_neighbor_color_error=18.)


def select_receiver(rgb, donor, tail, receivers, radius):
    """Compare the same indexed tail in a small crop of each candidate receiver."""
    y, x = np.nonzero(tail)
    points = np.column_stack((x, y))
    centered = points - points.mean(axis=0)
    axis = np.linalg.eigh(centered.T @ centered)[1][:, -1]
    pos = centered @ axis
    n = SETTINGS['length_sections']
    bins = np.minimum(n-1, np.floor(n*(pos-pos.min())/max(float(np.ptp(pos)), 1.)).astype(int))
    sizes = np.bincount(bins, minlength=n)
    around = attached_strip.neighborhood(tail, radius) > 0
    winner = None
    for pid, receiver in receivers:
        # Only pixels outside the parent supply independent neighborhood evidence.
        independent = receiver & ~donor
        nearby = (attached_strip.neighborhood(independent, radius) > 0)[y, x]
        if nearby.mean() < SETTINGS['minimum_near_fraction']:
            continue
        votes = np.bincount(bins[nearby], minlength=n)
        if np.any(votes < SETTINGS['minimum_section_fraction']*sizes):
            continue
        colors = rgb[around & independent]
        if len(colors) < strip_evidence.SETTINGS['minimum_pixels']:
            continue
        error = strip_evidence.palette_distance(rgb[y, x], colors)
        if error > SETTINGS['maximum_neighbor_color_error']:
            continue
        overlap = float(receiver[y, x].mean())
        evidence = dict(receiver=pid, overlap_fraction=overlap, near_fraction=float(nearby.mean()),
                        neighbor_color_error=float(error))
        rank = (-overlap, -nearby.mean(), error, pid)
        if winner is None or rank < winner[0]:
            winner = (rank, evidence)
    return None if winner is None else winner[1]


def refine(photo, parts, *, decisions=None):
    """Keep unsupported tails. Accepted removals change only their parent mask."""
    radius = max(1, round(SETTINGS['neighbor_radius_at_900px'] * photo.height / 900))
    masks = {}
    for part in parts:
        native = _native_alpha(part['support'], photo.size)
        if native is not None:
            masks[part['id']] = native

    def check(part, bounds, owned, removed):
        accepted = np.zeros_like(removed)
        for coords in _components(removed):
            y, x = coords.T
            left, top = bounds[:2]
            gx, gy = x + left, y + top
            tail_box = (int(gx.min()), int(gy.min()), int(gx.max()+1), int(gy.max()+1))
            l, t, r, b = tail_box
            crop = (l-radius, t-radius, r+radius, b+radius)
            # PIL pads rectangles beyond a mask or frame. Only real receiver pixels vote.
            def cut(native):
                alpha, (x0,y0,_,_) = native
                return np.asarray(alpha.crop((crop[0]-x0, crop[1]-y0, crop[2]-x0, crop[3]-y0)), bool)
            donors = cut(masks[part['id']])
            tail = np.zeros_like(donors)
            tail[gy-crop[1], gx-crop[0]] = True
            neighbors = ((pid,cut(native)) for pid,native in masks.items() if pid!=part['id']
                         and crop[0]<native[1][2] and crop[2]>native[1][0]
                         and crop[1]<native[1][3] and crop[3]>native[1][1])
            choice = select_receiver(np.asarray(photo.crop(crop)), donors, tail, neighbors, radius)
            if choice:
                accepted[y,x] = True
            if decisions is not None:
                decisions.append(dict(donor=part['id'], tail_xyxy=tail_box, pixels=len(coords),
                                      **(choice or dict(receiver=None, overlap_fraction=None, near_fraction=None,
                                                        neighbor_color_error=None))))
        return accepted

    return attached_strip.refine(photo, parts, gate=strip_evidence.gate, context_gate=check)

def clean_surfaces(photo, parts):
    """Run the shipped rule and record each accepted or declined strip for the manifest."""
    decisions = []
    cleaned, changes = refine(photo, parts, decisions=decisions)
    stats = dict(settings=dict(shape=attached_strip.SETTINGS, color=strip_evidence.SETTINGS, neighbor=SETTINGS),
                 changed_surfaces=len(changes), removed_pixels=sum(int(mask.sum()) for _, _, mask in changes),
                 decisions=decisions)
    return cleaned, stats
