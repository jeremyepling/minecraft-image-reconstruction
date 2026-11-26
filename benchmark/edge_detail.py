"""Historical photo-edge settings for controlled comparisons with the shipped cleanup."""
from benchmark.camera_round import ROOT, digest
from engine import photo_edges


SETTINGS = {name:photo_edges.collar_settings(radius) for name,radius in
            [('native2',2),('native3',3),('native4',4)]}
SETTINGS['native4-count-guard'] = dict(SETTINGS['native4'], fallback='native2 then original if an existing complete count changes')
SETTINGS['native4-stable-counts'] = dict(SETTINGS['native4'], fallback='native2 then original if a complete count changes or appears')


def predict(photo, parts, frame_sha, setting):
    if setting == 'native4-stable-counts':
        return photo_edges.refine_surfaces(photo,parts)[0]
    if setting == 'native4-count-guard':
        return photo_edges.guarded_adjustment(photo,parts,protect_unresolved=False)[0]
    return photo_edges.refine_at_radius(photo,parts,SETTINGS[setting])


def provenance(setting):
    return dict(settings=SETTINGS[setting],implementation_sha256=digest(ROOT/'engine/photo_edges.py'))
