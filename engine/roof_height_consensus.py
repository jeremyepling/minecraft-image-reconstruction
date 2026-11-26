"""Corroborate the main horizontal surface with its registered texture, rim and visible supports."""
from engine import roof_height_atlas, roof_height_rim


def select(image, parts, counts, camera, provisional, atlas_path=None):
    _, atlas = roof_height_atlas.recover(image, parts, counts, camera, provisional,
                                         atlas_path=atlas_path, mode='supported_wide', place_cells=False)
    decisions = atlas['height_decisions']
    total = sum(d['area'] for d in decisions)
    for d in decisions:
        d['atlas_accepted'] = d['accepted']
        d['accepted'] = bool(d['accepted'] and d['area'] >= .65*total and d['margin'] >= .18)
    candidates = [d for d in decisions if d['accepted']]
    if candidates:
        rim, _, _, _, _ = roof_height_rim.select(image, parts, camera, provisional)
        by_part = {d['part']: d for d in rim}
        for d in candidates:
            edge = by_part.get(d['part'])
            d['rim_plane'] = edge['plane_y'] if edge else None
            d['accepted'] = bool(edge and edge['accepted'] and edge['plane_y']==d['plane_y'])
    primary = next((d for d in candidates if d['accepted']), None)
    return dict(height_decisions=decisions, primary_height=primary)


def one_course_override(primary, original):
    if primary is None:
        return {}
    key = primary['part']
    target = primary['plane_y']
    return {key: target} if key in original and abs(original[key]-target)==1 else {}
