"""Test one-to-one wall exchanges behind frontmost photographed wall pixels."""
import numpy as np

from engine import nearby_face_decision as nearby


def matching(edges, weights):
    """Maximize disjoint source/destination exchanges, breaking ties by photographic evidence."""
    adjacent = {}
    for i in np.argsort(-weights, kind='stable'):
        s, t, axis = map(int, edges[i])
        adjacent.setdefault(s, {}).setdefault(t, axis)
    dest = {}

    def augment(source, seen):
        for target, axis in adjacent[source].items():
            if target in seen:
                continue
            seen.add(target)
            if target not in dest or augment(dest[target][0], seen):
                dest[target] = source, axis
                return True
        return False

    for source in sorted(adjacent, key=lambda s: len(adjacent[s])):
        augment(source, set())
    return [(s,t,axis) for t,(s,axis) in dest.items()]


def finalize(image, parts, camera, prior, data, pairs):
    decisions = ({(s,t) for s,t,_ in pairs}, {t for _,t,_ in pairs}, {(t,a) for _,t,a in pairs})
    return nearby.apply(image, parts, camera, prior, data, decisions)


def choose(image,parts,camera,prior,data,atlas_path=None):
    if data is None:
        return prior, {}
    try:
        model = nearby._weights(nearby.MODEL.read_bytes())
    except OSError:
        model = None
    if not model:
        return prior, {}
    mask = data['edges'][:,2] != 1
    if not mask.any():
        return prior, {}
    data = dict(data, edges=data['edges'][mask], features=data['features'][mask])
    if model['atlas']:
        from engine.nearby_face_art import enrich
        data = enrich(image,parts,camera,data,atlas_path=atlas_path)
    if data is None:
        return prior, {}
    source = nearby.predict_forest(model['source'],data['features'])
    dest = nearby.predict_forest(model['destination'],data['features'])
    take = (source <= min(.10,model['drop'])) & (dest >= max(.95,model['add']))
    pairs = matching(data['edges'][take], (dest-source)[take])
    after = finalize(image,parts,camera,prior,data,pairs)
    return after, dict(eligible=int(take.sum()), pairs=len(pairs), added=len(after-prior), removed=len(prior-after))


def recover(image,parts,counts,camera,prior,*,atlas_path=None):
    if not prior or not parts or camera.get('P') is None:
        return prior, {}
    return choose(image,parts,camera,prior,nearby.observations(image,parts,camera,prior),atlas_path)
