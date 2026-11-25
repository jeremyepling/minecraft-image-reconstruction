"""Color-distribution matching when a predicted cell can be a fragment of a block."""
import numpy as np


def descriptors(images):
    rgb = np.asarray(images, dtype=np.float32)[:, 1:-1, 1:-1] / 255
    light = rgb @ np.array([.213, .715, .072], dtype=np.float32)
    # Opponent colors expose multicolored pixels even when the average is gray.
    channels = np.stack([light, rgb[..., 0] - rgb[..., 1], rgb[..., 2] - rgb[..., 1]], axis=-1)
    mean = channels.mean((1, 2))
    quantiles = np.quantile(channels, [.10, .50, .90], axis=(1, 2)).transpose(1, 0, 2)
    weights = np.array([1, 1.5, 1.5])
    return np.concatenate([mean * weights, (quantiles * weights * .5).reshape(len(mean), -1)], axis=1)


def prototypes(sprites, sprite_ids, tiles, labels):
    """Use in-game exemplars, with the opaque atlas as a check on their identity."""
    raw = descriptors(sprites)
    rendered = descriptors(tiles)
    common = [(i, np.median(rendered[labels == name], axis=0))
              for i, name in enumerate(sprite_ids) if name in labels]
    if not common:
        raise ValueError('The opaque atlas and rendered library have no common textures')
    # Estimate the shared daylight transfer for this library. A sample from the
    # wrong color family (e.g. a transparent render) cannot stand in for a sprite.
    ii, observations = zip(*common)
    source = raw[list(ii), 0]
    actual = np.array(observations)[:, 0]
    mid = np.median(source)
    lo, hi = source < mid, source >= mid
    slope = ((np.median(actual[hi]) - np.median(actual[lo])) /
             max(.01, np.median(source[hi]) - np.median(source[lo]))) if lo.any() and hi.any() else 1.
    slope = np.clip(slope, .5, 1.25)
    intercept = np.median(actual - slope * source)
    prediction = slope * raw
    prediction[:, 0] += intercept
    prediction[:, [3, 6, 9]] += .5 * intercept
    for i, sample in common:
        if np.max(np.abs(sample[:3] - prediction[i, :3])) <= .065:
            prediction[i] = sample
    return prediction.astype(np.float32)


def cost_matrix(images, references):
    q = descriptors(images).astype(np.float32)
    return np.maximum(0, (q*q).sum(1)[:, None] + (references*references).sum(1) -
                      2 * q @ references.T)


def palette_cost(cost, shared):
    """Choose additional surface materials when enough visible color error is explained."""
    if not shared or len(cost) < 2:
        return cost
    # Prune only exact duplicate nearest-neighbor proposals. A distinctive
    # accent can pay the opening cost on its own; there is no palette size cap.
    candidates = np.unique(cost.argmin(axis=1))
    first = cost.mean(axis=0).argmin()
    chosen = [first]
    residual = cost[:, first].copy()
    penalty = .012
    while len(candidates):
        gain = np.maximum(0, residual[:, None] - cost[:, candidates]).sum(axis=0)
        winner = gain.argmax()
        if gain[winner] <= penalty:
            break
        selected = candidates[winner]
        chosen.append(selected)
        residual = np.minimum(residual, cost[:, selected])
        candidates = np.delete(candidates, winner)
    adjusted = cost + penalty
    adjusted[:, chosen] = cost[:, chosen]
    return adjusted


def appearance_rows(rgb, cells, face, references, names, mapping, stable, shared, motif_support=None):
    grouped = {}
    for index, cell in enumerate(cells):
        grouped.setdefault(cell['plane'], []).append(index)
    output = [None] * len(cells)
    for indices in grouped.values():
        batches = []
        for start in range(0, len(indices), 96):
            crops = [face(rgb, cells[i]['quad'], 16) for i in indices[start:start+96]]
            batches.append(cost_matrix(crops, references))
        costs = palette_cost(np.concatenate(batches), shared)
        if motif_support:
            for j,index in enumerate(indices):
                if index in motif_support:
                    label,confidence = motif_support[index]
                    costs[j,names!=label] += .10*confidence
        for index, row in zip(indices, costs):
            order = np.argsort(row, kind='stable')
            top = order[:5]
            margin = (row[top[1]] - row[top[0]]) / max(row[top[1]], 1e-6)
            output[index] = (names[top].tolist(), row[top].tolist(), float(margin),
                             stable(names[order], mapping))
    return output
