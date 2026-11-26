"""Register nine overlapping plane predictions on the full screenshot sampling grid."""
import numpy as np

from engine import planarity_source


STRIDE = planarity_source.SETTINGS['stride']
TILING = dict(grid=[3, 3], crop_fraction=.5, stride=STRIDE, artificial_edge='discard touching region',
              geometry='Average full-image camera normals and points under registered tile labels',
              order='row-major; for each parent retain one accepted split with the most supporting samples')


def crop_boxes(size):
    """Nine half-frame windows. Align edges to the same 4-pixel cell lattice as the original labels."""
    if min(size) < 4 * STRIDE:
        return []
    w, h = (v // STRIDE for v in size)
    width, height = (w + 1) // 2, (h + 1) // 2
    # The last tile sees the true image edge, including trailing pixels after its last full cell.
    return [(STRIDE*x, STRIDE*y, size[0] if x+width==w else STRIDE*(x+width),
             size[1] if y+height==h else STRIDE*(y+height))
            for y in (0, (h-height)//2, h-height)
            for x in (0, (w-width)//2, w-width)]


def mean_geometry(labels, normal, points):
    ids = labels.ravel()
    valid = np.isfinite(normal).all(axis=2) & np.isfinite(points).all(axis=2) & (points[..., 2] > 0)
    select = valid.ravel()
    count = np.maximum(1, np.bincount(ids[select], minlength=int(ids.max()) + 1))
    def means(field):
        values = field.reshape(-1, 3)[select]
        return np.stack([np.bincount(ids[select], weights=c, minlength=len(count)) for c in values.T], axis=1) / count[:, None]
    normals = means(normal)
    normals /= np.maximum(1e-8, np.linalg.norm(normals, axis=1, keepdims=True))
    return normals.astype('float32'), means(points).astype('float32')


def compute_crops(image, model, segmentation, device, normal, points, labels_key):
    timing = dict(crop_inference_s=0., crop_segmentation_s=0.)
    boxes = crop_boxes(image.size)
    tiles = []
    fields = dict(image_size=image.size, boxes=boxes)
    for i, box in enumerate(boxes):
        data, elapsed = planarity_source.infer_image(image.crop(box), model, segmentation, device, [labels_key])
        tile = data[labels_key]
        tiles.append(tile)
        x0, y0, x1, y1 = [v // STRIDE for v in box]
        fields[f'normals_{i}'], fields[f'centers_{i}'] = mean_geometry(tile, normal[y0:y1, x0:x1], points[y0:y1, x0:x1])
        timing['crop_inference_s'] += elapsed['inference_s']
        timing['crop_segmentation_s'] += elapsed['segmentation_s']
    fields['tile_labels'] = np.stack(tiles) if tiles else np.empty((0,0,0), dtype=np.uint16)
    return fields, timing


def registered_tiles(data, complete_only=True):
    """Yield (index, full-grid label image, normals, centers); behind a crop edge is unknown."""
    width, height = map(int, data['image_size'])
    boxes = crop_boxes((width, height))
    if not np.array_equal(data['boxes'], boxes) or len(data['tile_labels']) != len(boxes):
        raise ValueError('Cached tile grid differs from the frozen geometry')
    for i, (box, tile) in enumerate(zip(boxes, data['tile_labels'])):
        x0, y0, x1, y1 = [v // STRIDE for v in box]
        if tile.shape != (y1 - y0, x1 - x0):
            raise ValueError('Inconsistent crop transform')
        if complete_only:
            rims = [edge for edge, cut in ((tile[0], y0>0), (tile[-1], box[3]<height),
                                          (tile[:, 0], x0>0), (tile[:, -1], box[2]<width)) if cut]
            excluded = np.unique(np.concatenate(rims))
            tile = np.where(np.isin(tile, excluded), 0, tile)
        canvas = np.zeros((height//STRIDE, width//STRIDE), dtype=tile.dtype)
        canvas[y0:y1, x0:x1] = tile
        yield i, canvas, data[f'normals_{i}'], data[f'centers_{i}']
