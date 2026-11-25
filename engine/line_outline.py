"""Group whole-frame structural lines into a visible foreground outline."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.plane_candidates import main_component


def silhouette(image, lines):
    """Infer which side of coarse lines belongs to the subject at preview scale.

    Pixels vote from their difference to background colors at the same elevation.
    Neighboring labels prefer to agree except where a detected line cuts between
    them. Max-sum messages on the four-neighbor grid smooth weak isolated votes.
    """
    scale = min(1., 400 / max(image.size))
    size = tuple(max(1, round(n * scale)) for n in image.size)
    small = image.convert('RGB').resize(size, Image.Resampling.BOX)
    rgb = np.asarray(small, np.float32)
    if 'A' in image.getbands() and image.getchannel('A').getextrema()[0] < 128:
        return image.getchannel('A').point(lambda v: 255 if v >= 128 else 0)
    band = max(1, size[0] // 12)
    margins = np.concatenate((rgb[:, :band], rgb[:, -band:]), axis=1)
    palette = np.percentile(margins, [5, 25, 50, 75, 95], axis=1)
    difference = np.min([abs(rgb - c[:, None]).mean(2) for c in palette], axis=0)
    unary = np.clip((difference - 11.) / 11., -1., 1.8).astype(np.float32)
    # Separate horizontal and vertical neighbor links. A vertical wall edge
    # interrupts propagation across columns while preserving it along the edge.
    barriers = [Image.new('L', size), Image.new('L', size)]
    pens = [ImageDraw.Draw(barrier) for barrier in barriers]
    ratio = np.asarray(size) / image.size
    for line in lines:
        if line['scale'] != 'coarse':
            continue
        p, q = np.asarray([line['p'], line['q']]) * ratio
        delta = q - p
        length = np.linalg.norm(delta)
        if length < 1.:
            continue
        normal = abs(delta[::-1]) / length
        for axis, pen in enumerate(pens):
            shift = np.eye(2)[axis] * .5
            pen.line([tuple(p-shift), tuple(q-shift)], fill=round(255 * normal[axis]), width=2)
    # An open sky gradient can differ substantially from the color at the row
    # margins. Look for groups of nonparallel lines and local high-frequency
    # detail before treating an isolated change of background color as a wall.
    gray = image.convert('L')
    residual = abs(np.asarray(gray, np.int16) -
                   np.asarray(gray.filter(ImageFilter.GaussianBlur(1.8)), np.int16))
    detail = Image.fromarray(residual.astype('uint8')).resize(size, Image.Resampling.BOX)
    detail = np.asarray(detail.filter(ImageFilter.BoxBlur(2)), np.float32)
    support = [np.asarray(b.filter(ImageFilter.GaussianBlur(max(1., .018*max(size)))),
                          dtype=np.float32) for b in barriers]
    crossing = np.sqrt(support[0]*support[1])
    prior = Image.fromarray(np.uint8(difference > 10)*255).filter(ImageFilter.MedianFilter(3))
    prior = prior.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    density = np.asarray(prior.filter(ImageFilter.BoxBlur(3)), np.float32)/255.
    unary += 2.4*(density-.5)
    if lines:
        weak = (1 - np.clip(crossing/9, 0, 1)) * (1 - np.clip(detail/2.5, 0, 1))
        unary -= 2.9 * weak
        unary += .55 * np.clip((crossing-5)/15, 0, 1) * np.clip((detail-1)/3, 0, 1)
    cx, cy = [2.8 - 2.66 * np.asarray(b, dtype=np.float32) / 255 for b in barriers]
    incoming = np.zeros((4,) + unary.shape, dtype=np.float32)
    for _ in range(14):
        total = unary + incoming.sum(0)
        nxt = np.zeros_like(incoming)
        nxt[0, :, 1:] = np.clip((total-incoming[1])[:, :-1], -cx[:, :-1], cx[:, :-1])
        nxt[1, :, :-1] = np.clip((total-incoming[0])[:, 1:], -cx[:, :-1], cx[:, :-1])
        nxt[2, 1:] = np.clip((total-incoming[3])[:-1], -cy[:-1], cy[:-1])
        nxt[3, :-1] = np.clip((total-incoming[2])[1:], -cy[:-1], cy[:-1])
        incoming = .45 * incoming + .55 * nxt
    remaining = unary + incoming.sum(0) > 0
    mask = np.zeros_like(remaining)
    minimum = max(12., .003 * mask.size)
    for _ in range(4):
        component = main_component(remaining)
        count = component.sum()
        if count < minimum:
            break
        mask |= component
        remaining &= ~component
        minimum = max(minimum, .11 * count)
    return Image.fromarray(mask.astype('uint8') * 255).resize(image.size, Image.Resampling.NEAREST)


def group_lines(lines, mask):
    """Tag source segments as outer contour, interior structure, or background."""
    field = np.asarray(mask) >= 128
    height, width = field.shape
    offset = max(3., max(width, height) * .006)
    def sample(points):
        xy = np.rint(points).astype(int)
        return field[xy[:, 1].clip(0, height-1), xy[:, 0].clip(0, width-1)]
    grouped = []
    for line in lines:
        p, q = np.asarray([line['p'], line['q']])
        d = q-p
        normal = np.array([-d[1], d[0]]) / max(1., np.linalg.norm(d))
        pts = p + np.linspace(.06, .94, 27)[:, None] * d
        a, b = (sample(pts + k * normal * offset) for k in (-1, 1))
        if abs(a.mean() - b.mean()) >= .50:
            role = 'outline'
        elif sample(pts).mean() >= .55:
            role = 'interior'
        else:
            role = 'background'
        grouped.append(dict(line, role=role))
    return grouped


def contour(mask):
    return np.asarray(mask) > np.asarray(mask.filter(ImageFilter.MinFilter(3)))


def cube_outline(image, lines, initial):
    from engine.cube_axes import fit_axes
    from engine.cube_contour import refine
    axes = fit_axes(lines, image.size)
    mask, changes = refine(image, initial, lines, axes)
    return mask, dict(axes=axes, contour=changes)
