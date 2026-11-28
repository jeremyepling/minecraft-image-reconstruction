"""Deterministic architecture and paired camera placement for Floating Front 100."""
import math
from itertools import product

import numpy as np

FAMILIES = ("arcade", "sawtooth", "skybridge", "terrace", "portico", "bell-tower",
            "viaduct", "observatory", "gatehouse", "gabled-hall", "crane", "water-tower", "windmill", "vault")
PALETTES = (
    ("bricks", "spruce_planks", "polished_andesite"),
    ("tuff_bricks", "dark_oak_planks", "calcite"),
    ("cut_sandstone", "jungle_planks", "mud_bricks"),
    ("stone_bricks", "waxed_cut_copper", "polished_tuff"),
    ("dark_prismarine", "prismarine_bricks", "birch_planks"),
    ("mossy_stone_bricks", "packed_mud", "oak_planks"),
    ("quartz_bricks", "deepslate_tiles", "polished_diorite"),
    ("mud_bricks", "mangrove_planks", "chiseled_sandstone"),
)
SIZE_NAMES = ("small", "medium", "large")


def design(number):
    """Vary bay count, roof profile, openings, depth offsets, spans and masonry per site."""
    rng = np.random.default_rng(812504 + number)
    family_idx = (number - 1) % len(FAMILIES)
    family = FAMILIES[family_idx]
    size = (number - 1) % 3
    h = (int(rng.integers(6, 11)), int(rng.integers(15, 26)), 50)[size]
    w = int(rng.uniform(1.25, 2.25) * h)
    d = max(5, int(rng.uniform(.28, .52) * h))
    bays = int(rng.integers(2, 5 + size))
    phase = rng.uniform(.02, .25)
    palette = (number // 10 + family_idx * 3) % len(PALETTES)
    wall, roof, trim = PALETTES[palette]
    cells = {}
    for x in range(w):
        t = (x + .5) / w
        bay = (t * bays) % 1
        edge = min(t, 1 - t)
        ridge = 1 - abs(2 * bay - 1)
        skirt = 0
        back, front = -d, 0
        if family == "arcade":
            top = h
            skirt = int(h * (.43 + .36 * math.sin(math.pi * bay))) if .13 < bay < .87 else 0
        elif family == "sawtooth":
            top = int(h * (.57 + .43 * bay))
            front = int(d * .23 * (int(t * bays) % 2))
        elif family == "skybridge":
            pier = edge < .18 + phase / 3
            top = int(h * ((.82 if t < .5 else 1) if pier else .69))
            skirt = 0 if pier else int(h * .47)
            front = 0 if pier else -max(2, d // 3)
        elif family == "terrace":
            step = min(bays - 1, int(t * bays))
            top = int(h * (.34 + .66 * step / (bays - 1)))
            front = -int(d * .4 * step / bays)
        elif family == "portico":
            top = int(h * (.78 + .22 * (1 - abs(2 * t - 1))))
            front = max(2, d // 4) if bay < .2 or x in (0, w - 1) else 0
        elif family == "bell-tower":
            central = abs(t - .5) < .16 + phase / 3
            top = int(h * ((.88 + .12 * (1 - abs(t - .5) * 5)) if central else .4 + .13 * ridge))
            back += 0 if central else d // 4
            front += max(1, d // 5) if central else 0
        elif family == "viaduct":
            top = int(h * (.89 + .11 * (bay < .2)))
            skirt = int(h * .78 * math.sqrt(max(0., 1 - ((bay - .5) / .4) ** 2)))
        elif family == "observatory":
            dome = math.sqrt(max(0., 1 - ((t - .5) / (.32 + phase / 4)) ** 2))
            top = int(h * (.34 + .66 * dome))
            front = int(d * .27 * dome)
        elif family == "gatehouse":
            tower = edge < .2 + phase / 4
            top = int(h * ((.91 + .09 * (x % max(3, h // 6) < 2)) if tower else .67))
            skirt = int(h * .48 * math.sqrt(max(0., 1 - ((t - .5) / .16) ** 2)))
            front = d // 4 if tower else 0
        elif family == "crane":
            mast = .24 < t < .39
            top = h if mast else int(h * (.74 + .12 * (1 - t)))
            skirt = 0 if mast else max(0, top - max(1, h // 10))
            if t < .15:  # rear counterweight
                skirt -= max(1, h // 6)
            if .84 < t < .84 + 1.5 / w:  # cable and hanging load
                skirt = int(h * .29)
            back = -max(2, d // 3) if not mast else -d
        elif family == "water-tower":
            # A broad octagonal tank over splayed piers, with a shallow pitched cap.
            top = int(h * (.82 + .18 * (1 - abs(2 * t - 1))))
            skirt = int(h * (.47 + .09 * abs(2 * t - 1)))
            if .16 < t < .26 or .74 < t < .84:
                skirt = 0
            back += max(0, int((.24 - edge) * d * 1.5))
            front -= max(0, int((.24 - edge) * d * 1.5))
        elif family == "windmill":
            # Crossed stepped sails are added in front of a narrow central mill.
            if abs(t - .5) > .22:
                continue
            top = int(h * (.71 + .19 * (1 - abs(t - .5) / .22)))
            front = -max(0, int((.15 - edge) * d))
        elif family == "vault":
            top = max(3, int(h * math.sqrt(max(.03, 1 - (2 * t - 1) ** 2))))
            skirt = 0 if edge < .13 else max(1, top - max(2, h // 6))
            front = 2 if .08 < edge < .17 else 0
        else:
            top = int(h * (.53 + .47 * ridge))
            front = -int(d * .32 * (int(t * bays) % 2))
        top = max(skirt + 2, top)
        for y in range(skirt, top):
            cap = y == top - 1
            # Leave true openings through the walls with occasional recessed accent panels.
            window = (.23 < bay < .7 and y > skirt + 1 and y < top - 2
                      and (y + number) % max(4, h // 5) in (1, 2))
            zs = range(back, front + 1) if cap or y == skirt or x in (0, w - 1) else (back, front)
            for z in zs:
                if window and z in (back, front) and family not in ("arcade", "viaduct"):
                    if (int(t * bays) + number) % 2 == 0 and z == front:
                        cells[x, y, z - 1] = trim
                    continue
                cells[x, y, z] = roof if cap else trim if y == skirt or bay < .12 else wall
    if family == "windmill":
        hub_x, hub_y = w // 2, int(h * .54)
        for x in range(w):
            t = (x - hub_x) / max(hub_x, w - hub_x - 1)
            for slope in (-1, 1):
                y = min(h - 1, max(0, round(hub_y + slope * t * (h - hub_y - 1))))
                for yy in range(max(0, y - max(1, h // 20)), y + 1):
                    cells[x, yy, 2] = roof if abs(t) > .4 else trim
        cells[hub_x, hub_y, 1] = trim
    # Extend the highest ridge by its rounding remainder to attain the requested block height.
    ymax = max(y for _, y, _ in cells)
    for (x, y, z), material in list(cells.items()):
        if y == ymax:
            for yy in range(y + 1, h):
                cells[x, yy, z] = material
    ymin = min(y for _, y, _ in cells)
    if ymin > 0:
        # Narrow arch spans can discretize away their piers. Anchor both end columns.
        for x in (0, w - 1):
            bottom = min(y for xx, y, _ in cells if xx == x)
            for z in {z for xx, y, z in cells if xx == x and y == bottom}:
                for y in range(bottom):
                    cells[x, y, z] = trim
    decorations = {}
    def mount(frac, level, state):
        x = min(w - 1, int((w - 1) * frac))
        column = [(y, z) for xx, y, z in cells if xx == x]
        target = min(y for y, _ in column) + level * (max(y for y, _ in column) - min(y for y, _ in column))
        available = sorted({y for y, _ in column}, key=lambda y: (abs(y - target), y))
        for y in available:
            z = max(z for yy, z in column if yy == y) + 1
            if (x, y, z) not in decorations:
                decorations[x, y, z] = state
                return

    # Shutters lie flush against the south wall. Closed hatches project horizontally
    # from it, and slab/stair ledges have their own modeled silhouette and occlusion.
    wood = ("spruce", "dark_oak")[number % 2]
    for idx, frac in enumerate((.17, .49, .81)):
        half = ("bottom", "top")[(idx + number) % 2]
        opened = "false" if idx == 1 else "true"
        mount(frac, .43 + .11 * idx,
              f"{wood}_trapdoor[facing=south,half={half},open={opened},powered=false,waterlogged=false]")
    for idx in range(max(2, w // 12)):
        frac = .23 + idx / max(2, w // 12) * .48
        slab = ("bottom", "top")[(idx + number) % 2]
        mount(frac, .26, f"brick_slab[type={slab},waterlogged=false]")
        half = ("top", "bottom")[(idx + number) % 2]
        mount(frac, .72, f"stone_brick_stairs[facing=north,half={half},shape=straight,waterlogged=false]")

    # A side-mounted shutter closes against an east-facing exterior. Low roof
    # railings stand on actual cube tops inside the clean structure height.
    side = sorted(((x, y, z) for x, y, z in cells if x == w - 1),
                  key=lambda p: (abs(p[1] - h * .37), -p[2]))
    x, y, z = side[0]
    decorations[x + 1, y, z] = f"{wood}_trapdoor[facing=east,half=top,open=true,powered=false,waterlogged=false]"
    top_candidates = sorted((p for p in cells if p[1] < h - 1
                             and (p[0], p[1] + 1, p[2]) not in cells),
                            key=lambda p: (-p[2], -p[1], p[0]))
    placed = 0
    for x, y, z in top_candidates:
        if (x, y + 1, z) not in decorations and x % max(2, w // 7) == 0:
            decorations[x, y + 1, z] = "dark_oak_fence[east=false,north=false,south=false,waterlogged=false,west=false]"
            placed += 1
            if placed == 3:
                break
    length = max(3, w // 3)
    start = int(w * (.12 + phase))
    fence_z = max(z for _, _, z in cells) + 2
    for i in range(length):
        east, west = str(i < length - 1).lower(), str(i > 0).lower()
        decorations[start + i, max(1, h // 6), fence_z] = (
            f"oak_fence[east={east},north=false,south=false,waterlogged=false,west={west}]")
    layout = f"front-{number:03d}-{family}"
    return dict(layout=layout, family=family, size=SIZE_NAMES[size], palette=f"stonework-{palette + 1}",
                split=("train" if family_idx < 10 else "validation" if family_idx < 12 else "test"),
                dimensions=[w, h, max(z for _, _, z in cells) - min(z for _, _, z in cells) + 1],
                angle_deg=int(rng.integers(-27, 28)), elevation_deg=int(rng.integers(-7, 10))), cells, decorations


def camera(cells, azimuth, elevation):
    # Every silhouette extremum comes from a corner of a cell. Only the extrema of each
    # vertical column can attain the extrema in an arbitrary linear direction.
    columns = {}
    for x, y, z in cells:
        low, high = columns.get((x, z), (y, y))
        columns[x, z] = min(low, y), max(high, y)
    pts = np.array([(x + i, yy, z + j) for (x, z), (low, high) in columns.items()
                    for i, yy, j in product((0, 1), (low, high + 1), (0, 1))])
    az, el = np.radians([azimuth, elevation])
    outward = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    right = np.array([np.cos(az), 0, -np.sin(az)])
    up = np.cross(outward, right)
    eye_depth = pts @ outward
    slopes = np.tan(np.radians(55 / 2)) * np.array([1.6, 1]) * .90
    # Find the nearest eye depth where all points fit between both pairs of frustum planes.
    uv = pts @ np.stack([right, up]).T
    near_bounds = (uv + eye_depth[:, None] * slopes).max(axis=0)
    far_bounds = (uv - eye_depth[:, None] * slopes).min(axis=0)
    distance = np.max((near_bounds - far_bounds) / (2 * slopes))
    uv_center = (near_bounds + far_bounds) / 2
    eye = outward * distance + right * uv_center[0] + up * uv_center[1]
    return dict(x=eye[0], feetY=eye[1] - 1.62, z=eye[2], target=(eye - outward * 20).tolist())
