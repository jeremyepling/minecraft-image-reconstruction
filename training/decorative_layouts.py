"""Small mixed-shape fixtures with decorative geometry in front of opaque blocks."""

from itertools import product


# The wall ID, wood family and slab/stair family accompany the three full-cube
# materials. Vary the material pairing across families and across split boundaries.
PALETTES = {
    "warm": (
        ("bricks", "smooth_sandstone", "stripped_spruce_wood", "brick_wall", "spruce", "brick"),
        ("mud_bricks", "cut_sandstone", "packed_mud", "mud_brick_wall", "pale_oak", "mud_brick"),
        ("sandstone", "red_sandstone", "oak_planks", "sandstone_wall", "spruce", "sandstone"),
        ("calcite", "stone_bricks", "dark_oak_planks", "stone_brick_wall", "dark_oak", "stone_brick"),
        ("terracotta", "smooth_red_sandstone", "cut_sandstone", "red_sandstone_wall", "birch", "red_sandstone"),
    ),
    "cool": (
        ("polished_andesite", "light_gray_concrete", "dark_oak_planks", "andesite_wall", "dark_oak", "andesite"),
        ("deepslate_tiles", "polished_deepslate", "light_blue_terracotta", "deepslate_tile_wall", "spruce", "deepslate_tile"),
        ("dark_prismarine", "prismarine_bricks", "waxed_oxidized_copper", "prismarine_wall", "dark_oak", "prismarine"),
        ("blue_terracotta", "smooth_stone", "cobbled_deepslate", "cobbled_deepslate_wall", "spruce", "cobbled_deepslate"),
        ("tuff_bricks", "polished_tuff", "green_terracotta", "tuff_brick_wall", "dark_oak", "tuff_brick"),
    ),
}


def state(block, **properties):
    return block + "[" + ",".join(f"{key}={value}" for key, value in properties.items()) + "]"


def gate(wood, *, opened, in_wall=False):
    return state(f"{wood}_fence_gate", facing="south", in_wall=str(in_wall).lower(),
                 open=str(opened).lower(), powered="false")


def set_decoration(build, point, block, role):
    if point in build.cells:
        raise ValueError(f"{role} overwrites {build.cells[point]} at {point}")
    build.cells[point] = (block, role)


def fence_run(build, x0, x1, y, z, wood, *, gate_x, opened):
    for x in range(x0, x1):
        if x == gate_x:
            block = gate(wood, opened=opened)
            role = "foreground_fence_gate"
        else:
            block = state(f"{wood}_fence", north="false", south="false",
                          east=str(x < x1 - 1).lower(), west=str(x > x0).lower(), waterlogged="false")
            role = "foreground_fence"
        set_decoration(build, (x, y, z), block, role)


def wall_run(build, x0, x1, y, z, wall, wood, *, gate_x, opened):
    for x in range(x0, x1):
        if x == gate_x:
            block = gate(wood, opened=opened, in_wall=True)
            role = "foreground_wall_gate"
        else:
            end = x in (x0, x1 - 1)
            block = state(wall, north="none", south="none",
                          east="none" if x == x1 - 1 else "low",
                          west="none" if x == x0 else "low",
                          up=str(end or x in (gate_x - 1, gate_x + 1)).lower(), waterlogged="false")
            role = "foreground_masonry_wall"
        set_decoration(build, (x, y, z), block, role)


def shutter(build, x, y, z, wood, *, opened=True, half="bottom"):
    block = state(f"{wood}_trapdoor", facing="south", half=half,
                  open=str(opened).lower(), powered="false", waterlogged="false")
    set_decoration(build, (x, y, z), block, "facade_trapdoor")


def trim_slab(build, lo, hi, masonry, *, half="bottom"):
    block = state(f"{masonry}_slab", type=half, waterlogged="false")
    for point in product(*(range(start, stop) for start, stop in zip(lo, hi))):
        set_decoration(build, point, block, "partial_slab_trim")


def stairs(build, x0, x1, y, z, masonry):
    block = state(f"{masonry}_stairs", facing="north", half="bottom",
                  shape="straight", waterlogged="false")
    for x in range(x0, x1):
        set_decoration(build, (x, y, z), block, "partial_stone_stair")


def market_build(b, variant, wall, trim, accent):
    width = 21 + 3 * variant
    b.cuboid((0, 0, -4), (width, 7, 1), "stall_backdrop", wall)
    for x in range(2, width - 2, 5):
        for p in product(range(x, x + 3), range(2, 5), range(-1, 1)):
            b.cells.pop(p, None)
        b.cuboid((x, 1, 0), (x + 3, 2, 2), "stall_counter", trim)
        b.cuboid((x - 1, 5, 1), (x + 3, 6, 3 + variant), "projecting_awnings", accent)
    b.cuboid((0, 7, -4), (width, 8, 2), "common_cornice", trim)
    b.cuboid((width - 7, 8, -5), (width - 1, 11 + variant, -1), "upper_warehouse", accent)
    return width


def stable_build(b, variant, wall, trim, accent):
    width = 19 + 4 * variant
    b.cuboid((0, 0, -5), (width, 2, 1), "stable_knee_wall", trim)
    b.cuboid((0, 2, -5), (width, 6, -1), "recessed_stall_backdrop", wall)
    for x in range(0, width - 1, 4):
        b.cuboid((x, 2, -1), (x + 1, 6, 1), "stable_post", accent)
    for layer in range(3):
        b.cuboid((-1, 6 + layer, -6), (width + 1, 7 + layer, 2 - layer), "rising_lean_to", trim)
    left = 1 if not variant else width - 8
    b.cuboid((left, 8, -5), (left + 6, 12 + variant, -2), "feed_loft", wall)
    b.cuboid((left - 1, 12 + variant, -6), (left + 7, 13 + variant, -1), "loft_roof", accent)
    return width


def mill_build(b, variant, wall, trim, accent):
    width = 20 + 4 * variant
    b.cuboid((0, 0, -6), (width, 9, 1), "mill_backdrop", wall)
    for x in range(1, width - 1, 5):
        b.cuboid((x, 2, 0), (x + 1, 9, 1), "heavy_timber", accent)
    for x in range(-1, width + 1):
        b.cuboid((x, 9, -7), (x + 1, 10 + (x + variant) % 5, 2), "sawtooth_roof", trim)
    wheel = width - 6 if not variant else 5
    b.cuboid((wheel - 1, 2, 1), (wheel + 2, 3, 3), "wheel_rim", accent)
    b.cuboid((wheel - 1, 8, 1), (wheel + 2, 9, 3), "wheel_rim", accent)
    for x in (wheel - 2, wheel + 2):
        b.cuboid((x, 3, 1), (x + 1, 8, 3), "wheel_rim", accent)
    b.cuboid((wheel, 3, 2), (wheel + 1, 8, 3), "wheel_spoke", trim)
    b.cuboid((wheel - 1, 5, 2), (wheel + 2, 6, 3), "wheel_spoke", trim)
    return width


def chapel_build(b, variant, wall, trim, accent):
    width = 19 + 3 * variant
    b.cuboid((0, 0, -5), (width, 7, 1), "chapel_backdrop", wall)
    center = width // 2
    for i in range(5):
        b.cuboid((center - 5 + i, 7 + i, -7), (center + 6 - i, 8 + i, 2), "corbelled_gable", trim)
    over = width - 7 if not variant else 1
    b.cuboid((over, 7, -7), (over + 5, 15 + variant, -2), "bell_tower", wall)
    for p in product(range(over + 2, over + 4), range(11, 14), range(-4, -2)):
        b.cells.pop(p, None)
    b.cuboid((over - 1, 15 + variant, -8), (over + 6, 16 + variant, -1), "tower_cap", accent)
    b.cuboid((center - 2, 0, 0), (center + 3, 5, 1), "contrasting_entry", accent)
    return width


def garden_build(b, variant, wall, trim, accent):
    width = 22 + 3 * variant
    b.cuboid((0, 0, -5), (width, 6, -2), "garden_rear_enclosure", wall)
    b.cuboid((0, 0, -2), (width, 3, 1), "foreground_planting_bed", trim)
    for x in range(1, width, 5):
        b.cuboid((x, 3, -2), (x + 1, 8, 0), "open_pergola_piers", accent)
    b.cuboid((0, 8, -3), (width, 9, 1), "pergola_beam", trim)
    for x in range(0, width, 2):
        b.cuboid((x, 9, -6), (x + 1, 10, 2), "pergola_rafters", accent)
    tower = 2 if not variant else width - 10
    b.cuboid((tower, 6, -8), (tower + 8, 11 + variant, -5), "garden_pavilion", wall)
    b.cuboid((tower - 1, 11 + variant, -9), (tower + 9, 12 + variant, -4), "pavilion_roof", trim)
    return width


FAMILIES = (
    ("train", "market-court", market_build),
    ("train", "stable-yard", stable_build),
    ("train", "riverside-mill", mill_build),
    ("validation", "hillside-chapel", chapel_build),
    ("test", "walled-garden", garden_build),
)


def layout(builder, family, variant):
    def build(b, wall, trim, accent, masonry_wall, wood, masonry):
        width = builder(b, variant, wall, trim, accent)
        middle = width // 2
        # Distinct railing types cross the same contrasting facade from both obliques.
        # The foreground pieces leave gaps onto the visible full cubes behind them.
        b.cuboid((1, 0, 1), (width - 1, 1, 4), "full_cube_terrace", trim)
        fence_run(b, 2, middle, 1, 3, wood, gate_x=middle - 3, opened=bool(variant))
        wall_run(b, middle + 1, width - 2, 1, 2, masonry_wall, wood,
                 gate_x=middle + 3, opened=not variant)
        stairs(b, middle - 1, middle + 2, 0, 4, masonry)
        trim_slab(b, (1, 0, 4), (middle - 1, 1, 5), masonry)
        trim_slab(b, (middle + 2, 0, 4), (width - 1, 1, 5), masonry, half="top")
        shutter(b, 1, 2, 1, wood, opened=False, half="top")
        shutter(b, 2, 3, 1, wood)
        shutter(b, width - 2, 3, 1, wood)
        shutter(b, width - 2, 4, 1, wood, half="top")

    build.__name__ = f"{family}-{variant + 1}"
    return build


GROUPS = [(split, layout(builder, family, i)) for split, family, builder in FAMILIES for i in range(2)]


def palettes_for(shape):
    family, variant = shape.__name__.rsplit("-", 1)
    family_index = next(i for i, (_, name, _) in enumerate(FAMILIES) if name == family)
    index = (family_index + int(variant) - 1) % len(FAMILIES)
    return {name: entries[index] for name, entries in PALETTES.items()}
