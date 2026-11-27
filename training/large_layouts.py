"""Full-cube structure and material recipes for Visible Builds v3."""
from itertools import product


# Each palette supplies masonry, facing, trim, roof, infill and accent.
PALETTES = {
    "limestone": ("smooth_sandstone", "cut_sandstone", "calcite", "red_terracotta", "dark_oak_planks", "chiseled_sandstone"),
    "basalt": ("polished_blackstone_bricks", "deepslate_tiles", "polished_andesite", "dark_prismarine", "cyan_terracotta", "gilded_blackstone"),
    "marble": ("smooth_quartz", "quartz_bricks", "chiseled_quartz_block", "blue_terracotta", "spruce_planks", "polished_diorite"),
    "travertine": ("mud_bricks", "orange_terracotta", "cut_red_sandstone", "acacia_planks", "bricks", "smooth_red_sandstone"),
    "riverstone": ("stone_bricks", "cobblestone", "polished_granite", "prismarine_bricks", "stripped_dark_oak_wood", "mossy_stone_bricks"),
    "kiln": ("bricks", "brown_terracotta", "waxed_cut_copper", "polished_blackstone", "iron_block", "yellow_terracotta"),
    "seafoam": ("prismarine_bricks", "white_concrete", "green_terracotta", "waxed_oxidized_cut_copper", "mangrove_planks", "dark_prismarine"),
    "chalk": ("light_gray_concrete", "gray_concrete", "smooth_stone", "waxed_weathered_copper", "brown_concrete", "black_terracotta"),
    "ochre": ("yellow_terracotta", "gold_block", "smooth_red_sandstone", "warped_planks", "polished_granite", "orange_concrete"),
    "ashlar": ("polished_tuff", "tuff_bricks", "chiseled_tuff_bricks", "deepslate_bricks", "gray_terracotta", "polished_andesite"),
    "orchid": ("purple_terracotta", "magenta_terracotta", "purpur_block", "cherry_planks", "pink_terracotta", "smooth_quartz"),
    "verdigris": ("green_concrete", "lime_terracotta", "waxed_exposed_copper", "waxed_weathered_cut_copper", "bamboo_mosaic", "calcite"),
}


def shell(b, x, z, width, depth, bottom, top, role, material):
    for x0, z0, x1, z1 in ((x, z, x + width, z + 1), (x, z + depth - 1, x + width, z + depth),
                           (x, z + 1, x + 1, z + depth - 1), (x + width - 1, z + 1, x + width, z + depth - 1)):
        b.cuboid((x0, bottom, z0), (x1, top, z1), role, material)


def cut(b, lo, hi):
    for pos in product(*(range(start, stop) for start, stop in zip(lo, hi))):
        b.cells.pop(pos, None)


def castle(b, size, masonry, facing, trim, roof, infill, accent):
    width, height, depth = ((40, 29, 19), (70, 40, 28), (100, 50, 36))[size]
    tower = (7, 9, 11)[size]
    wall_top = height // 2 + 1
    shell(b, 1, 1, width - 2, depth - 2, 0, wall_top, "curtain_wall", masonry)
    shell(b, 1, 1, width - 2, depth - 2, 3, 4, "plinth_course", facing)
    shell(b, 1, 1, width - 2, depth - 2, wall_top - 1, wall_top, "wall_walk_rim", trim)
    for x in range(tower, width - tower, 4):
        for z in (1, depth - 2):
            b.cuboid((x, wall_top, z), (x + 2, wall_top + 2, z + 1), "wall_merlon", facing)
    for index, (x, z) in enumerate(product((0, width - tower), (0, depth - tower))):
        top = height - 2 - (index % 3) * (size + 2)
        shell(b, x, z, tower, tower, 0, top, "corner_tower", facing)
        for y in range(6, top - 4, 7):
            shell(b, x, z, tower, tower, y, y + 1, "tower_course", masonry)
            for edge in (z, z + tower - 1):
                cut(b, (x + tower // 2, y + 1, edge), (x + tower // 2 + 1, y + 4, edge + 1))
        b.cuboid((x, top - 1, z), (x + tower, top, z + tower), "tower_platform", roof)
        shell(b, x, z, tower, tower, top, top + 1, "tower_parapet", trim)
        for offset in range(0, tower, 3):
            for px, pz in ((x + offset, z), (x + offset, z + tower - 1), (x, z + offset), (x + tower - 1, z + offset)):
                b.cuboid((px, top + 1, pz), (px + 1, top + 2, pz + 1), "tower_merlon", accent)
    middle = width // 2
    gate_width = 3 + size
    cut(b, (middle - gate_width, 0, depth - 3), (middle + gate_width + 1, 7 + size, depth))
    cut(b, (middle - gate_width + 1, 7 + size, depth - 3), (middle + gate_width, 10 + size, depth))
    for x in (middle - gate_width - 1, middle + gate_width + 1):
        b.cuboid((x, 0, depth - 2), (x + 1, 11 + size, depth), "gate_jamb", accent)
    b.cuboid((middle - gate_width - 1, 10 + size, depth - 2),
             (middle + gate_width + 2, 11 + size, depth), "gate_lintel", trim)
    keep = middle - tower, depth // 2 - 4
    shell(b, *keep, tower * 2, 8, 0, wall_top + 5, "inner_keep", infill)
    b.cuboid((keep[0], wall_top + 5, keep[1]), (keep[0] + 2 * tower, wall_top + 6, keep[1] + 8), "keep_roof", roof)


def pillars(b, size, masonry, facing, trim, roof, infill, accent):
    count, pitch, tallest = ((5, 5, 16), (6, 8, 27), (8, 12, 42))[size]
    for i in range(count):
        x = i * pitch
        z = -((i * 3 + size) % 3) * (size + 1) if i % 2 == 0 else -(2 + size * 2)
        height = tallest - ((i + size) % 4) * (size + 1)
        shaft = 2 + (i + size) % 3
        b.cuboid((x, 0, z), (x + shaft + 2, 2, z + shaft + 2), "pillar_plinth", masonry)
        b.cuboid((x + 1, 2, z + 1), (x + shaft + 1, height - 2, z + shaft + 1), "pillar_shaft", facing)
        for y in range(5, height - 3, 6):
            b.cuboid((x + 1, y, z + 1), (x + shaft + 1, y + 1, z + shaft + 1), "pillar_band", infill)
        b.cuboid((x, height - 2, z), (x + shaft + 2, height - 1, z + shaft + 2), "pillar_capital", trim)
        b.cuboid((x + 1, height - 1, z + 1), (x + shaft + 1, height, z + shaft + 1), "pillar_coping", roof)
        for y in range(3, height - 3):
            b.cells[x + 1, y, z + shaft] = accent, "pillar_flute"


def viaduct(b, size, masonry, facing, trim, roof, infill, accent):
    bays, span, width, height = ((3, 7, 5, 14), (5, 11, 7, 23), (7, 13, 9, 33))[size]
    length = bays * span + 2
    deck = height - 3
    b.cuboid((0, deck, 0), (length, deck + 1, width), "deck", roof)
    for x in range(0, length, 3):
        for z in (0, width - 1):
            b.cuboid((x, deck + 1, z), (x + 1, height, z + 1), "bridge_rail_pier", accent)
    for i in range(bays + 1):
        x = i * span
        y = (i % 3) * size
        b.cuboid((x, 0, 0), (x + 2, y + 2, width), "pier_base", infill)
        b.cuboid((x, y + 2, 1), (x + 2, deck, width - 1), "bridge_pier", masonry)
        if i == bays:
            continue
        for y in range(deck // 2, deck):
            reach = min((span - 2) // 2, (y - deck // 2) * (size + 1) // 2 + 1)
            for start in (x + 2, x + span - reach):
                b.cuboid((start, y, 1), (start + reach, y + 1, width - 1), "stepped_arch", facing)
        for z in (0, width - 1):
            b.cuboid((x, deck - 2, z), (x + span, deck, z + 1), "spandrel_band", trim)


def factory(b, size, masonry, facing, trim, roof, infill, accent):
    width, depth, height = ((22, 12, 10), (43, 22, 17), (63, 31, 27))[size]
    bays = 4 + size * 2
    pitch = width // bays
    shell(b, 0, 0, width, depth, 0, height, "workshop_shell", masonry)
    for x in range(1, width - 4, pitch):
        cut(b, (x + 1, 3, depth - 1), (min(width - 1, x + pitch - 1), height - 2, depth))
        b.cuboid((x + 1, 3, depth - 3), (min(width - 1, x + pitch - 1), height - 2, depth - 2), "window_backing", infill)
        b.cuboid((x, 0, depth - 1), (x + 1, height, depth + 1), "workshop_buttress", trim)
    for x in range(width):
        y = height + (x % pitch) // 2
        b.cuboid((x, y, 0), (x + 1, y + 1, depth), "sawtooth_roof", roof)
        b.cuboid((x, height, 0), (x + 1, y, 1), "gable_end", facing)
        b.cuboid((x, height, depth - 1), (x + 1, y, depth), "gable_end", facing)
        if x % pitch in (0, pitch - 1):
            b.cuboid((x, height, 1), (x + 1, height + pitch // 2, depth - 1), "sawtooth_end", facing)
    for n in range(2 + size):
        x = width - 4 - n * pitch
        y = height + 5 + (n % 3) * 3
        shell(b, x, 3, 3, 3, height - 1, y, "chimney", facing)
        shell(b, x, 3, 3, 3, y, y + 1, "chimney_rim", accent)


def amphitheater(b, size, masonry, facing, trim, roof, infill, accent):
    width, depth, rows = ((24, 16, 5), (46, 26, 9), (72, 37, 13))[size]
    for row in range(rows):
        y = 2 * row
        x = rows - row
        z = rows - row
        for a, c, d, e in ((x, z, width - x, z + 2), (x, z, x + 2, depth),
                           (width - x - 2, z, width - x, depth)):
            b.cuboid((a, y, c), (d, y + 1, e), "seat_platform", masonry if row % 3 else facing)
            b.cuboid((a, y + 1, c), (d, y + 2, c + 1), "seat_back", infill)
    for x in (1, width - 3):
        b.cuboid((x, 0, rows - 1), (x + 2, rows * 2 + 1, depth), "grandstand_side", facing)
        b.cuboid((x, rows * 2 + 1, rows - 1), (x + 2, rows * 2 + 2, depth), "coping", trim)
    b.cuboid((1, 0, 0), (width - 1, 1, rows + 2), "floor_apron", roof)
    for x in range(2, width - 2, 3):
        b.cuboid((x, 0, depth - 2), (x + 1, 2, depth + 1), "arena_bollard", accent)


def shrine(b, size, masonry, facing, trim, roof, infill, accent):
    width, depth, terraces = ((21, 16, 3), (37, 25, 4), (63, 37, 5))[size]
    for i in range(terraces):
        inset = i * 2
        y = i * 4
        shell(b, inset, inset, width - 2 * inset, depth - 2 * inset, y, y + 3, "terrace_riser", masonry)
        b.cuboid((inset, y + 3, inset), (width - inset, y + 4, depth - inset), "terrace_walk", facing)
        shell(b, inset, inset, width - 2 * inset, depth - 2 * inset, y + 3, y + 4, "terrace_border", trim)
    middle = width // 2
    for y in range(terraces * 4):
        z = depth + terraces * 2 - y
        b.cuboid((middle - 2, y, z), (middle + 3, y + 1, z + 1), "front_stair_tread", accent)
        for x in (middle - 3, middle + 3):
            b.cuboid((x, 0, z), (x + 1, y + 1, z + 1), "stair_side", trim)
    y = terraces * 4
    for x, z in product((middle - 4, middle + 3), (terraces * 2, depth - terraces * 2 - 1)):
        b.cuboid((x, y, z), (x + 1, y + 5, z + 1), "sanctuary_pier", infill)
    for i in range(3):
        b.cuboid((middle - 5 + i, y + 5 + i, terraces * 2 - 1 + i),
                 (middle + 5 - i, y + 6 + i, depth - terraces * 2 + 1 - i), "sanctuary_roof", roof)


def observatory(b, size, masonry, facing, trim, roof, infill, accent):
    diameter, base = ((14, 10), (23, 18), (34, 29))[size]
    center = diameter // 2
    shell(b, 0, 0, diameter, diameter, 0, 3, "plaza_plinth", masonry)
    octagon = {(x, z) for x, z in product(range(1, diameter - 1), repeat=2)
               if abs(x + 0.5 - center) + abs(z + 0.5 - center) <= diameter * 0.72}
    perimeter = {(x, z) for x, z in octagon if any((x + dx, z + dz) not in octagon
                 for dx, dz in ((0, -1), (-1, 0), (0, 1), (1, 0)))}
    for x, z in sorted(perimeter):
        for y in range(3, base):
            material = trim if y in (4, base - 2) else facing if (x + z) % 5 == 0 else masonry
            if z >= diameter - 3 and abs(x - center) <= 1 and y < 8:
                material = infill
            b.cells[x, y, z] = material, "observatory_drum"
    for x, z in sorted(octagon):
        y = base + max(0, (center - max(abs(x - center), abs(z - center))) // 2)
        b.cells[x, y, z] = (accent if abs(x - center) <= 1 else roof), "polygonal_dome"
    b.cuboid((center - 1, base + center // 2, center - 1),
             (center + 2, base + center // 2 + 3, center + 2), "roof_octahedron_top", trim)


def harbor(b, size, masonry, facing, trim, roof, infill, accent):
    width, depth, height = ((27, 19, 13), (53, 30, 20), (83, 38, 32))[size]
    b.cuboid((0, 0, 0), (width, 4, 4), "quay_bulkhead", masonry)
    b.cuboid((0, 4, 0), (width, 5, 5), "quay_cap", trim)
    for i, x in enumerate(range(3, width - 7, 7 + size * 3)):
        far = depth - (i % 3) * (2 + size)
        b.cuboid((x, 3, 4), (x + 3 + size, 4, far), "pier_deck", roof)
        for z in range(5, far, 4 + size):
            for post in (x, x + 2 + size):
                b.cuboid((post, 0, z), (post + 1, 5, z + 1), "pier_post", infill)
        shed_height = 7 + i % 3 + size
        shell(b, x, 0, 5, 3, 5, shed_height, "quayside_shed", facing)
        b.cuboid((x, shed_height, 0), (x + 5, shed_height + 1, 3), "shed_cap", accent)
    x = width - 7
    for y in range(4, height - 2):
        inset = min(2, (y - 4) // max(1, (height - 4) // 3))
        material = accent if (y - 4) % 5 == 0 else facing
        shell(b, x + inset, inset, 7 - inset * 2, 7 - inset * 2, y, y + 1, "navigation_tower", material)
    b.cuboid((x + 1, height - 2, 1), (x + 6, height - 1, 6), "lantern_platform", trim)
    b.cuboid((x + 2, height - 1, 2), (x + 5, height, 5), "lantern_cap", roof)


FAMILIES = (
    ("train", pillars, ("marble", "basalt")),
    ("train", viaduct, ("riverstone", "travertine")),
    ("train", factory, ("kiln", "chalk")),
    ("train", amphitheater, ("orchid", "ashlar")),
    ("train", shrine, ("ochre", "limestone")),
    ("train", observatory, ("verdigris", "seafoam")),
    ("validation", harbor, ("seafoam", "kiln")),
    ("test", castle, ("limestone", "ashlar")),
)


def _layout(family, size, palettes):
    def build(b, *materials):
        family(b, size, *materials)

    build.__name__ = f"{family.__name__}-{size + 1:02}"
    build.family = family.__name__
    build.palettes = palettes
    return build


GROUPS = tuple((split, _layout(family, index, palettes))
               for split, family, palettes in FAMILIES for index in range(3))


def family_for(shape):
    return shape.family


def palettes_for(shape):
    return {name: PALETTES[name] for name in shape.palettes}
