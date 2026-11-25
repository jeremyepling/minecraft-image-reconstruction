"""Package two captured Java 26.3 worlds and verify their stored Anvil blocks and settings."""
import argparse
import gzip
import json
import struct
import zlib
from functools import lru_cache
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from calibration.world import GROUND, PACK, nbt_string, tag
from engine.atlas import digest
from engine.nbt import read_nbt


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def portable_level(raw, title):
    """Remove the capture player's association and set a distinct world-select title."""
    raw = gzip.decompress(raw)
    tree = read_nbt(raw)
    data = tree["Data"]
    uid = data.pop("singleplayer_uuid", None)
    if uid is not None:
        field = bytes([11]) + nbt_string("singleplayer_uuid") + tag((11, uid))[1]
        if raw.count(field) != 1:
            raise ValueError("Unrecognized player identity in saved level metadata")
        raw = raw.replace(field, b"", 1)
    key = bytes([8]) + nbt_string("LevelName")
    previous = key + nbt_string(data["LevelName"])
    if raw.count(previous) != 1:
        raise ValueError("Unrecognized world title field")
    raw = raw.replace(previous, key + nbt_string(title), 1)
    data["LevelName"] = title
    if read_nbt(raw) != tree:
        raise ValueError("Unexpected metadata change while making portable world")
    return gzip.compress(raw, mtime=0)


def pack(save, archive, title):
    paths = sorted(p for p in save.rglob("*") if p.is_file() and (
        p.relative_to(save).as_posix() in ("level.dat", "icon.png", "datapacks/"+PACK)
        or (p.relative_to(save).parts[0] in ("data", "dimensions") and p.suffix in (".dat", ".mca"))))
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=9) as out:
        for p in paths:
            content = p.read_bytes()
            if p.name == "level.dat":
                content = portable_level(content, title)
            entry = ZipInfo(f"{title}/{p.relative_to(save).as_posix()}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            out.writestr(entry, content, compress_type=ZIP_DEFLATED, compresslevel=9)


def stored_blocks(zf, prefix):
    """Random-access block lookup inside the zipped Minecraft 26.3 overworld."""
    @lru_cache(maxsize=None)
    def region(rx, rz):
        return zf.read(f"{prefix}/dimensions/minecraft/overworld/region/r.{rx}.{rz}.mca")

    @lru_cache(maxsize=None)
    def sections(cx, cz):
        raw = region(cx//32, cz//32)
        slot = 4 * (cx%32 + 32*(cz%32))
        offset = int.from_bytes(raw[slot:slot+3], "big") * 4096
        if not offset:
            raise ValueError(f"Missing saved chunk at {cx}, {cz}")
        length, codec = struct.unpack_from(">IB", raw, offset)
        if codec != 2:
            raise ValueError(f"Expected a zlib-compressed fixture chunk, found codec {codec}")
        block = read_nbt(zlib.decompress(raw[offset+5:offset+4+length]))
        if (block["xPos"], block["zPos"], block["Status"]) != (cx, cz, "minecraft:full"):
            raise ValueError(f"Incomplete or misplaced chunk at {cx}, {cz}")
        return {s["Y"]: s["block_states"] for s in block["sections"] if "block_states" in s}

    def lookup(x, y, z):
        section = sections(x//16, z//16)[y//16]
        palette = section["palette"]
        index = 0
        if len(palette) > 1:
            bits = max(4, (len(palette)-1).bit_length())
            linear = 256*(y%16) + 16*(z%16) + x%16
            word, idx = divmod(linear, 64//bits)
            index = (section["data"][word] >> (idx*bits)) & ((1 << bits)-1)
        return palette[index]

    return lookup


def audit(archive, title, fixture, floor):
    layout = load(fixture/"layout.json")
    views = load(fixture/"views.json")
    sites = load(fixture/"buildings.json")
    with ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise ValueError(f"Damaged archive: {archive}")
        def nbt(relative):
            return read_nbt(gzip.decompress(zf.read(f"{title}/{relative}")))
        def state(name):
            return nbt("data/minecraft/" + name + ".dat")["data"]
        level = nbt("level.dat")["Data"]
        if (level["LevelName"] != title or level["Version"]["Name"] != "26.3" or "singleplayer_uuid" in level
            or "file/"+PACK not in level["DataPacks"]["Enabled"]):
            raise ValueError("Incorrect level version, identity, title or enabled datapack")
        clock = state("world_clocks")["minecraft:overworld"]["total_ticks"]
        weather = state("weather")
        rules = state("game_rules")
        layers = state("world_gen_settings")["dimensions"]["minecraft:overworld"]["generator"]["settings"]["layers"]
        if (clock != 6000 or weather["raining"] or weather["thundering"] or rules["minecraft:advance_time"]
            or rules["minecraft:advance_weather"] or layers[-1]["block"] != "minecraft:"+floor):
            raise ValueError(f"Invalid day, weather, or terrain settings in {archive.name}")
        lookup = stored_blocks(zf, title)
        for b in layout["blocks"]:
            if lookup(b["x"], b["y"], b["z"]) != "minecraft:"+b["block_id"]:
                raise ValueError(f"Saved region disagrees with {b}")
        # Probe at the camera stations as well, including the original smooth-stone world.
        for v in views:
            if lookup(int(v["x"]//1), GROUND, int(v["z"]//1)) != "minecraft:"+floor:
                raise ValueError(f"Incorrect stored floor beneath {v['id']}")
    return dict(floor=floor, archive=archive.name, folder=title, bytes=archive.stat().st_size,
                sha256=digest(archive), clock_ticks=clock, raining=False, thundering=False,
                advance_time=False, advance_weather=False, matched_sites=len(sites),
                matched_layout_cells=sum(b["role"] != "floor_probe" for b in layout["blocks"]),
                matched_camera_floor_probes=len(views))


def build(smooth, concrete, out):
    out.mkdir(parents=True, exist_ok=False)
    worlds = []
    for fixture, floor in ((smooth, "smooth_stone"), (concrete, "light_gray_concrete")):
        title = "Exterior v1 - " + floor.replace("_", " ").capitalize()
        archive = out/(floor.replace("_", "-")+".zip")
        pack(fixture/"client/saves/Palette Lab", archive, title)
        worlds.append(audit(archive, title, fixture, floor))
    sites = []
    views = load(smooth/"views.json")
    for b in load(smooth/"buildings.json"):
        v = next(v for v in views if v["page"] == b["page"] and v["angle"] == "low-left")
        pose = load(smooth/"captures"/(v["id"]+".json"))
        yaw, pitch = pose["yaw_pitch_deg"]
        sites.append(dict(layout=b["layout"], palette=b["palette"], split=b["split"],
                          bounds_xyz=b["bounds"], low_left_view=v["id"],
                          teleport=f"/tp @s {v['x']:.5f} {v['feetY']:.5f} {v['z']:.5f} {yaw:.5f} {pitch:.5f}"))
    index = dict(schema=1, game="Minecraft Java Edition", version="26.3",
                 install="Extract either ZIP directly into the Java game directory's saves folder. Open the extracted 'Exterior v1' world in Singleplayer. Its level.dat belongs immediately inside the named world folder.",
                 bounds="World XYZ, lower inclusive and upper exclusive.", worlds=worlds, sites=sites)
    (out/"world-index.json").write_text(json.dumps(index, indent=2)+"\n", encoding="utf8")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-run", type=Path, required=True)
    parser.add_argument("--concrete-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.smooth_run, args.concrete_run, args.out)
