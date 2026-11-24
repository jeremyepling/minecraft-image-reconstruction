"""Make a searchable, pixel-preserving atlas from a local vanilla Java client jar."""

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


TILE_SIZE = 128
PREFIX = "assets/minecraft/"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def discover(version):
    home = Path.home()
    minecraft = Path(os.environ.get("APPDATA", home / ".config")) / ".minecraft"
    gradle = Path(os.environ.get("GRADLE_USER_HOME", home / ".gradle"))
    for path in (
        minecraft / "versions" / version / f"{version}.jar",
        gradle / "caches/neoformruntime/artifacts" / f"minecraft_{version}_client.jar",
        home / ".minecraft/versions" / version / f"{version}.jar",
        home / "Library/Application Support/minecraft/versions" / version / f"{version}.jar",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find the {version} client locally. Pass --jar PATH.")


def resource_id(name):
    return name if ":" in name else "minecraft:" + name


def model_entries(value):
    for entry in value if isinstance(value, list) else [value]:
        if "model" in entry:
            yield resource_id(entry["model"])


def model_index(jar):
    """Resolve child texture overrides against inherited geometry and face texture aliases."""
    names = set(jar.namelist())
    face_blocks = defaultdict(set)
    particle_blocks = defaultdict(set)
    tinted_blocks = defaultdict(set)

    @lru_cache(maxsize=None)
    def inherited(model_id):
        ns, path = model_id.split(":", 1)
        path = f"assets/{ns}/models/{path}.json"
        if path not in names:
            # Built-in model parents can be implemented by the game renderer.
            return {}, []
        model = json.loads(jar.read(path))
        variables, elements = inherited(resource_id(model["parent"])) if "parent" in model else ({}, [])
        return {**variables, **model.get("textures", {})}, model.get("elements", elements)

    def resolve(value, variables):
        seen = set()
        while True:
            if isinstance(value, dict):
                value = value.get("sprite")
            if not isinstance(value, str) or not value.startswith("#"):
                break
            if value in seen:
                return None
            seen.add(value)
            value = variables.get(value[1:])
        return resource_id(value) if isinstance(value, str) else None

    states = sorted(n for n in names if n.startswith(PREFIX + "blockstates/") and n.endswith(".json"))
    for path in states:
        block = "minecraft:" + path.removeprefix(PREFIX + "blockstates/").removesuffix(".json")
        state = json.loads(jar.read(path))
        models = set()
        for variants in state.get("variants", {}).values():
            models.update(model_entries(variants))
        for part in state.get("multipart", []):
            models.update(model_entries(part["apply"]))
        for model in models:
            variables, elements = inherited(model)
            particle = resolve(variables.get("particle"), variables)
            if particle:
                particle_blocks[particle].add(block)
            for element in elements:
                for face in element.get("faces", {}).values():
                    texture = resolve(face.get("texture"), variables)
                    if texture:
                        face_blocks[texture].add(block)
                        if face.get("tintindex", -1) >= 0:
                            tinted_blocks[texture].add(block)
    return face_blocks, particle_blocks, tinted_blocks


def representative(image, animation):
    if animation is None:
        return image, None
    w, h = image.size
    fw = animation.get("width")
    fh = animation.get("height")
    if fw is None and fh is None:
        fw = fh = min(w, h)
    else:
        fw, fh = fw or w, fh or h
    if fw <= 0 or fh <= 0 or w % fw or h % fh:
        raise ValueError(f"Invalid animation grid {fw}x{fh} for image {w}x{h}")
    frames = animation.get("frames", [])
    first = frames[0] if frames else 0
    first = first["index"] if isinstance(first, dict) else first
    if not 0 <= first < (w // fw) * (h // fh):
        raise ValueError(f"Animation first frame {first} lies outside the sheet")
    x, y = first % (w // fw) * fw, first // (w // fw) * fh
    details = {"representative_index": first, "source_rect": [x, y, fw, fh],
               "grid_frames": (w // fw) * (h // fh), "animation": animation}
    return image.crop((x, y, x + fw, y + fh)), details


def make_atlas(jar_path, out, requested_version, force=False):
    png = out / "atlas.png"
    metadata = out / "atlas.json"
    settings = {"client_sha256": digest(jar_path), "generator_sha256": digest(Path(__file__)),
                "tile_size": TILE_SIZE}
    if not force and png.is_file() and metadata.is_file():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if (info.get("recipe") == settings and (not requested_version or info["version"] == requested_version)
                and info.get("atlas_sha256") == digest(png)):
            return info, "cached"

    with ZipFile(jar_path) as jar:
        version = json.loads(jar.read("version.json"))["id"]
        if requested_version and version != requested_version:
            raise ValueError(f"Requested {requested_version}, but the jar reports {version}")
        names = set(jar.namelist())
        paths = sorted(n for n in names if n.startswith(PREFIX + "textures/block/") and n.endswith(".png"))
        if not paths:
            raise ValueError("Client jar contains no block texture PNGs")
        face_blocks, particles, tinted = model_index(jar)
        columns = math.ceil(math.sqrt(len(paths)))
        rows = math.ceil(len(paths) / columns)
        textures = []
        sprites = []
        for path in paths:
            texture_id = "minecraft:" + path.removeprefix(PREFIX + "textures/").removesuffix(".png")
            meta_path = path + ".mcmeta"
            meta = json.loads(jar.read(meta_path)) if meta_path in names else {}
            with Image.open(io.BytesIO(jar.read(path))) as raw:
                source_size = list(raw.size)
                art, animation = representative(raw.convert("RGBA"), meta.get("animation"))
            sprites.append(art)
            faces = face_blocks[texture_id]
            fallback = particles[texture_id] - faces
            entry = {"id": texture_id, "source": path, "source_size": source_size,
                     "blocks": sorted(faces or fallback), "face_blocks": sorted(faces),
                     "particle_only_blocks": sorted(fallback), "tinted_blocks": sorted(tinted[texture_id]),
                     "mapping": "model_face" if faces else "particle_only" if fallback else "unmapped"}
            if animation is not None:
                entry["animation"] = animation
            textures.append(entry)
    tile = max(TILE_SIZE, max(max(art.size) for art in sprites))
    sheet = Image.new("RGBA", (columns * tile, rows * tile))
    for index, (art, entry) in enumerate(zip(sprites, textures)):
        scale = tile // max(art.size)
        w, h = art.width * scale, art.height * scale
        x = index % columns * tile + (tile - w) // 2
        y = index // columns * tile + (tile - h) // 2
        sheet.paste(art.resize((w, h), Image.Resampling.NEAREST), (x, y))
        entry.update(rect=[x, y, w, h], scale=scale)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp_png, tmp_json = Path(tmp) / png.name, Path(tmp) / metadata.name
        sheet.save(tmp_png, compress_level=6)
        info = {"format_version": 1, "version": version, "atlas": png.name, "size": list(sheet.size),
                "atlas_sha256": digest(tmp_png), "recipe": settings, "columns": columns, "rows": rows,
                "scope": "Vanilla assets/minecraft/textures/block/**/*.png; untinted RGBA. "
                         "Animated sprites show their first scheduled frame. Block IDs are derived from "
                         "blockstate model faces, with particle-only associations flagged separately. "
                         "Unmapped sprites include overlays and textures handled by custom renderers.",
                "textures": textures}
        tmp_json.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp_png.replace(png)
        tmp_json.replace(metadata)
    return info, "built"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="For auto-discovery, or to verify an explicitly supplied jar")
    parser.add_argument("--jar", type=Path, help="Vanilla Java Edition client jar")
    parser.add_argument("--out", required=True, type=Path, help="Directory for atlas.png and atlas.json")
    parser.add_argument("--force", action="store_true", help="Regenerate even when the cache is current")
    args = parser.parse_args()
    if not args.jar and not args.version:
        parser.error("pass --version or --jar")
    info, result = make_atlas(args.jar or discover(args.version), args.out, args.version, args.force)
    print(f"{result}: {info['version']}, {len(info['textures'])} textures, "
          f"{info['size'][0]}x{info['size'][1]} pixels, {args.out}")


if __name__ == "__main__":
    main()
