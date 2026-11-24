"""Populate a local vanilla client cache from Mojang's version and asset manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen

from .capture import applies

VERSION_INDEX = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"
ASSET_OBJECTS = "https://resources.download.minecraft.net"


def read_json(url, expected=None):
    with urlopen(url, timeout=45) as stream:
        data=stream.read()
    if expected and hashlib.sha1(data).hexdigest()!=expected:
        raise ValueError(f"Manifest checksum mismatch: {url}")
    return data,json.loads(data)


def checked_path(root,relative):
    target=(root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Invalid manifest path: {relative}")
    return target


def verify(path, info):
    if not path.is_file() or path.stat().st_size!=info["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha1").hexdigest()==info["sha1"]


def fetch(job):
    path,info=job
    if verify(path,info):
        return
    path.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
        temporary=Path(tmp.name)
    try:
        with urlopen(info["url"], timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response,out,1024*1024)
        if not verify(temporary,info):
            raise ValueError(f"Download checksum mismatch: {info['url']}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def install(version, destination, plan_only=False):
    if not re.fullmatch(r"[A-Za-z0-9._-]+",version) or version in (".",".."):
        raise ValueError("Invalid version ID")
    root=destination.resolve()
    _,versions=read_json(VERSION_INDEX)
    entry=next((v for v in versions["versions"] if v["id"]==version),None)
    if entry is None:
        raise ValueError(f"Mojang's version index has no release or snapshot named {version}")
    raw,manifest=read_json(entry["url"],entry["sha1"])
    if manifest["id"]!=version:
        raise ValueError("Version manifest ID mismatch")
    base=Path("versions")/version
    jobs={checked_path(root,base/f"{version}.jar"):manifest["downloads"]["client"]}
    for lib in manifest["libraries"]:
        if applies(lib):
            artifact=lib["downloads"]["artifact"]
            jobs[checked_path(root,Path("libraries")/artifact["path"])]=artifact
    index=manifest["assetIndex"]
    _,assets=read_json(index["url"],index["sha1"])
    jobs[checked_path(root,Path("assets/indexes")/(index["id"]+".json"))]=index
    for obj in assets["objects"].values():
        digest=obj["hash"]
        if not re.fullmatch(r"[0-9a-f]{40}",digest):
            raise ValueError("Malformed asset checksum")
        rel=f"{digest[:2]}/{digest}"
        jobs[checked_path(root,Path("assets/objects")/rel)]={"sha1":digest,"size":obj["size"],"url":f"{ASSET_OBJECTS}/{rel}"}
    print(f"{version}: {len(jobs)} files, {sum(j['size'] for j in jobs.values())/1_000_000:.1f} MB before cache reuse")
    if plan_only:
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(fetch,jobs.items()):
            pass
    # Commit the manifest only after all of its referenced files have validated.
    manifest_path=checked_path(root,base/f"{version}.json")
    with tempfile.NamedTemporaryFile(dir=manifest_path.parent,delete=False) as out:
        out.write(raw)
    os.replace(out.name,manifest_path)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("version")
    p.add_argument("--destination",type=Path,required=True)
    p.add_argument("--plan",action="store_true",help="Inspect manifests and print the download size")
    args=p.parse_args()
    install(args.version,args.destination,args.plan)
