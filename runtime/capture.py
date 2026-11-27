"""Compile against an installed vanilla Java client and run isolated calibration worlds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from zipfile import ZipFile

if __package__:
    from .mode import require_game_runtime
else:  # Preserve the standalone runtime/capture.py entry point.
    from mode import require_game_runtime

HERE = Path(__file__).resolve().parent


def installed_directory():
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / ".minecraft"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/minecraft"
    return Path.home() / ".minecraft"


def applies(lib):
    """Launcher library rules are ordered; the last matching rule wins."""
    system = {"win32":"windows", "darwin":"osx"}.get(sys.platform, "linux")
    machine = platform.machine().lower()
    arch = {"amd64":"x86_64", "arm64":"aarch64"}.get(machine, machine)
    rules = lib.get("rules", [])
    allowed = not rules
    for rule in rules:
        os_rule = rule.get("os", {})
        if os_rule.get("name", system) != system:
            continue
        if not re.search(os_rule.get("arch", ".*"), arch):
            continue
        if not re.search(os_rule.get("version", ".*"), platform.release()):
            continue
        if any(rule.get("features", {}).values()):
            continue
        allowed = rule["action"] == "allow"
    parts = lib["name"].split(":")
    if len(parts) > 3 and parts[3].startswith("natives-"):
        os_name = {"osx":"macos"}.get(system,system)
        machine_suffix = {"aarch64":"-arm64","x86":"-x86","i386":"-x86"}.get(arch, "")
        allowed &= parts[3] == "natives-" + os_name + machine_suffix
    return allowed


def first_file(paths, description, sha1=None):
    for path in paths:
        if path and path.is_file():
            if sha1:
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream,"sha1").hexdigest() != sha1:
                        continue
            return path.resolve()
    raise FileNotFoundError(f"Missing {description}; install the selected vanilla version with its dependencies.")


def classpath(args):
    version = args.version
    if not re.fullmatch(r"[A-Za-z0-9._-]+",version) or version in (".",".."):
        raise ValueError("Invalid version ID")
    installed = args.minecraft_dir or installed_directory()
    cache = args.gradle_home / "caches"
    neo = cache / "neoformruntime/artifacts"
    base = installed / "versions" / version
    manifests=[neo / f"minecraft_{version}_version_manifest.json",base/f"{version}.json"]
    clients=[neo / f"minecraft_{version}_client.jar",base/f"{version}.jar"]
    if args.minecraft_dir:
        manifests.reverse()
        clients.reverse()
    manifest_file = first_file([args.manifest] if args.manifest else manifests,"version manifest")
    manifest = json.loads(manifest_file.read_text())
    if manifest["id"] != version:
        raise ValueError(f"Manifest is for {manifest['id']}, requested {version}")
    jar = first_file([args.jar] if args.jar else clients,
                     "client JAR",manifest["downloads"]["client"]["sha1"])
    with ZipFile(jar) as archive:
        metadata = json.loads(archive.read("version.json"))
    with jar.open("rb") as stream:
        metadata["client_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if metadata["id"] != version:
        raise ValueError("Client JAR belongs to another version")
    jars = [jar]
    for lib in manifest["libraries"]:
        if not applies(lib):
            continue
        artifact = lib["downloads"]["artifact"]
        rel = Path(artifact["path"])
        group,name,num,*_ = lib["name"].split(":")
        gradle = cache / "modules-2/files-2.1" / group / name / num
        jars.append(first_file([installed/"libraries"/rel, *gradle.glob("*/"+rel.name)],
                               lib["name"],artifact["sha1"]))
    index = manifest["assetIndex"]["id"]
    assets = next((p for p in (installed/"assets",cache/"neoformruntime/assets") if
                   (p/"indexes"/f"{index}.json").is_file()), None)
    return os.pathsep.join(map(str,jars)),metadata,index,assets


def run_client(cmd, game, log_file, timeout, predict_frames=None):
    with log_file.open("w", encoding="utf8") as log:
        with subprocess.Popen(cmd, cwd=game, stdout=log, stderr=subprocess.STDOUT) as client:
            deadline = time.monotonic() + timeout

            def capture_active():
                status = client.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f"Client exited with status {status}. See {log_file}")
                    return False
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Client timed out. See {log_file}")
                return True

            try:
                if predict_frames:
                    predict_frames(capture_active)
                client.wait(timeout=max(0, deadline - time.monotonic()))
                capture_active()
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Client timed out. See {log_file}") from exc
            finally:
                if client.poll() is None:
                    client.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action",required=True)
    for name in ("registry","compile","capture"):
        mode=modes.add_parser(name)
        mode.add_argument("--version", required=True)
        mode.add_argument("--jar",type=Path)
        mode.add_argument("--manifest",type=Path)
        mode.add_argument("--minecraft-dir",type=Path)
        mode.add_argument("--gradle-home",type=Path,default=Path(os.environ.get("GRADLE_USER_HOME", Path.home()/".gradle")))
        mode.add_argument("--java-home",type=Path)
        if name=="registry":
            mode.add_argument("--out",type=Path,required=True)
        elif name=="capture":
            mode.add_argument("--run",type=Path,required=True)
            mode.add_argument("--resolution",nargs=2,type=int,metavar=("WIDTH","HEIGHT"),default=(1440,900))
            mode.add_argument("--render-distance", type=int, choices=range(4, 17), default=8)
            mode.add_argument("--resource-pack",type=Path,help="Local resource pack to enable in this private client")
            mode.add_argument("--predict-split",choices=("train","dev","test"),
                              help="Predict geometry for this shape-fixture split as captures arrive")
            mode.add_argument("--methods",nargs="+",help="Plane prediction methods (default: all)")
            mode.add_argument("--workers",type=int,help="Prediction processes (default: 2)")
            mode.add_argument("--prediction-dir",type=Path,help="Private prediction directory for this fixture")
    args=parser.parse_args()
    if args.action in ('capture', 'registry'):
        require_game_runtime(args.action)
    predict_frames = None
    if args.action=="capture":
        if args.predict_split:
            from benchmark import planes
            methods = args.methods or planes.METHODS
            if set(methods)-set(planes.METHODS):
                parser.error("choose prediction methods from " + ", ".join(planes.METHODS))
            if args.workers is not None and args.workers<1:
                parser.error("--workers must be positive")
            def predict_frames(capture_active):
                planes.run(args.run, [args.predict_split], args.workers or 2, 0, methods,
                           args.prediction_dir, capture_active)
        elif any(value is not None for value in (args.methods, args.workers, args.prediction_dir)):
            parser.error("prediction options require --predict-split")
    cp,meta,index,assets=classpath(args)
    java=str(args.java_home/"bin/java") if args.java_home else "java"
    javac=str(args.java_home/"bin/javac") if args.java_home else "javac"
    source = "CoreRegistry" if args.action=="registry" else "PaletteCapture"
    with tempfile.TemporaryDirectory(prefix="mcir-driver-") as tmp:
        sources = [HERE/f"{source}.java"]
        if source=="PaletteCapture":
            adapter="26_1" if re.fullmatch(r"26\.1(?:\.\d+)?", args.version) else "current"
            sources.append(HERE/"adapters"/adapter/"ClientView.java")
        compiled=subprocess.run([javac,"-encoding","UTF-8","-proc:none","-cp",cp,"-d",tmp,*map(str,sources)])
        if compiled.returncode:
            raise SystemExit(f"The {source} driver did not compile for {args.version}. Check the JDK and client API diagnostics above.")
        if args.action=="compile":
            return
        cmd=[java,"--enable-native-access=ALL-UNNAMED","--add-exports=java.base/jdk.internal.misc=ALL-UNNAMED",
             "-cp",tmp+os.pathsep+cp]
        if args.action=="registry":
            path=args.out.resolve()
            path.parent.mkdir(exist_ok=True,parents=True)
            subprocess.run(cmd+[source,str(path)],check=True,cwd=tmp)
            return
        if assets is None:
            raise FileNotFoundError(f"Missing installed assets for index {index}")
        root=args.run.resolve()
        views=json.loads((root/"views.json").read_text())
        layout=json.loads((root/"layout.json").read_text())
        if layout.get("version")!=args.version:
            raise ValueError("Layout version must equal the requested game version")
        if layout.get("client_sha256",meta["client_sha256"])!=meta["client_sha256"]:
            raise ValueError("Fixture and capture use different client JARs")
        if not views or any(not re.fullmatch(r"[a-zA-Z0-9_-]+",v["id"]) for v in views):
            raise ValueError("Invalid or empty view list")
        game=root/"client"
        if game.exists() or (root/"captures").exists():
            raise FileExistsError("Use a fresh capture directory; this run already has a client or captures")
        save=game/"saves/Palette Lab"
        width,height=args.resolution
        if min(width,height)<64:
            raise ValueError("Capture dimensions must be at least 64 pixels")
        if args.resource_pack and not args.resource_pack.is_file():
            raise FileNotFoundError(args.resource_pack)
        shutil.copytree(root/"fixture/saves/Palette Lab",save)
        packs=[]
        if args.resource_pack:
            private=game/"resourcepacks"/"capture-profile.zip"
            private.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.resource_pack,private)
            packs=["file/"+private.name]
            cmd.insert(1,"-Dpalette.capture.pack="+packs[0])
        (game/"options.txt").write_text(f"version:{meta['world_version']}\nonboardAccessibility:false\ntutorialStep:none\n"
            f"resourcePacks:{json.dumps(packs)}\n"
            f"pauseOnLostFocus:false\nrenderDistance:{args.render_distance}\nsimulationDistance:5\nmaxFps:60\n"
            "fullscreen:false\nenableVsync:false\nbobView:false\nfovEffectScale:0.0\n")
        # Dialogs must be handled in this private client; capture the game's own framebuffer.
        cmd[1:1]=["-Xms256m","-Xmx2200m","-XX:StackShadowPages=32",f"-Dpalette.capture.dir={root/'captures'}"]
        if sys.platform=="darwin":
            cmd.insert(1,"-XstartOnFirstThread")
        cmd.extend([source,"--version",args.version,"--username","PaletteCamera","--uuid",
                    uuid.uuid5(uuid.NAMESPACE_DNS,"local.palette.camera").hex,"--accessToken","0",
                    "--versionType","release","--gameDir",str(game),"--assetsDir",str(assets),
                    "--assetIndex",index,"--width",str(width),"--height",str(height),"--quickPlaySingleplayer",save.name])
        log_file=root/"client-run.log"
        run_client(cmd, game, log_file, max(420,len(views)*20+120), predict_frames)
        for v in views:
            stem=root/"captures"/v["id"]
            if not stem.with_suffix(".png").is_file() or json.loads(stem.with_suffix(".json").read_text())["minecraft_version"]!=args.version:
                raise RuntimeError(f"Capture verification failed for {stem.name}")


if __name__=="__main__":
    main()
