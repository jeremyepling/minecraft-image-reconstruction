# Vanilla capture runtime

Use Python 3.11 or newer and a JDK compatible with the selected release. Minecraft 26.1 through 26.3 use JDK 25. Install the vanilla client and assets locally. The runtime discovers the launcher installation and also recognizes NeoForm's cached client, manifest, libraries and assets. Pass `--minecraft-dir`, `--gradle-home`, `--manifest`, `--jar` or `--java-home` to change those locations. Client and library SHA-1 values are checked against the selected launch manifest.

From the repository root:

```console
python -m runtime.install 26.3 --destination runs/minecraft-cache --plan
python -m runtime.install 26.3 --destination runs/minecraft-cache
```

The optional installer reads the official Mojang manifests and populates a separate Minecraft directory. Supply `--minecraft-dir runs/minecraft-cache` to the runtime commands when using this cache and use `runs/minecraft-cache/versions/26.3/26.3.jar` for the fixture's JAR argument. Existing launcher or NeoForm installations can start directly with the steps below.

```console
python -m runtime.capture compile --version 26.3
python -m runtime.capture registry --version 26.3 --out runs/26.3/registry.json
python -m calibration.fixture --version 26.3 --jar PATH_TO_CLIENT_JAR --registry runs/26.3/registry.json --out runs/26.3
python -m runtime.capture capture --version 26.3 --run runs/26.3
```

The last command starts a private, windowed local client with a separate game directory under the run. The driver teleports the in-game camera, executes each fixture page, reads back the server's block states, and saves native screenshot pixels plus actual camera pose, FOV, dimensions, HUD, time, weather and game version. A full capture takes a few minutes. The game writes its save in `client/saves/Palette Lab`. Pick a new output directory for a new capture to preserve earlier worlds and evidence.

The driver compiles against the exact JAR being tested. For new client releases, try the compile check followed by the complete registry, world and image checks. A new release that changes client APIs may require an adapter. The 26.1 adapter handles older HUD, camera and render-target accessors. Newer releases use the current adapter. The fixture generator independently selects the world's data and pack versions. Obfuscated older distributions need additional mapped client support for this Java driver. Public scripts consume local game assets and generate calibration data on demand.

## Matched live conditions for the shape benchmark

After generating a shape fixture in `RUN`, capture its full-resolution views. Stage and run two additional private clients for the adjacency panels. All three use the same block coordinates and camera poses.

```console
python -m runtime.capture capture --version 26.3 --run RUN
python -m runtime.benchmark_conditions RUN --jar CLIENT_JAR
python -m runtime.capture capture --version 26.3 --run RUN/conditions/small --resolution 432 270
python -m runtime.capture capture --version 26.3 --run RUN/conditions/warm-shader --resource-pack RUN/conditions/warm-shader/warm-terrain.zip
python -m runtime.benchmark_audit RUN
```

The roof on each adjacency panel casts game skylight shade. Per-block readbacks record both sky and emitted light in the cell immediately south of its face. The small run uses a native 432 × 270 framebuffer. The shader condition applies cool shadows, warm highlights and a contrast adjustment inside the release's terrain fragment shader. The resource pack is generated locally from the matching client JAR. It is enabled only in the private capture world. The audit pairs actual framebuffer pixels across matching camera poses and records the active resource packs, lighting range, projected block size, and any incorrect block IDs. The contact sheet and JSON report are written to the run directory.
