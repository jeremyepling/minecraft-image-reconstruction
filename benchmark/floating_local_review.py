"""Adapt normal Floating local-grid runs to the same paired stage report and visual review queue."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from benchmark.expanded_scores import evaluate_pack, load_reference
from benchmark.floating_stage_report import COUNTS, DATASETS, group, handoff_metrics, hash_file, read
from benchmark.floating_visual_review import identity, paint, select, strict_delta


def checked_arm(folder, entry, variant, reference, shared_handoff):
    case = read(folder / "case.json")
    score = case["score"]
    evidence = case["evidence"]
    sha = entry["variants"][variant]["source_sha256"]
    if (case["id"] != entry["id"] or case["variant"] != variant or evidence["source_sha256"] != sha
            or score["source_sha256"] != sha or score["reference_sha256"] != reference.hashes
            or score["dataset_manifest_sha256"] != entry["dataset_manifest_sha256"]):
        raise ValueError(f"Case belongs to another fixture or variant: {folder}")
    projection = evidence["sheet_or_global_P"]
    fresh, aligned = evaluate_pack(entry, folder / "reconstruction.zip", dict(P=projection,
                                  scale_status="supported" if projection is not None else "uncertain"), reference=reference)
    archive = read(folder / "aligned-placements.json.gz")
    expected = {tuple(row[:3]):row[3] for row in archive}
    if len(expected) != len(archive) or aligned != expected or fresh["pack_sha256"] != score["pack_sha256"]:
        raise ValueError(f"Decoded local-grid pack differs from its saved registration: {folder}")
    scores = {}
    for measure in ("geometry","scores"):
        for regime in ("strict","facade"):
            counts = {k:score[measure][regime][k] for k in COUNTS}
            if counts != {k:fresh[measure][regime][k] for k in COUNTS}:
                raise ValueError(f"Decoded local-grid scores differ: {folder}")
            scores[f"{measure}_{regime}"] = counts
    strict = scores["geometry_strict"]["correct"]
    stages = dict(axes=(int(projection is not None), 1), material=(scores["scores_strict"]["correct"],strict),
                  placement=(strict, len(reference.truth)))
    package_path = folder / "surfaces.json"
    if package_path.exists():
        package = read(package_path)
        provenance = package["provenance"]
        if (package["sourceSha256"] != sha or provenance["arms"]["Candidate"]["packSha256"] != fresh["pack_sha256"]):
            raise ValueError(f"Surface evidence differs from the scored output: {folder}")
        for handoff in (folder / "source-handoff", shared_handoff):
            receipt = handoff / "build-manifest.json"
            if receipt.is_file() and hash_file(receipt) == provenance["handoffSha256"]:
                stages.update(handoff_metrics(handoff, {"source_sha256":sha}, {"counts_sha256":provenance["countsSha256"]}))
                break
    # No selected-component ID contract in this runner. Distinct sheet names across arms
    # cannot establish that the image contains different surfaces.
    return dict(scores=scores, stages=stages, exported=len(aligned), unresolved=bool(aligned and projection is None),
                selected=[], pack_sha256=fresh["pack_sha256"]), aligned


def prepare(fixture, baseline, candidate, variant, output, label, available, names, *, case_ids=()):
    from training.floating_combined import select_active_cases
    entries = [e for e in read(fixture)["cases"] if e["dataset"] != "decorative-builds-v1"]
    entries = list(select_active_cases(entries))
    if not entries:
        raise ValueError("No eligible visible views in this fixture.")
    if len({e["id"] for e in entries}) != len(entries):
        raise ValueError("Use one dataset and variant per repeated frame ID.")
    if available < len(entries):
        raise ValueError("Available views must include every tested view.")
    frames, metadata = [], {}
    for e in entries:
        marker = e["id"]
        sha = e["variants"][variant]["source_sha256"]
        source = Path(e["variants"][variant]["image"])
        if hash_file(source) != sha:
            raise ValueError(f"Wrong source image: {marker}")
        reference = load_reference(e)
        arms, aligned = {}, []
        for key, root in (("baseline", baseline), ("current", candidate)):
            arms[key], positions = checked_arm(root / marker, e, variant, reference, baseline / marker / "source-handoff")
            aligned.append(positions)
        gained, lost, paired = strict_delta(reference.truth, *aligned)
        change = dict(changed_pack=arms["baseline"]["pack_sha256"] != arms["current"]["pack_sha256"],
                      dropped=[], added=[], disjoint=False, paired=paired)
        frames.append(dict(id=marker, dataset=e["dataset"], panel=variant, source_sha256=sha,
                           reference_sha256=reference.hashes, arms=arms, changes={"current":change}))
        metadata[marker] = e, gained, lost
    comparison = hashlib.sha256(json.dumps(frames, sort_keys=True, separators=(",",":")).encode()).hexdigest()
    groups = {DATASETS[0]:group(frames, tuple(names), available), DATASETS[1]:group([], tuple(names), 40)}
    report = dict(schema=1, comparison_sha256=comparison, cohort_label=label,names=names,cohorts=groups,
                  frames=frames,perfect_stage=None)
    queue = output / "queue"
    queue.mkdir(parents=True,exist_ok=True)
    selected = select(frames,"current")
    by_id = {f["id"]:f for f in frames}
    for marker in case_ids:
        selected.setdefault(identity(by_id[marker]), []).append("Requested visual trace")
    cases = []
    for (_,_,marker), reasons in selected.items():
        f = by_id[marker]
        e, gained, lost = metadata[marker]
        original = Path(e["frame"])
        source = Path(e["variants"][variant]["image"])
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"))
        labels = read(original / "labels.json")
        with np.load(original / "labels.npz",allow_pickle=False) as z:
            front = z["face_id"]
        if front.shape != rgb.shape[:2]:
            raise ValueError(f"Reference dimensions differ: {marker}")
        if f["source_sha256"] != e["rgb_sha256"]:
            with Image.open(original / "rgb.png") as im:
                if not np.array_equal(np.asarray(im.convert("RGB"))[front>0],rgb[front>0]):
                    raise ValueError(f"Variant changed reference foreground pixels: {marker}")
        ids = np.zeros(int(front.max())+1,np.uint8)
        for face in labels["faces"]:
            if face["visible_pixels"]:
                xyz = tuple(face["xyz_world"])
                ids[face["id"]] = 1 if xyz in gained else 2 if xyz in lost else 0
        visible = ids[front]
        layers = Path("visible") / variant / marker
        (queue / layers).mkdir(parents=True,exist_ok=True)
        Image.fromarray(paint(paint(rgb,visible == 1,(0,214,225)),visible == 2,(255,66,135))).save(queue / layers / "placement.png")
        frame = output / "frames" / e["split"] / "frames" / marker
        frame.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source, frame / "rgb.png")
        for name in ("labels.npz","labels.json","camera.json"):
            shutil.copyfile(original / name,frame / name)
        cases.append(dict(id=marker,source_sha256=f["source_sha256"],reference_sha256=f["reference_sha256"],
                          baseline=dict(pack_sha256=f["arms"]["baseline"]["pack_sha256"]),
                          candidate=dict(pack_sha256=f["arms"]["current"]["pack_sha256"]),paired=f["changes"]["current"]["paired"],
                          component=None,reasons=reasons,cohort=DATASETS[0],panel=variant,folder=layers.as_posix()))
    (queue / "receipts.json").write_text(json.dumps(dict(comparison_sha256=comparison,cases=cases)),encoding="utf-8")
    return report, queue, output / "frames"
