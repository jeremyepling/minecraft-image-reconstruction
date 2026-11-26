"""Local CPU smoke test of the authors' MoGe-2 ViT-S ONNX normal map on one image."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.atlas import digest
from engine.geometry import bilinear


MODEL = "Ruicheng/moge-2-vits-normal-onnx"
REVISION = "e50ffda41565591092adea54c6ac83d6212e1e23"
SHA256 = "24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0"
SOURCE_REVISION = "74fbce054ebed49800de42d0ad0e83495065719a"


def normal_image(vectors):
    # Authors' RGB visualization of camera normals (+X right, +Y down, +Z forward).
    return Image.fromarray(np.uint8(np.clip(vectors*[.5,-.5,-.5]+.5,0,1)*255))


def statistics(errors):
    return dict(n=len(errors), mean_deg=float(errors.mean()), median_deg=float(np.median(errors)),
                within_15_deg=float(np.mean(errors<=15)), within_30_deg=float(np.mean(errors<=30)))


def diagnostic(fixture, frame_id, image, image_hash, normal):
    from benchmark.camera_round import read
    from benchmark.surface_metrics import DOWNSAMPLE, truth

    matches = [f for f in read(fixture/"index.json")["frames"] if f["id"]==frame_id]
    if len(matches)!=1 or matches[0]["image_sha256"]!=image_hash:
        raise ValueError("Diagnostic frame ID and screenshot SHA-256 must match the fixture")
    f = matches[0]
    t = truth(read(fixture/f["pose"]),read(fixture/f["labels"])["faces"])
    ids = t["id_map"]
    y,x = np.where(ids>0)
    if not len(x):
        raise ValueError("Diagnostic frame has no labeled visible surface samples")
    xy = (np.column_stack((x,y))+.5)*DOWNSAMPLE*np.array(normal.shape[1::-1])/image.size
    samples = bilinear(normal,xy)
    lengths = np.linalg.norm(samples,axis=1,keepdims=True)
    if np.any(lengths<1e-6):
        raise ValueError("Undefined normal on a labeled surface")
    truth_normals = np.zeros((len(t["patches"])+1,3))
    directions = np.zeros(len(truth_normals),dtype=object)
    for pid,p in t["patches"].items():
        truth_normals[pid] = p["normal_camera"]
        axis,sign = p["axis"],p["normal"][p["axis"]]
        directions[pid] = (("west","east"),("down","up"),("north","south"))[axis][int(sign>0)]
    expected = truth_normals[ids[y,x]]
    errors = np.degrees(np.arccos(np.clip(((samples/lengths)*expected).sum(1),-1,1)))
    tags = directions[ids[y,x]]
    report = dict(frame=frame_id,floor=f.get("floor","smooth_stone"),split=f["split"],
                  evaluation="Signed camera-normal angles on every labeled foreground sample; quarter-resolution native pixel centers; bilinear prediction sampling and renormalization.",
                  diagnostic=statistics(errors),directions={d:statistics(errors[tags==d]) for d in sorted(set(tags))})
    ref = normal_image(truth_normals[ids])
    ref.paste((34,34,34),mask=Image.fromarray(ids==0))
    low = np.maximum(0,[x.min()-6,y.min()-6])*DOWNSAMPLE
    high = np.minimum(image.size,(np.array([x.max(),y.max()])+7)*DOWNSAMPLE)
    return report,ref,tuple(map(int,[*low,*high]))


def comparison(source, panels, box):
    width,height = 450,250
    scale = min(width/(box[2]-box[0]),height/(box[3]-box[1]))
    size = (max(1,round((box[2]-box[0])*scale)),max(1,round((box[3]-box[1])*scale)))
    sheet = Image.new("RGB",((width+12)*len(panels)+12,height+52),"#f5f5f0")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf",19)
    except OSError:
        font = ImageFont.load_default(size=19)
    for i,(title,im,resampling) in enumerate(panels):
        left = 12+i*(width+12)
        draw.text((left,9),title,fill="#142522",font=font)
        crop = im.resize(source.size,resampling).crop(box).resize(size,resampling)
        sheet.paste(crop,(left+(width-size[0])//2,42+(height-size[1])//2))
    return sheet


def probe(checkpoint, image, out, fixture=None, frame_id=None):
    import onnxruntime as ort

    if (fixture is None)!=(frame_id is None):
        raise ValueError("Use --fixture and --frame-id together for optional truth scoring")
    if digest(checkpoint)!=SHA256:
        raise ValueError("Checkpoint differs from the authors' pinned MoGe-2 ViT-S ONNX graph")
    with Image.open(image) as opened:
        source = opened.convert("RGB")
    resized = source.copy()
    resized.thumbnail((720,720),Image.Resampling.LANCZOS)
    batch = (np.asarray(resized,dtype=np.float32)/255.).transpose(2,0,1)[None]
    options = ort.SessionOptions()
    options.intra_op_num_threads,options.inter_op_num_threads = 4,1
    start = perf_counter()
    session = ort.InferenceSession(str(checkpoint),sess_options=options,providers=["CPUExecutionProvider"])
    ready = perf_counter()
    raw, = session.run(["normal"],dict(image=batch,num_tokens=np.array(1200,dtype=np.int64)))
    done = perf_counter()
    if raw.shape!=(1,resized.height,resized.width,3) or not np.isfinite(raw).all():
        raise ValueError(f"Unexpected normal map {raw.shape}")
    vectors = raw[0]
    preview = normal_image(vectors)
    out.mkdir(parents=True,exist_ok=True)
    preview.save(out/"normal.png")
    np.savez_compressed(out/"normals.npz",normal=vectors)
    image_hash = digest(image)
    report = dict(model=MODEL,checkpoint_revision=REVISION,checkpoint_sha256=SHA256,
                  checkpoint_bytes=checkpoint.stat().st_size,upstream_source_revision=SOURCE_REVISION,
                  docs=f"https://github.com/microsoft/MoGe/blob/{SOURCE_REVISION}/docs/onnx.md",
                  upstream_graph=f"https://huggingface.co/{MODEL}/blob/{REVISION}/model.onnx",
                  image_sha256=image_hash,source_size=source.size,input_size=resized.size,
                  preprocessing="Aspect-preserving Lanczos, RGB float32 in [0,1], NCHW",num_tokens=1200,
                  requested_output=["normal"],raw_normal_shape=list(raw.shape),onnxruntime=ort.__version__,
                  providers=session.get_providers(),threads=[4,1],loading_seconds=ready-start,
                  inference_seconds=done-ready,diagnostic=None,
                  artifacts=dict(normal_map="normal.png",raw_vectors="normals.npz",comparison="comparison.jpg"))
    panels = [("Screenshot",source,Image.Resampling.LANCZOS),("MoGe-2 normals",preview,Image.Resampling.BILINEAR)]
    box = (0,0,*source.size)
    if fixture is not None:
        extra,ref,box = diagnostic(fixture,frame_id,source,image_hash,vectors)
        report.update(extra)
        panels.append(("Recorded surface normals",ref,Image.Resampling.NEAREST))
    comparison(source,panels,box).save(out/"comparison.jpg",quality=94)
    report["comparison_native_crop_xyxy"] = box
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","image","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--fixture",type=Path,help="Optional source fixture directory for the diagnostic")
    parser.add_argument("--frame-id",help="Exact ID of the fixture screenshot")
    args = parser.parse_args()
    result = probe(args.checkpoint,args.image,args.out,args.fixture,args.frame_id)
    print(json.dumps(dict(inference_seconds=result["inference_seconds"],diagnostic=result["diagnostic"])))
