"""Summarize photographed seams and verify the identity of every retained pack."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont
from benchmark.camera_model_controls import inventory_hash
from benchmark.typed_structures import pooled_scores
from benchmark.visible_wall_corner_run import DEV, HOLD, OUT, ROOT, SOURCE, COHORTS, NATIVE_RUN, fixed, frames, policy_pack, read, save, decoded_pack

REVIEW = ROOT/'benchmark/reviews/visible-wall-corner-01'


def overlay(rows, entries):
    picks = [DEV[0], HOLD[2], HOLD[4]]
    canvas = Image.new('RGB',(1080,320),'#101820')
    view = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('arial.ttf',16)
    small = ImageFont.truetype('arial.ttf',13)
    for i,key in enumerate(picks):
        source = entries[key]['rgb']
        with Image.open(source) as opened:
            image = opened.convert('RGB')
        painter = ImageDraw.Draw(image)
        for seam in rows[key]['audit']['seams']:
            color = '#25e2da' if sum(seam['anchors']) == 1 else '#ffba59'
            painter.line([tuple(p) for p in seam['seam_px']],fill=color,width=8)
        image.thumbnail((360,225),Image.Resampling.LANCZOS)
        canvas.paste(image,(360*i,0))
        x = 360*i + 8
        view.text((x,232),key.replace('validation-','').replace('train-','').replace('test-',''),font=font)
        view.text((x,254),f"{rows[key]['audit']['walls']} vertical masks, {len(rows[key]['audit']['seams'])} measured seams",font=small)
    view.text((10,291),'Cyan marks a seam with one anchored wall. Amber marks a seam with zero or two anchors.',font=font)
    canvas.save(REVIEW/'source-seams.jpg',quality=90)


def main():
    REVIEW.mkdir(parents=True,exist_ok=True)
    entries = {e['id']: dict(e,rgb=__import__('pathlib').Path(e['frame'])/'rgb.png') for e in frames(SOURCE,COHORTS)}
    entries.update((e['id'],e) for e in fixed.fixed_frames())
    base = {r['id']:r for r in read(ROOT/'benchmark/reviews/camera-native-recovery-controls-01/independent-default-cases.json.gz')}
    rows = {}
    with ZipFile(fixed._release()[2]) as archive:
        for key in DEV + HOLD:
            row = read(OUT/key/'result.json')
            b, e = base[key],entries[key]
            if e['dataset']=='fixed-104':
                raw = archive.read(key+'/reconstruction.zip')
            else:
                folder = NATIVE_RUN/'builds'/e['dataset']/key if b['native_floor_coverage']=='candidate_export' else policy_pack(e,b)
                raw = (folder/'reconstruction.zip').read_bytes()
            assert sha256(raw).hexdigest()==b['pack_sha256']==row['incumbent_sha256']
            cells = decoded_pack(BytesIO(raw))
            assert inventory_hash(cells)==b['decoded_sha256'] and len(cells)==b['exported']
            assert row['kept'] and not row['audit']['changes']
            row.update(dataset=e['dataset'], split=e['split'], reference_sha256=b['reference_sha256'],
                       decoded_sha256=b['decoded_sha256'], geometry=b['geometry'],scores=b['scores'])
            rows[key]=row
    save(REVIEW/'cases.json.gz',list(rows.values()))
    full = read(ROOT/'benchmark/reviews/camera-native-recovery-controls-01/independent-default-summary.json')['datasets']
    stats = {}
    for ds in full:
        subset = [r for r in rows.values() if r['dataset']==ds]
        stats[ds] = dict(tested=len(subset),available=full[ds]['available'],
                         geometry=pooled_scores([r['geometry'] for r in subset]),
                         scores=pooled_scores([r['scores'] for r in subset]))
    save(REVIEW/'summary.json',dict(pilot=stats,declined=Counter(s['declined'] for r in rows.values() for s in r['audit']['seams']),
                                    carried_packs=13, eligible_alternate_exports=0))
    overlay(rows,entries)


if __name__=='__main__':
    main()
