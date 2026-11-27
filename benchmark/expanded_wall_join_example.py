"""Draw the source evidence for the two harmful one-edge castle joins."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import ROOT, read, save
from benchmark.expanded_wall_grids import OUT, cases
from benchmark.expanded_wall_join_audit import true_wall_origins


def font(size):
    for name in ('C:/Windows/Fonts/segoeui.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    name = 'test-castle-03-limestone-high-right'
    case = next(c for c in cases() if c['id'] == name)
    frame = Path(case['frame'])
    image = Image.open(frame / 'rgb.png').convert('RGB')
    receipt = read(OUT / 'signed/assembly/connected-atlas/sky' / name / 'score.json')
    packet = read(OUT / 'signed/charts/sky' / name / 'charts.json')
    charts = {c['id']: c for c in packet['charts']}
    origins, _ = true_wall_origins(packet['charts'], read(frame / 'labels.json'))
    translation = receipt['score']['fixed']['translation']
    drift = {k: np.array(t) + translation - origins[k] for k, t in receipt['selected_origins'].items()}
    canvas = Image.new('RGB', (1080, 495), '#f5f4f1')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), 'Two castle joins that pass a one-edge image check', fill='#18212b', font=font(27))
    canvas.paste(image.resize((656, 410), Image.Resampling.LANCZOS), (16, 54))
    records = []
    for i, pair in enumerate((['plane-39', 'plane-32'], ['plane-67', 'plane-54'])):
        j = next(j for j in receipt['joins'] if j['chart_ids'] == pair)
        assert j['kind'] == 'edge' and j['edges'] == 1 and not np.any(drift[pair[0]])
        points = np.array(j['image_edges'][0])
        shift = drift[pair[1]]
        moved = [k for k, d in drift.items() if np.array_equal(shift, d)]
        cells = sum(len(charts[k]['cells']) for k in moved)
        record = dict(charts=pair, image_error_px=j['error'], estimated_shift=j['shift'],
            native_shift=(origins[pair[1]] - origins[pair[0]]).tolist(), displaced_by=shift.tolist(),
            charts_at_displacement=moved, face_cells_at_displacement=cells, segment_px=points.tolist())
        records.append(record)
        mid = points.mean(0)
        x, y = mid * (656 / image.width) + [16, 54]
        color = ('#ffa726', '#54d8ed')[i]
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=color, width=3)
        draw.text((x + 10, y - 16), str(i+1), fill=color, font=font(19), stroke_width=1, stroke_fill='#111315')
        left, top = (np.rint(mid) - [90, 40]).astype(int)
        crop = image.crop((left, top, left+180, top+80)).resize((360, 160), Image.Resampling.NEAREST)
        line = (points - [left, top]) * 2
        ImageDraw.Draw(crop).line([tuple(q) for q in line], fill=color, width=3)
        y0 = 56 + i * 210
        canvas.paste(crop, (700, y0))
        draw.text((700, y0 + 163), f"{i+1}. {j['error']:.2f} px apart   ·   {cells:,} displaced face cells",
                  fill='#18212b', font=font(16))
        draw.text((700, y0 + 186), f"Group offset   ({shift[0]:+d}, {shift[1]:+d}, {shift[2]:+d}) blocks",
                  fill='#37495d', font=font(15))
    dest = ROOT / 'benchmark/reviews/expanded-wall-grids-01'
    canvas.save(dest / 'castle-false-joins.jpg', quality=92, optimize=True)
    save(dest / 'castle-false-joins.json', dict(id=name, source_sha256=case['rgb_sha256'], joins=records))


if __name__ == '__main__':
    main()
