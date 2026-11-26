"""Publish a compact, linked index of the stage maps and their evaluation records."""

def write_report(out, ranking, traces):
    rows = [
        '# Ten lowest-scoring structures',
        '',
        'Open the [overlay viewer](viewer.html) from a local checkout to switch between the source photograph, surfaces, measured grid, reconstruction, visible reference and placement errors at full resolution. The opacity slider reveals the source pixels under each overlay.',
        '',
        'Each of the 104 fixed, whole-structure views contributes its visible-block counts. For each distinct layout, we pooled true positives, extras and misses over all of its views and ranked by relative geometry F1. The 40 layout identities come from full occupancy after accounting for translations, rotations, horizontal reflections and palette changes. All parts of a predicted build receive one common translation during scoring. Each column shows its layout’s lowest-scoring view. [Zoomable stage map](comparison.jpg) · [All ten layouts and individual frame scores](ranking.json) · [Measured stages and source hashes](stage-traces.json)',
        '',
        '| Rank | Layout and stage sheet | Views | Layout F1 | Shown view F1 | Correct / visible | Extras |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |'
    ]
    for r in ranking['layouts']:
        v = r['representative']
        rows.append(f"| {r['rank']} | [{r['name']}]({r['sheet']}) | {r['frames']} | {r['f1']:.2%} | {v['f1']:.2%} | {v['correct']} / {v['target']} | {v['extras']} |")
    rows.extend([
        '',
        '## Reading a column',
        '',
        'The first row shows the entire source photograph. The detected-surface overlay traces every returned region in orange, with one surface highlighted in cyan. The crop, straightened texture, measured grid and atlas tile follow that surface. The selection favors a complete tile, then a usable chart with a located axis or measured dimensions. Among comparable charts it favors overlap with missed visible blocks, prioritizing the roof on canopy probes. Evaluation labels guide this illustration after scoring.',
        '',
        'The Surface Finder count is a display diagnostic. It compares masks with same-plane, edge-connected visible reference surfaces at native resolution, greedily pairing the highest IoU first down to 0.50. A partial outline can contribute pixels to the build even if it falls below the threshold. The Block Counter uses its frozen per-surface chart, dimensions and grid origins. The source-grid overlay projects each measured axis through its chart and clips every line to the detected mask; two located origins are needed to place a complete tile. Question marks indicate a missing estimate.',
        '',
        'The material row replays the calibrated vanilla atlas matcher on a complete tile. If the frozen chart lacked a grid origin, a separate atlas diagnostic tries to register it at the already measured spacing. Each panel identifies which source located the tile. The original measurement still supplies both grid rows and the evaluated coordinates. The comparison material comes from the labeled face at the center of the tile. Unresolved panels indicate insufficient evidence. The stage trace records the materials encoded in the evaluation pack.',
        '',
        'The final rows place the decoded export on the source photo in teal, draw only the visible reference blocks in purple, then compare both after the shared scoring shift. Green is correct, red is extra and blue is missing. For overlap, the image shows the frontmost colored face. The counts include all scored coordinates. The overhead panel shows the highest scored block in each X/Z column. Each source overlay has a native-resolution copy in the overlays folder.',
        '',
        '## Stage counts for the ten shown views',
        '',
        '| Layout | Surface matches / visible | Detected | Straightened | Both dimensions measured | Both grid origins located | Sampled tile guess / recorded center |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |'
    ])
    for r in traces:
        c = r['counts']
        sample = r['material_replay']
        def short(s):
            return s.removeprefix('minecraft:') if s else 'unresolved'
        material = f"{short(sample.get('measured_block_id'))} / {short(sample.get('recorded_id_at_tile_center'))}"
        rows.append(f"| {r['layout']} | {c['matched_at_iou_half']} / {c['visible']} | {c['predicted']} | {c['charts']} | {c['two_dimension_counts']} | {c['two_grid_origins']} | {material} |")
    rows.extend(['', '## Ten lowest individual views', '',
                 'A layout can appear more than once in this separate list.',
                 '', '| View | F1 | Correct / visible | Extra |', '| --- | ---: | ---: | ---: |'])
    for v in ranking['lowest_views']:
        rows.append(f"| {v['id']} | {v['f1']:.2%} | {v['correct']} / {v['target']} | {v['extras']} |")
    rows.extend(['','## Reproduce','','Run from the repository with the scored fixture data and calibrated atlas available. Supply the winning score directory, including the exported packs and decoded coordinates.','',
                 '```powershell','python -m benchmark.roof_span_worst --run runs/conservative --out benchmark/reviews/roof-span-01/worst','```',''])
    (out/'README.md').write_text('\n'.join(rows),encoding='utf8')
