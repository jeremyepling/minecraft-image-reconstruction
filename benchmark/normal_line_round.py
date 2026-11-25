"""Score two frozen line extractors on the 48 previously reviewed exterior frames."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from benchmark.camera_round import ROOT, read, save
from benchmark.mlsd_boundary_score import run as evaluate_boundaries
from benchmark.mlsd_probe import probe, sha256 as file_hash


CASES = ROOT/'benchmark/reviews/surfaces-01/cases.json'
CASES_HASH = '342eb56c8f14f1c5313f35a1fabb153dd6c1e6c92497abde4ca27e4199def289'
WEIGHTS_HASH = 'e869f90601ffa263f21ac47e09ff0d65dc936d1aeb47126b8499d5504adfb100'
METHODS = ('mlsd-tiny-512-fp16', 'structural_lines.detect')
COUNT_FIELDS = ('segments', 'all_line_px', 'evaluated_line_px', 'line_px_within_tolerance',
                'truth_boundary_px', 'recovered_boundary_px')
REPRODUCE = ("& 'C:/Users/jerem/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' "
             "-m benchmark.normal_line_round --runtime-dir ../surface-round/runs/mlsd-probe/lib "
             "--checkpoint ../surface-round/runs/mlsd-probe/M-LSD_512_tiny_fp16.tflite "
             "--out runs/normal-lines-01")


def json_hash(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def enable_runtime(folder):
    if folder:
        sys.path.insert(0, str(folder))
    if version('ai-edge-litert') != '2.4.0':
        raise ValueError('This comparison uses ai-edge-litert 2.4.0.')


def evaluate(job):
    case, checkpoint, runtime = job
    enable_runtime(runtime)
    image = (CASES.parent/case['source']).resolve()
    fixture = image.parent.parent
    if file_hash(image) != case['sha256']:
        raise ValueError(f'Framebuffer hash mismatch for {image}')
    # Reuse the original one-frame detector and scorer verbatim, including
    # their order of prediction and first access to evaluator-only labels.
    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        tmp = Path(temp)
        probe(checkpoint, image, tmp/'mlsd')
        evaluate_boundaries(fixture, case['id'], tmp/'mlsd/segments.json', tmp/'scores.json')
        result = read(tmp/'scores.json')
    frame, = [f for f in read(fixture/'index.json')['frames'] if f['id'] == case['id']]
    if (fixture/frame['image']).resolve() != image:
        raise ValueError(f'Evaluator selected another source for {case["id"]}')
    info = {k: case[k] for k in ('id', 'floor', 'layout', 'cohort', 'palette')}
    return dict(**info, source=image.relative_to(ROOT).as_posix(), image_sha256=case['sha256'],
                labels_json_sha256=json_hash(read(fixture/frame['labels'])),
                pose_json_sha256=json_hash(read(fixture/frame['pose'])),
                roi_px=result['roi_px'], results=result['results'])


def totals(rows):
    group = {}
    for method in METHODS:
        counts = {key: sum(row['results'][method][key] for row in rows) for key in COUNT_FIELDS}
        counts['line_precision'] = counts['line_px_within_tolerance'] / counts['evaluated_line_px']
        counts['boundary_recall'] = counts['recovered_boundary_px'] / counts['truth_boundary_px']
        group[method] = counts
    return dict(frames=len(rows), results=group)


def summarize(rows):
    result = {'all': totals(rows)}
    for key in ('layout', 'floor'):
        for value in sorted({row[key] for row in rows}):
            result[value] = totals([row for row in rows if row[key] == value])
    return result


def report(path, summary):
    mlsd, structural = (summary['all']['results'][m] for m in METHODS)
    result = (f"M-LSD covered {mlsd['boundary_recall']:.1%} of visible boundary pixels with "
              f"{mlsd['segments']:,} segments. The structural detector covered "
              f"{structural['boundary_recall']:.1%} with {structural['segments']:,} segments. "
              f"Their pooled line precisions were {mlsd['line_precision']:.1%} and "
              f"{structural['line_precision']:.1%}, respectively.")
    lines = ['# Physical boundary coverage on 48 Minecraft frames', '',
             result, '',
             'Both detectors saw one Java 26.3 screenshot per prediction. This fixed comparison reuses '
             'the six layouts and their original splits from the first surface round. Counts pool pixels '
             'within each group. Floor and palette variants of a layout are repeated observations. '
             'The concrete group includes eight workshop and crossing views available only on that floor.', '',
             'Precision is the fraction of unique predicted line pixels within five native pixels of a '
             'physical boundary, evaluated inside the building and its five-pixel exterior band. Coverage '
             'is the fraction of true visible boundary pixels reached by those predictions. The scorer '
             'counts turns, depth changes and silhouettes, and excludes coplanar block seams.', '',
             '| Group | Frames | M-LSD segments | Precision | Coverage | Structural segments | Precision | Coverage |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, group in summary.items():
        row = [name, str(group['frames'])]
        for method in METHODS:
            m = group['results'][method]
            row += [f"{m['segments']:,}", f"{m['line_precision']:.1%}", f"{m['boundary_recall']:.1%}"]
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', 'The manifest contains pooled pixel counts and method settings. The frames file contains '
              'the source and evaluator hashes and both pixel denominators for every paired comparison.', '',
              'Run from the repository root using the previously downloaded checkpoint and private runtime.', '',
              '```powershell', REPRODUCE, '```', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, help='Optional local ai-edge-litert package directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    runtime = args.runtime_dir.resolve() if args.runtime_dir else None
    enable_runtime(runtime)
    cases = read(CASES)
    if json_hash(cases) != CASES_HASH or file_hash(args.checkpoint) != WEIGHTS_HASH:
        raise ValueError('Expected the frozen 48-frame cases file and pinned M-LSD Tiny FP16 checkpoint.')
    jobs = [(c, args.checkpoint.resolve(), runtime) for c in cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, jobs))
    previous = read(ROOT/'benchmark/reviews/local-models-01/measurements.json')['physical_boundaries']
    repeat, = [r for r in rows if r['image_sha256'] == previous['image_sha256']]
    if repeat['results'] != previous['results']:
        raise ValueError('The original matched concrete-frame measurement did not reproduce.')
    summary = summarize(rows)
    files = ('engine/structural_lines.py', 'benchmark/mlsd_probe.py', 'benchmark/mlsd_boundary_score.py',
             'benchmark/normal_line_round.py')
    code = {f: sha256((ROOT/f).read_text(encoding='utf-8').encode()).hexdigest() for f in files}
    manifest = dict(run='normal-lines-01', version='Java 26.3', round_type='fixed ancillary diagnostic',
                    methods_revision='29b50e6fbfe2162f1258e58e57cc03be0f3e9412', code_sha256_lf=code,
                    cases=CASES.relative_to(ROOT).as_posix(), cases_json_sha256=CASES_HASH,
                    fixture_splits={k: sum(c['cohort'] == k for c in cases) for k in sorted({c['cohort'] for c in cases})},
                    runtime={k: version(k) for k in ('numpy', 'pillow', 'ai-edge-litert')},
                    methods={METHODS[0]: dict(checkpoint=args.checkpoint.name, sha256=WEIGHTS_HASH,
                        source='https://huggingface.co/spaces/navervision/MLSD/blob/main/tflite_models/M-LSD_512_tiny_fp16.tflite',
                        input='512 square BOX, RGB + constant 1 as float32', score_greater_than=.10,
                        min_output_length_exclusive=20, cpu_threads=1),
                        METHODS[1]: dict(entry='engine.structural_lines.detect', settings='all defaults',
                                         passes=[[460, 1.75, .055], [820, 1.05, .017]],
                                         pass_settings=['max analysis dimension', 'Gaussian blur px', 'minimum length fraction'])},
                    truth=previous['truth'], tolerance_native_px=5,
                    roi='Foreground plus five native pixels. Unique raster pixels within each frame.',
                    original_result_reproduced=repeat['image_sha256'], summary=summary,
                    artifacts=['frames.json', 'README.md'], reproduce=REPRODUCE)
    args.out.mkdir(exist_ok=True, parents=True)
    save(args.out/'frames.json', rows)
    save(args.out/'manifest.json', manifest)
    report(args.out/'README.md', summary)
    print(f'Scored {len(rows)} frames in {args.out}')


if __name__ == '__main__':
    main()
