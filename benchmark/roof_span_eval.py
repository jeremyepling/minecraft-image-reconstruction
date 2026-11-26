"""Score image-derived roof spans against the shipped 99655be placement packs."""
from benchmark import framed_depth_eval as framed
from benchmark.texture_camera_eval import ROOT, STABLE, TRAINING_RELEASE, checked, digest, read, save
from benchmark.visible_recovery_report import summarize


BASELINE = ROOT / 'benchmark/reviews/framed-depth-01'
PACKS = BASELINE / 'scores.zip'


def cases(splits):
    proof = read(BASELINE / 'shape-check.json')
    checked(PACKS, proof['archive_sha256'])
    summary = read(BASELINE / 'summary.json')
    assert summary['revision'].startswith('0e0c229')
    assert summary['cases_sha256'] == proof['cases_sha256']
    rows = read(checked(BASELINE / 'cases.json', proof['cases_sha256']))
    indexed = {(r['split'], r['id']): r for r in rows}
    assert len(indexed) == len(rows) == sum(framed.SIZES.values())
    for split, frame, previous in framed.cases(splits):
        row = indexed[split, frame['id']]
        assert row['rgb_sha256'] == previous['rgb_sha256']
        assert row['reference_hashes'] == frame['files']
        assert all(previous[k] == v for k, v in row['input_hashes'].items())
        yield split, frame, {**row, **row['input_hashes']}


def score(job):
    split, frame, prior, _, predictions = job
    if framed.visible_depth_score.predict is None or predictions is not None:
        # A baseline/replay must fingerprint the same image-only inputs as a live candidate.
        framed.photo(split, frame, prior)
    row = framed.score_archive(job, framed.archive(PACKS), split)
    if 'roof' in prior:
        assert row['previous_roof'] == prior['roof'], frame['id']
    return row


def compare(args):
    sources = (TRAINING_RELEASE, framed.ARCHIVE, framed.NEW_ARCHIVE, PACKS,
               framed.REVIEW / 'whole-structure-framing-01/evaluation-only.json')
    summary, groups = framed.compare(args, case_source=cases, scorer=score,
                                     control='99655be', default_engine='0e0c229', source_files=sources)
    summary['camera_registered'] = {
        group: summarize([{'relative_geometry': row['metrics']} for row in rows])
        for group, rows in groups.items()
    }
    atlas = STABLE / 'runs/26.3/atlas'
    summary['calibrated_atlas'] = {name: digest(atlas / name) for name in ('atlas.json', 'atlas.png')}
    summary['roof_probes_sha256'] = digest(framed.openings.PROBES)
    save(args.out / 'summary.json', summary)


if __name__ == '__main__':
    compare(framed.parser(__doc__).parse_args())
