"""Evaluate roof heights against the published strict atlas-strip default at 54c9fee."""
from functools import lru_cache
from hashlib import sha256
import json
from zipfile import ZipFile

from benchmark import framed_depth_eval as framed, roof_span_eval as span
from benchmark.texture_camera_eval import ROOT, checked, read


REVIEW = ROOT / 'benchmark/reviews/roof-span-01'
CHANGES = REVIEW / 'comparisons.zip'
CONTROL = '54c9fee'
ENGINE = '016bb26'


@lru_cache(maxsize=1)
def changes():
    manifest = read(REVIEW/'results.json')
    return ZipFile(checked(CHANGES, manifest['archive_sha256']))


def previous_rows(prefix,bundle=None):
    if bundle is None:
        bundle=changes()
    encoded = bundle.read(f'{prefix}/cases.json')
    meta = json.loads(bundle.read(f'{prefix}/summary.json'))
    assert sha256(encoded).hexdigest() == meta['cases_sha256']
    return json.loads(encoded)


def cases(splits):
    rows = previous_rows('conservative')
    index = {(r['split'],r['id']):r for r in rows}
    assert len(index) == len(rows) == sum(framed.SIZES.values())
    for split,frame,old in span.cases(splits):
        row = index[split,frame['id']]
        assert row['control_pack_sha256'] == old['pack_sha256']
        assert row['rgb_sha256'] == old['rgb_sha256']
        assert row['input_hashes'] == old['input_hashes'] and row['reference_hashes'] == frame['files']
        yield split, frame, {**row, **row['input_hashes']}


def score(job):
    split,frame,prior,_,predictions = job
    if framed.visible_depth_score.predict is None or predictions is not None:
        framed.photo(split,frame,prior)
    if prior['changed']:
        bundle, folder = changes(), f'conservative/{split}'
    else:
        assert prior['pack_sha256'] == prior['control_pack_sha256']
        bundle, folder = framed.archive(span.PACKS), split
    row = framed.score_archive(job, bundle, folder)
    if 'roof' in prior:
        assert json.loads(json.dumps(row['previous_roof'])) == prior['roof'], frame['id']
    return row


if __name__=='__main__':
    args = framed.parser(__doc__).parse_args()
    span.compare(args, case_source=cases, scorer=score, control=CONTROL,
                 default_engine=ENGINE, additional_sources=(CHANGES,))
