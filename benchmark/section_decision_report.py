"""Package branch comparisons and readable six-family fitting outcomes."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.texture_camera_eval import ROOT, read, save

OUT=ROOT/'benchmark/reviews/plane-section-01'
RUN=ROOT/'runs'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    training=RUN/'plane-section'
    hand=read(training/'train.json')
    held=read(training/'held-applied.json')
    families=sorted({row['family'] for row in hand})
    def aggregate(items):
        return dict(sum((Counter(m) for m in items),Counter()))
    save(OUT/'development.json',dict(
        hand={variant:{family:aggregate(row['rules'][variant]['relative_geometry'] for row in hand
                           if row['family']==family) for family in families} for variant in hand[0]['rules']},
        held_family_complete={family:aggregate(row['relative_geometry'] for row in held if row['family']==family)
                              for family in families},selection=read(training/'loo.json')))
    for name in ('strict','learned','trim','deeper'):
        path=RUN/('sealed-'+name)
        dest=OUT/name
        dest.mkdir(exist_ok=True)
        summary=read(path/'summary.json')
        rows=read(path/'cases.json')
        assert sha256((path/'cases.json').read_bytes()).hexdigest()==summary['cases_sha256']
        with ZipFile(dest/'builds.zip','w',ZIP_DEFLATED) as archive:
            archive.write(path/'cases.json','cases.json')
            for row in rows:
                folder=Path(row['split'])/row['id']
                for file in ('reconstruction.zip','cells.json'):
                    item=path/folder/file
                    if file.endswith('zip'):
                        assert sha256(item.read_bytes()).hexdigest()==row['pack_sha256']
                    archive.write(item,(folder/file).as_posix())
        save(dest/'summary.json',summary)


if __name__=='__main__':
    main()
