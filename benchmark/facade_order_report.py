"""Archive matching image signatures, individual scores and changed exports for order trials."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.texture_camera_eval import checked, digest, read, save


def collect(runs, out, folders):
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    fingerprints = {}
    bundle = out / 'comparisons.zip'
    with ZipFile(bundle, 'w', ZIP_DEFLATED) as archive:
        for name in folders:
            source = runs / name
            summary = read(source / 'summary.json')
            cases = checked(source / 'cases.json', summary['cases_sha256'])
            summaries[name] = summary
            prefix = name + '/'
            archive.write(cases, prefix + 'cases.json')
            archive.write(source / 'summary.json', prefix + 'summary.json')
            for row in read(cases):
                key = row.get('split', ''), row['id']
                for field in ('rgb_sha256', 'reference_hashes', 'input_hashes', 'control_pack_sha256'):
                    if field in row:
                        assert fingerprints.setdefault((key, field), row[field]) == row[field], (key, field)
                if row.get('changed') or (not key[0] and row.get('control_pack_sha256') != row['pack_sha256']):
                    entry = Path(*filter(None, key)) / 'reconstruction.zip'
                    archive.write(checked(source / entry, row['pack_sha256']), prefix + entry.as_posix())
    with ZipFile(bundle) as archive:
        assert archive.testzip() is None
    save(out / 'results.json', dict(control='1c2f77d', archive_sha256=digest(bundle), runs=summaries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('folders', nargs='+')
    args = parser.parse_args()
    collect(args.runs, args.out, args.folders)
