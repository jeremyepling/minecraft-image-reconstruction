"""Check the ordinary placer against archived final-pass trials, including the CLI."""
import argparse
from pathlib import Path

from benchmark.mask_blocks import decoded_pack
from benchmark.roof_span_verify import records
from benchmark.texture_camera_eval import checked, digest, read, save


def verify(runs, winner, transfer_winner, out):
    groups = []
    for sources, produced in [
        ([winner+'-dev', winner], 'production'),
        (['transfer-'+transfer_winner], 'transfer-production'),
        (['side-'+transfer_winner], 'side-production'),
        (['elevated-'+transfer_winner], 'elevated-production'),
        (['native-'+transfer_winner], 'native-production'),
    ]:
        expected = {}
        for name in sources:
            for k, row in records(runs/name).items():
                assert k not in expected
                expected[k] = (runs/name, row)
        actual = records(runs/produced)
        assert actual.keys() == expected.keys()
        for (split, key), (directory, row) in expected.items():
            other = actual[split, key]
            assert all(row[field] == other[field] for field in ('rgb_sha256','relative_geometry','reference_hashes'))
            a = checked(directory/split/key/'reconstruction.zip', row['pack_sha256'])
            b = checked(runs/produced/split/key/'reconstruction.zip', other['pack_sha256'])
            assert decoded_pack(a) == decoded_pack(b)
        groups.append(dict(production=produced, cases=len(actual), cases_sha256=digest(runs/produced/'cases.json')))
    cli = runs/'cli-canopy'
    manifest = read(cli/'build-manifest.json')
    key = 'shape-benchmark-26.3--train-canopy-06-left'
    cells = decoded_pack(cli/'reconstruction.zip')
    verify_row = records(runs/'production')['old', key]
    assert verify_row['rgb_sha256'] == manifest['image_sha256']
    assert cells == decoded_pack(runs/'production/old'/key/'reconstruction.zip')
    status = manifest['placement']
    assert len(cells) == status['placed_blocks'] == status['strict_blocks'] + status['added_blocks']
    assert status['final_vertical_blocks'] > 0
    save(out, dict(coordinates_and_materials_match=True, views=sum(g['cases'] for g in groups), groups=groups,
                   cli=dict(id=key,rgb_sha256=manifest['image_sha256'],pack_sha256=digest(cli/'reconstruction.zip'),
                            final_vertical_blocks=status['final_vertical_blocks'],total=len(cells))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--winner',required=True)
    parser.add_argument('--transfer-winner',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    verify(args.runs,args.winner,args.transfer_winner,args.out)
