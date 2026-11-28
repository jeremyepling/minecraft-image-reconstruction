import json
from pathlib import Path
import tempfile
import unittest

from benchmark.mobile_surface_evidence import pair_evidence, reuse_evidence
from benchmark.floating_stage_report import read


class SurfaceEvidenceTest(unittest.TestCase):
    def test_pair_preserves_different_masks_and_requires_both_pack_hashes(self):
        case = dict(id='frame', source_sha256='rgb', baseline=dict(pack_sha256='A'), candidate=dict(pack_sha256='B'))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            files = []
            for pack in ('A', 'B'):
                surface = dict(id='wall', mask=pack, exports={'Candidate': dict(cells=[])}, layers=[], decisions=[])
                package = dict(schema=2, id='frame', size=[2, 2], sourceSha256='rgb', surfaces=[surface], gaps=[],
                               provenance=dict(surfaceSha256=pack, countsSha256='counts', arms=dict(Candidate=dict(packSha256=pack))))
                path = folder / pack
                path.write_text(json.dumps(package), encoding='utf8')
                files.append(path)
            target = folder / 'paired.json'
            pair_evidence(*files, target, case, (2, 2))
            result = read(target)
            self.assertEqual(['A', 'B'], [s['mask'] for s in result['surfaces']])
            self.assertEqual(['Baseline', 'Candidate'], [s['arm'] for s in result['surfaces']])
            reuse_evidence(target, folder/'copy.json', case, (2, 2))
            self.assertEqual(target.read_bytes(), (folder/'copy.json').read_bytes())
            case['candidate']['pack_sha256'] = 'changed'
            with self.assertRaises(ValueError):
                reuse_evidence(target, folder/'copy.json', case, (2, 2))
            with self.assertRaises(ValueError):
                pair_evidence(*files, target, case, (2, 2))


if __name__ == '__main__':
    unittest.main()
