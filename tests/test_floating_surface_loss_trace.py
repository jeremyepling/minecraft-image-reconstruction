import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmark.floating_stage_report import hash_file
from benchmark.floating_surface_loss_trace import trace_case
from engine.export import JAVA_26_3_FORMAT, write_pack


STONE = 'minecraft:stone'


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


class SurfaceLossTraceTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.frame = Path(temp.name)
        (self.frame / 'rgb.png').write_bytes(b'source fingerprint')
        save(self.frame / 'labels.json', dict(
            blocks=[dict(id=1, block_id=STONE)],
            faces=[dict(block=1, xyz_world=[x, 3, 0], visible_pixels=10) for x in [8, 9]],
        ))
        self.case = dict(id='frame', source_sha256=hash_file(self.frame / 'rgb.png'),
                         reference_sha256={'labels.json': hash_file(self.frame / 'labels.json')},
                         paired=dict(gain=1, loss=1, add=0, remove=0, exact_gain=1, exact_loss=1),
                         component=dict(dropped=['wall']))
        self.arm_proofs = {}
        self.package = dict(schema=2, id='frame', sourceSha256=self.case['source_sha256'],
                            provenance=dict(arms=self.arm_proofs), surfaces=[])

    def arm(self, label, xyz, yaw, shift, world, chart, *, unresolved=False, local_grid=False):
        folder = self.frame / label
        folder.mkdir()
        pack = folder / 'reconstruction.zip'
        write_pack(pack, {xyz: STONE.removeprefix('minecraft:')} if xyz else {}, JAVA_26_3_FORMAT)
        digest = hash_file(pack)
        fixed = None if unresolved else dict(yaw_quarters=yaw, translation=shift)
        proof = dict(id='frame', source_sha256=self.case['source_sha256'],
                     reference_sha256=self.case['reference_sha256'], pack_sha256=digest)
        if local_grid:
            transform = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            score = dict(proof, registration=dict(transform=transform if xyz and not unresolved else None),
                         translation=shift)
            save(folder / 'case.json', dict(id='frame', score=score))
            (folder / 'aligned-placements.json.gz').write_bytes(gzip.compress(json.dumps(
                [[*world, STONE]] if world else []).encode()))
        else:
            save(folder / 'score.json', dict(proof, score=dict(fixed=fixed)))
            save(folder / 'placements.json', dict(local=[[xyz, STONE]], aligned=[[world, STONE]]))
        self.case[label.lower()] = dict(pack_sha256=digest)
        self.arm_proofs[label] = dict(packSha256=digest)
        self.package['surfaces'].append(dict(id=chart, exports={label: dict(cells=[
            dict(xyz=xyz, block=STONE, chart=chart, quad=[[1, 1], [3, 1], [3, 5], [1, 5]])
        ] if xyz else [])}))
        return folder

    def test_dropped_surface_counts_use_each_pack_registration(self):
        old = self.arm('Baseline', (0, 0, 2), 1, [10, 3, 0], (8, 3, 0), 'wall~p1')
        new = self.arm('Candidate', (0, 0, 2), 3, [7, 3, 0], (9, 3, 0), 'roof')
        traced = trace_case(self.case, self.package, self.frame, old, new)
        self.assertEqual('verified', traced['status'])
        self.assertEqual(1, traced['droppedLosses'])
        self.assertEqual(1, traced['surfaces']['roof']['gained'])
        wall = traced['surfaces']['wall~p1']
        self.assertEqual(1, wall['lost'])
        self.assertEqual(dict(kind='lost', chart='wall~p1', source=[2.0, 3.0], xyz=(8, 3, 0)), wall['markers'][0])
        self.package['surfaces'][0]['exports']['Baseline']['cells'][0]['block'] = 'minecraft:dirt'
        with self.assertRaisesRegex(ValueError, 'ownership differs'):
            trace_case(self.case, self.package, self.frame, old, new)

    def test_local_grid_empty_arm_is_measurable(self):
        old = self.arm('Baseline', None, None, [0, 0, 0], None, 'Baseline / wall', local_grid=True)
        new = self.arm('Candidate', (0, 0, 0), None, [9, 3, 0], (9, 3, 0), 'Candidate / roof', local_grid=True)
        self.case.update(paired=dict(gain=1, loss=0, add=0, remove=0, exact_gain=1, exact_loss=0), component=None)
        traced = trace_case(self.case, self.package, self.frame, old, new)
        self.assertEqual('verified', traced['status'])
        self.assertEqual(0, traced['droppedLosses'])
        self.assertEqual(1, traced['surfaces']['Candidate / roof']['gained'])

    def test_unregistered_nonempty_arm_reports_unavailability(self):
        old = self.arm('Baseline', (0, 0, 2), 1, [10, 3, 0], (8, 3, 0), 'wall')
        new = self.arm('Candidate', (0, 0, 2), None, [0, 0, 0], None, 'roof',
                       unresolved=True, local_grid=True)
        traced = trace_case(self.case, self.package, self.frame, old, new)
        self.assertEqual('unavailable', traced['status'])
        self.assertIn('Candidate export has no saved registration', traced['reason'])
        self.assertFalse(traced['surfaces'])


if __name__ == '__main__':
    unittest.main()
