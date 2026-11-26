from unittest import TestCase
from unittest.mock import patch

from engine import roof_height_consensus as height


class RoofHeightAgreementTests(TestCase):
    def select(self, rim_level, areas=(90, 10)):
        surfaces = [dict(part=str(i), area=area, accepted=True, margin=.25, plane_y=6)
                    for i, area in enumerate(areas)]
        rim = [dict(part='0', accepted=True, plane_y=rim_level)]
        with patch.object(height.roof_height_atlas, 'recover', return_value=(set(), dict(height_decisions=surfaces))), \
             patch.object(height.roof_height_rim, 'select', return_value=(rim, None, None, None, None)):
            return height.select(None, [], None, {}, set())['primary_height']

    def test_height_requires_agreement_on_dominant_patch(self):
        self.assertIsNone(self.select(5))
        self.assertIsNone(self.select(6, (55, 45)))
        self.assertEqual(self.select(6)['part'], '0')

    def test_revision_only_moves_a_resolved_surface_one_course(self):
        primary = dict(part='roof', plane_y=6)
        for levels in ({}, {'other':5}, {'roof':6}, {'roof':3}):
            self.assertEqual(height.one_course_override(primary, levels), {})
        self.assertEqual(height.one_course_override(primary, {'roof':5}), {'roof':6})
        self.assertEqual(height.one_course_override(None, {'roof':5}), {})
