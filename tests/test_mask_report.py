import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark.mask_replay import cache_for_run
from benchmark.mask_report import render_report
from benchmark.normal_fields import inference_signatures


def write_cache(folder, sha):
    np.savez(folder/(sha+'.npz'),normal=np.zeros((2,2,3)),depth=np.zeros((2,2)),
             signature=sha+':'+inference_signatures()[0])


class MaskReplayTest(unittest.TestCase):
    def test_complete_cache_is_read_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_cache(root,'a')
            result=cache_for_run({'a'},[root/'missing',root],root/'private')
            self.assertEqual((root,False),result)
            self.assertFalse((root/'private').exists())

    def test_incomplete_shared_caches_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a,b=root/'a',root/'b'
            a.mkdir()
            b.mkdir()
            write_cache(a,'one')
            write_cache(b,'two')
            private,fresh=cache_for_run({'one','two','three'},[a,b],root/'private')
            self.assertTrue(fresh)
            self.assertEqual(sorted(p.name for p in private.iterdir()),['one.npz','two.npz'])
            self.assertEqual(sorted(p.name for p in a.iterdir()),['one.npz'])
            self.assertFalse((b/'three.npz').exists())
            _,fresh=cache_for_run({'one','two'},[a,b],root/'complete')
            self.assertFalse(fresh)


class ReportTest(unittest.TestCase):
    def test_counts_errors_pixel_denominators_and_diagnostic_are_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            keys=('eligible proposed matched eligible_pixels correct_pixels wrong_plane_pixels unclaimed_pixels '
                  'background_claimed opening_pixels opening_filled gaps gaps_half_filled real_connections '
                  'proposed_connections correct_connections').split()
            score={k:0 for k in keys}
            for metric in ('edge','corner'):
                for suffix in ('true','proposed','distance_count','distance_sum','true_near','proposed_near'):
                    score[metric+'_'+suffix]=0
            score.update(eligible=3,matched=2,proposed=5,eligible_pixels=7,correct_pixels=4,
                         wrong_plane_pixels=1,unclaimed_pixels=2)
            scores={m:score for m in ('current-b','mask-first','scalelsd','deeplsd')}
            count=dict(serialized=1,visible_truth=5,correct_visible=1,extra=0,missed=4)
            entry=dict(score=count,diagnostic=count,regions={})
            blocks=dict(baseline=entry,selected=entry,selected_with_B_charts=dict(score=count,regions={}))
            row=dict(id='view<&',sha256='a',layout='wing',floor='smooth_stone',cohort='development',
                     source='source with spaces.png',predictions='p.json',scores=scores,blocks=blocks,
                     timing_seconds=dict(scalelsd_inference_recorded=1.25))
            for path,payload in [('cases.json',[row,row]),('scores.json',dict(all=scores)),
                                 ('selection.json',dict(selected='mask-first')),('blocks.json',dict(all=blocks))]:
                (root/path).write_text(json.dumps(payload),encoding='utf8')
            self.assertEqual(render_report(root),'report.html')
            html=(root/'report.html').read_text(encoding='utf8')
            for term in ('2 / 3 (66.7%)','4 / 7 (57.1%)','1 / 7 (14.3%)','3 / 5 (60.0%)',
                         'view&lt;&amp;', 'source%20with%20spaces.png','Fixed depth and scale check', '1 / 5 (20.0%)'):
                self.assertIn(term,html)
            self.assertNotIn('2.500 s',html)
            self.assertEqual(html.count('Runtime for this picture'),2)
