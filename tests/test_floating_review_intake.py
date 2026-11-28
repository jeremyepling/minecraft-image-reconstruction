import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from benchmark.floating_review_intake import connect, entries, ingest, private_history, record_outcome


class MobileReviewIntakeTest(unittest.TestCase):
    def test_download_export_roundtrip_carry_forward_and_measured_outcome(self):
        manifest_path = Path(__file__).resolve().parents[1] / 'benchmark/reviews/mobile-review-runs/floating-v1-guarded-refit/manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        case = manifest['cases'][0]
        gate = dict(id='arcade-coverage', caseId=case['id'], sourceSha256=case['sourceSha256'], type='preference',
                    theme='coverage', stage='Component selection', prompt='Which treatment?', cue='Inspect the seam.',
                    firstPass='The offset trades coverage for alignment.', checked='Both exports were decoded.',
                    choices=[dict(answer='B', experiment='Test coverage'), dict(answer='A', experiment='Test alignment')])
        note = dict(caseId=case['id'], sourceSha256=case['sourceSha256'], text='Private observation.',
                    tags=['Covers more visible area'], preference='B', actionId=gate['id'], questionType='preference',
                    answer='B', imageView='source')
        export = dict(schema=1, runId=manifest['id'], comparisonSha256=manifest['comparisonSha256'], feedback=[note])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            database = folder / 'history.sqlite3'
            downloads = folder / 'downloads'
            downloads.mkdir()
            for suffix in ('', ' (1)'):
                (downloads / (manifest['id'] + '-feedback' + suffix + '.json')).write_text(json.dumps(export))
            (downloads / 'private.json').write_text('not a review file')
            state = private_history(manifest, database=database, downloads=downloads, questions=[gate])
            self.assertEqual(len(state['entries']), 1)
            feedback = state['entries'][0]
            self.assertEqual((feedback['status'], feedback['answer'], feedback['tags']), ('answered', 'B', note['tags']))
            self.assertEqual(state['questions'], [gate])
            with closing(connect(database)) as db:
                unsolicited = dict(caseId=case['id'], preference='B', text='Volunteered note')
                self.assertEqual(ingest(db, dict(export, feedback=[unsolicited]), manifest), 1)
                self.assertEqual(entries(db)[1]['status'], 'observed')
                with self.assertRaises(ValueError):
                    ingest(db, dict(export, feedback=[dict(unsolicited, text='rollback'), dict(note, answer='unknown')]), manifest)
                self.assertEqual(len(entries(db)), 2)
                record_outcome(db, feedback['feedbackId'], True, 'Private result', 'Coverage trial', manifest)
            later = private_history(dict(manifest, id='later-run'), database=database, downloads=downloads)
            self.assertEqual(later['questions'], [gate])
            tested = later['entries'][0]
            self.assertEqual((tested['status'], tested['changedExperiment'], tested['measuredEffect']), ('tested', True, True))
            baseline, candidate = manifest['methods']
            self.assertEqual(tested['measuredDelta']['correct'], candidate['scores']['geometry_strict']['correct'] -
                             baseline['scores']['geometry_strict']['correct'])

    def test_criterion_only_annotation_and_surface_gated_answer(self):
        manifest = dict(id='round',comparisonSha256='sha',cases=[dict(id='arcade',sourceSha256='source')])
        gate = dict(id='wall-grid',caseId='arcade',sourceSha256='source',surfaceId='Baseline / mask-2',
                    type='visible-ambiguity',theme='grid',stage='Block Counter',prompt='Which seam?',cue='Inspect the roof.',
                    firstPass='Earlier line is plausible.',checked='Compared both overlays.',
                    choices=[dict(answer='Near',experiment='Near test'),dict(answer='Far',experiment='Far test')])
        observation = dict(caseId='arcade', surfaceId=gate['surfaceId'], gridAttempt='Refit', gridOccurrence=1,
                           assessments={'coverage':{'answer':'no','note':'Misses rim.'},
                                        'blockSize':{'answer':'yes','note':''},
                                        'alignment':{'answer':'unclear','note':'Occluded seam.'}})
        answer = dict(caseId='arcade',surfaceId=gate['surfaceId'],actionId=gate['id'],
                      questionType=gate['type'],answer='Near')
        export = dict(schema=1,runId='round',comparisonSha256='sha')
        with tempfile.TemporaryDirectory() as dir_name:
            folder = Path(dir_name)
            database = folder/'history.sqlite3'
            history = private_history(manifest, database=database, downloads=folder, questions=[gate])
            self.assertEqual(history['questions'][0]['surfaceId'], gate['surfaceId'])
            with closing(connect(database)) as db:
                self.assertEqual(ingest(db, dict(export,feedback=[observation,answer]), manifest), 2)
                observed, answered = entries(db)
                self.assertEqual(observed['assessments'], observation['assessments'])
                self.assertEqual(observed['gridOccurrence'], 1)
                self.assertEqual(answered['status'], 'answered')
                with self.assertRaises(ValueError):
                    ingest(db, dict(export,feedback=[dict(answer,surfaceId='mask-2')]), manifest)
