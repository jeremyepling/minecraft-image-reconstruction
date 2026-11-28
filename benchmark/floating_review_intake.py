"""Ingest mobile exports and record experiment outcomes in the repository's private Git directory."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys


def private_database():
    root = Path(__file__).resolve().parents[1]
    git_dir = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True).strip()
    return Path(git_dir) / 'floating-review-history.sqlite3'


def connect(database):
    db = sqlite3.connect(database)
    db.execute('CREATE TABLE IF NOT EXISTS feedback (id TEXT PRIMARY KEY, entry TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS outcomes '
               '(feedback_id TEXT PRIMARY KEY REFERENCES feedback(id), changed INTEGER, result TEXT, experiment TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS questions '
               '(action TEXT, case_id TEXT, source TEXT, gate TEXT NOT NULL, PRIMARY KEY (action, case_id, source))')
    db.execute('CREATE TABLE IF NOT EXISTS measurements '
               '(feedback_id TEXT PRIMARY KEY REFERENCES feedback(id), receipt TEXT NOT NULL)')
    db.execute('PRAGMA foreign_keys = ON')
    return db


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def register(db, manifest, questions):
    cases = {case['id']: case['sourceSha256'] for case in manifest['cases']}
    with db:
        for supplied in questions:
            fields = ('id', 'caseId', 'sourceSha256', 'theme', 'stage', 'type', 'prompt', 'cue', 'firstPass', 'checked')
            gate = {field: supplied.get(field) for field in fields}
            if not all(isinstance(value, str) and value.strip() for value in gate.values()):
                raise ValueError('Question requires a case, source digest, cue, first pass and completed checks.')
            if (not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}', gate['id']) or
                    cases.get(gate['caseId']) != gate['sourceSha256'] or
                    gate['type'] not in ('preference', 'visible-ambiguity')):
                raise ValueError('Question identity, source or type does not match this comparison.')
            choices = supplied.get('choices')
            if not isinstance(choices, list) or len(choices) < 2:
                raise ValueError('Supply at least two choices.')
            clean = []
            for choice in choices:
                if not all(isinstance(choice.get(k), str) and choice[k].strip() for k in ('answer', 'experiment')):
                    raise ValueError('Each answer must select an experiment.')
                clean.append({k: choice[k].strip() for k in ('answer', 'experiment')})
            if len({c['answer'] for c in clean}) != len(clean) or len({c['experiment'] for c in clean}) < 2:
                raise ValueError('Supply distinct answers that affect the experiment.')
            gate['choices'] = clean
            if 'surfaceId' in supplied:
                surface = supplied['surfaceId']
                if not isinstance(surface, str) or not re.fullmatch(r'(?:Baseline / |Candidate / )?[A-Za-z0-9_.~-]{1,128}', surface):
                    raise ValueError('Invalid surface ID.')
                gate['surfaceId'] = surface
            if 'priority' in supplied:
                gate['priority'] = int(supplied['priority'])
            db.execute('INSERT OR REPLACE INTO questions VALUES (?, ?, ?, ?)',
                       (gate['id'], gate['caseId'], gate['sourceSha256'], json.dumps(gate)))


def check_answer(db, manifest, entry):
    if 'answer' not in entry and 'actionId' not in entry:
        return
    action_id = entry.get('actionId')
    if not isinstance(action_id, str) or not isinstance(entry.get('answer', ''), str):
        raise ValueError('A formal answer requires a question ID and a text choice.')
    case_id = entry['caseId']
    saved = db.execute('SELECT gate FROM questions WHERE action=? AND case_id=? AND source=?',
                       (action_id, case_id, entry['sourceSha256'])).fetchone()
    question = json.loads(saved[0]) if saved else None
    for action in (manifest.get('analysis') or {}).get('actions', []):
        if action['id'] == action_id and case_id in action['caseIds'] and action.get('question'):
            question = action['question']
            break
    if (question is None or entry.get('questionType') != question['type'] or
            (question.get('surfaceId') and question['surfaceId'] != entry.get('surfaceId')) or
            (entry.get('answer') and entry['answer'] not in {choice['answer'] for choice in question['choices']})):
        raise ValueError('Answer does not match a recorded question, case and choice.')


def ingest(db, export, manifest):
    if (export.get('schema') != 1 or export.get('runId') != manifest['id'] or
            export.get('comparisonSha256') != manifest['comparisonSha256']):
        raise ValueError('Export is for a different run or comparison.')
    cases = {case['id']: case for case in manifest['cases']}
    if not isinstance(export.get('feedback'), list):
        raise ValueError('Expected a feedback array.')
    added = 0
    with db:
        for note in export['feedback']:
            if not isinstance(note, dict) or note.get('caseId') not in cases:
                raise ValueError('Unknown feedback case.')
            case = cases[note['caseId']]
            if note.get('sourceSha256', case['sourceSha256']) != case['sourceSha256']:
                raise ValueError('Feedback source differs from the measured case.')
            entry = {key: value for key, value in note.items() if key in {
                'actionId', 'questionType', 'answer', 'caseId', 'surfaceId', 'text', 'tags', 'spot',
                'imageView', 'chartId', 'gridAttempt', 'gridOccurrence', 'assessments', 'preference', 'updatedAt'}}
            entry.update(runId=manifest['id'], comparisonSha256=manifest['comparisonSha256'],
                         sourceSha256=case['sourceSha256'])
            if 'assessments' in entry:
                assessments = entry['assessments']
                if not isinstance(assessments, dict):
                    raise ValueError('Expected a map of criterion judgments.')
                for key, criterion in assessments.items():
                    if (key not in ('coverage', 'alignment', 'blockSize') or not isinstance(criterion, dict) or
                            set(criterion) - {'answer', 'note'} or
                            criterion.get('answer', '') not in ('', 'yes', 'no', 'unclear') or
                            not isinstance(criterion.get('note', ''), str)):
                        raise ValueError('Invalid criterion judgment.')
                entry['assessments'] = {key: item for key, item in assessments.items()
                                        if item.get('answer') or item.get('note', '').strip()}
            check_answer(db, manifest, entry)
            if not any(entry.get(key) for key in ('answer', 'text', 'tags', 'spot', 'preference', 'assessments')):
                continue
            content = json.dumps(entry, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            feedback_id = hashlib.sha256(content.encode()).hexdigest()
            added += db.execute('INSERT OR IGNORE INTO feedback VALUES (?, ?)', (feedback_id, content)).rowcount
    return added


def entries(db):
    rows = db.execute('SELECT id, entry, changed, result, experiment, receipt FROM feedback '
                      'LEFT JOIN outcomes ON outcomes.feedback_id = id '
                      'LEFT JOIN measurements ON measurements.feedback_id = id ORDER BY feedback.rowid')
    records = []
    for key, content, changed, result, experiment, receipt in rows:
        entry = json.loads(content)
        status = 'answered' if entry.get('actionId') and entry.get('answer') else 'observed'
        entry.update(feedbackId=key, status=status)
        if changed is not None:
            entry.update(status='tested', changedExperiment=bool(changed), outcome=result, experiment=experiment)
        if receipt:
            entry.update(json.loads(receipt))
        records.append(entry)
    return records


def discover(downloads, run):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', run):
        raise ValueError('Invalid run ID.')
    pattern = re.compile(re.escape(run) + r'-feedback(?: \(\d+\))?\.json', re.IGNORECASE)
    return sorted(p for p in downloads.glob(run + '-feedback*.json') if pattern.fullmatch(p.name)
                  and p.is_file() and not p.is_symlink())


def private_history(manifest, *, database=None, downloads=None, questions=None):
    """Return local records after importing matching exports. Pass only to the analysis step."""
    with closing(connect(database or private_database())) as db:
        register(db, manifest, questions or [])
        for path in discover(Path(downloads) if downloads is not None else Path.home() / 'Downloads', manifest['id']):
            try:
                ingest(db, read(path), manifest)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                print(f'Skipped {path.name} ({type(exc).__name__}).', file=sys.stderr)
        return {'entries': entries(db), 'questions': [json.loads(row[0]) for row in db.execute('SELECT gate FROM questions ORDER BY action, case_id, source')]}


def record_outcome(db, feedback_id, changed, result, experiment, score_manifest=None):
    if not db.execute('SELECT 1 FROM feedback WHERE id = ?', (feedback_id,)).fetchone():
        raise ValueError('Unknown feedback ID. Use list to see imported records.')
    receipt = None
    if score_manifest is not None:
        base, candidate = [item['scores'] for item in score_manifest['methods']]
        delta = {k: candidate['geometry_strict'][k] - base['geometry_strict'][k] for k in ('correct', 'missed', 'extras')}
        for label, key in [('geometry_strict', 'geometry_strict'), ('material_strict', 'scores_strict')]:
            delta[label] = candidate[key]['f1_percent'] - base[key]['f1_percent']
        receipt = dict(measuredEffect=True, measuredDelta=delta, measuredComparison=score_manifest['comparisonSha256'],
                       measuredCohort=score_manifest['cohort'], measuredViews=score_manifest['views'])
    with db:
        db.execute('INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?)', (feedback_id, changed, result, experiment))
        db.execute('DELETE FROM measurements WHERE feedback_id=?', (feedback_id,))
        if receipt:
            db.execute('INSERT INTO measurements VALUES (?, ?)', (feedback_id, json.dumps(receipt)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli = parser.add_subparsers(dest='command', required=True)
    intake = cli.add_parser('ingest', help='Import a matching export or discover run-specific download files.')
    intake.add_argument('manifest', type=Path)
    source = intake.add_mutually_exclusive_group()
    source.add_argument('--file', type=Path)
    source.add_argument('--downloads', type=Path)
    cli.add_parser('list', help='List private feedback IDs and linkage without printing notes.')
    outcome = cli.add_parser('outcome', help='Attach the result of an experiment to an imported annotation.')
    outcome.add_argument('feedback_id')
    outcome.add_argument('--experiment', required=True)
    outcome.add_argument('--result', required=True)
    outcome.add_argument('--score-manifest', type=Path, help='Subsequent paired mobile score manifest for this test.')
    outcome.add_argument('--changed-experiment', choices=['yes', 'no'], required=True)
    args = parser.parse_args()
    with closing(connect(private_database())) as db:
        if args.command == 'ingest':
            manifest = read(args.manifest)
            if args.file:
                print(f'{ingest(db, read(args.file), manifest)} new annotations.')
            else:
                print(f"{len(private_history(manifest, downloads=args.downloads)['entries'])} annotations in private history.")
        elif args.command == 'list':
            for entry in entries(db):
                print(entry['feedbackId'], entry['runId'], entry['caseId'], entry.get('actionId', 'observation'), entry['status'])
        else:
            record_outcome(db, args.feedback_id, args.changed_experiment == 'yes', args.result, args.experiment,
                           read(args.score_manifest) if args.score_manifest else None)
            print('Experiment outcome saved privately.')


if __name__ == '__main__':
    main()
