"""Compare a frozen entrance challenger with the incumbent across labeled layouts."""
import argparse
import json
from pathlib import Path


def decide(candidate, validation, confirmation):
    incumbent = candidate['incumbent']
    challenger = candidate['selected']
    if confirmation['selected'] != challenger:
        raise ValueError('Confirmation was scored with a different challenger')
    comparisons = []
    for report in (validation, confirmation):
        baseline = report['metrics'][incumbent]
        proposed = report['metrics'][challenger]
        if not baseline['real'] or baseline['real'] != proposed['real']:
            raise ValueError('Both methods must be evaluated on the same positive openings')
        comparisons.append({key: proposed[key] - baseline[key]
                            for key in ('exact', 'wrong_depth', 'false_recess')})
    safe = all(d['exact'] >= 0 and d['wrong_depth'] <= 0 and d['false_recess'] <= 0
               for d in comparisons)
    promoted = safe and any(d['exact'] > 0 for d in comparisons)
    return dict(selected=challenger if promoted else incumbent,
                review_extras=not promoted, challenger=challenger,
                promoted=promoted, rule=candidate['promotion_rule'],
                validation_delta=comparisons[0], confirmation_delta=comparisons[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'validation', 'confirmation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = decide(*(json.loads(p.read_text()) for p in
                      (args.candidate, args.validation, args.confirmation)))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
