"""B's existing vanilla texture-spacing measurement on a supplied sampling chart."""
from engine.surface_pitch import fit


def count(sample, version='26.3'):
    if sample is None:
        return dict(method='legacy-repeat', counts=[None, None], phase=[None, None],
                    reason='No sampling chart', evidence={})
    measured = fit(sample['source_image'], sample['quad'], version)
    return dict(method='legacy-repeat', counts=measured['counts'] or [None, None],
                phase=measured['evidence'].get('phase_by_axis', [None, None]),
                confidence=measured['confidence'], reason=measured['reason'],
                evidence=measured['evidence'])
