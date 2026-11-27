"""Run a full build with masked repetition preserved and only its texture-guided retry removed."""
import sys
from unittest.mock import patch

from engine import texture_guided_count
import reconstruct


def without_retry(photo, part, sample, baseline=None, version='26.3'):
    if baseline is None:
        baseline = texture_guided_count.count(sample, version=version)
    return baseline, sample


if __name__ == '__main__':
    with patch.object(texture_guided_count,'count_surface',without_retry), patch.object(sys,'argv',sys.argv[1:]):
        reconstruct.main()
