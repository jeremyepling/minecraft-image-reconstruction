"""Exercise the candidate from a normal RGB CLI with a single experimental Surface Finder hook."""
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from benchmark.inset_return_grid import refine as pockets
from engine import guarded_wall_outlines
from engine.camera_metric import predict as camera


def main():
    target=Path(sys.argv[1]).resolve()
    sys.argv=sys.argv[1:]
    original=guarded_wall_outlines.refine
    def refined(image, parts):
        parts,events=original(image,parts)
        result,changes=pockets(image,parts,camera(image,'26.3'))
        return result,events+[dict(inset_return=event) for event in changes]
    with patch.object(guarded_wall_outlines,'refine',refined):
        runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':
    main()
