"""Replay the four native roof views using the shared visible-height scorer."""
from benchmark.roof_height_transfer_panel import main
from benchmark.texture_camera_eval import ROOT


if __name__ == '__main__':
    main(ROOT / 'benchmark/reviews/roof-height-new-views/selection.json')
