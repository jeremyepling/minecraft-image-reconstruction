"""Independently reassemble the split-border packet and compare decoded local and aligned IDs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from benchmark import floating_pregrid_depth_refit as prior, floating_v1_run as v1
from benchmark.expanded_baseline import digest,read,save
from benchmark.mask_blocks import decoded_pack


def verify(case):
    base=v1.OUT/"split-border"
    out=v1.OUT/"no-op/assembly"
    n=case["id"]
    target=base/"assembly"/v1.PREFIX/n
    other=out/v1.PREFIX/n
    with patch.object(v1.assembly,"load_reference",v1.score_reference),patch.object(prior,"data",lambda g:(v1.SOURCE,)):
        prior.replay("v1",case,base/"charts",out,{})
    assert decoded_pack(other/"reconstruction.zip")==decoded_pack(target/"reconstruction.zip")
    assert read(other/"placements.json")==read(target/"placements.json")
    assert read(other/"score.json")["score"]==read(target/"score.json")["score"]
    evidence=dict(id=n,chart=digest(base/"charts/sky"/n/"charts.json"),new_pack=digest(other/"reconstruction.zip"),
                  original_pack=digest(target/"reconstruction.zip"),decoded_equal=True,registered_equal=True)
    save(v1.OUT/"no-op/rows"/(n+".json"),evidence)
    return n


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    jobs=[c for c in v1.tasks() if (v1.OUT/"rows"/(c["id"]+".json.gz")).is_file()
          and not (v1.OUT/"no-op/rows"/(c["id"]+".json")).is_file()]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(verify,jobs))
