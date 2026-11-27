"""Replay the production-capable geometric and texture gate before count and placement."""
from time import perf_counter

from benchmark.wall_texture_trial import rebuild
from engine.diagonal_texture_join import refine


def predict(image,parts,counts,camera,prior):
    start=perf_counter()
    refined,groups=refine(image,parts,camera)
    elapsed=perf_counter()-start
    status=dict(groups=groups,surface_seconds=elapsed)
    if not groups:
        return prior,status
    cells,detail=rebuild(image,counts,camera,refined)
    return cells,dict(**status,**detail)
