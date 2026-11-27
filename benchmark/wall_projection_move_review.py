"""Grade three predeclared image-only choices and the best possible single move per photo."""
import argparse
from collections import Counter
from pathlib import Path

from benchmark.texture_camera_eval import read,save
from benchmark.wall_source_projection import LABELS


def winner(trials,method):
    if method=='geometry':
        candidates=[t for t in trials if t['geometry_gain']>=6 and t['support']>=.8]
        key=lambda t:t['geometry_gain']
    elif method=='appearance':
        candidates=[t for t in trials if t['held_out']>=16 and t['rgb_gain']>=2 and t['geometry_gain']>=0]
        key=lambda t:t['rgb_gain']
    else:
        candidates=[t for t in trials if t['held_out']>=16 and t['rgb_gain']>=2 and t['geometry_gain']>=6
                    and t['support']>=.8]
        key=lambda t:t['geometry_gain']*min(t['rgb_gain'],12)
    return max(candidates,key=key,default=None)


def run(folder):
    rows=[]
    def delta(t):
        return int(t['new_correct'])-int(t['old_correct'])
    for name in LABELS:
        trials=read(folder/(name.lower().replace(' ','-')+'-moves.json'))
        groups={}
        for group,samples in [('all',trials),('positive_geometry',[t for t in trials if t['geometry_gain']>=6]),
                              ('positive_appearance',[t for t in trials if t['held_out']>=16 and t['rgb_gain']>=2]),
                              ('positive_both',[t for t in trials if t['geometry_gain']>=6 and t['held_out']>=16
                                                and t['rgb_gain']>=2])]:
            groups[group]=dict(candidates=len(samples),correct_source_to_miss=sum(delta(t)<0 for t in samples),
                               extra_to_correct=sum(delta(t)>0 for t in samples))
        methods={}
        for method in ('geometry','appearance','both'):
            chosen=winner(trials,method)
            methods[method]=dict(delta=delta(chosen),trial=chosen) if chosen else dict(delta=0,trial=None)
        rows.append(dict(name=name,groups=groups,methods=methods,
                         best_single_move_delta=max([0]+[delta(t) for t in trials])))
    save(folder/'move-review.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    run(p.parse_args().folder)
