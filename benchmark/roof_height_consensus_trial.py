"""Test a corroborated one-course revision before grid cells are assigned to the final build."""
from unittest.mock import patch

from benchmark.roof_height_early import UnchangedPlanes
from benchmark.texture_camera_eval import STABLE
from engine import atlas_plane_depth, roof_height_atlas, roof_height_consensus
from engine.foreground_conflicts import ForegroundConflicts


def consensus(image,parts,counts,camera,prior):
    atlas_path=STABLE/'runs/26.3/atlas'
    status=roof_height_consensus.select(image,parts,counts,camera,prior,atlas_path)
    primary=status['primary_height']
    if primary is None:
        return prior,status
    previous=ForegroundConflicts.resolve
    def resolve(policy):
        chosen=previous(policy)
        changes=roof_height_consensus.one_course_override(primary,chosen)
        status['early_overrides']={key:[chosen[key],h] for key,h in changes.items()}
        if not changes:
            raise UnchangedPlanes
        chosen.update(changes)
        policy.levels.update(changes)
        return chosen
    try:
        with patch.object(ForegroundConflicts,'resolve',resolve):
            cells,_=atlas_plane_depth._place_ranked_once(image,'26.3',parts,counts,camera,atlas_path=atlas_path)
    except UnchangedPlanes:
        cells=prior
    return cells,status


def hybrid(image,parts,counts,camera,prior):
    cells,status=consensus(image,parts,counts,camera,prior)
    final,detail=roof_height_atlas.recover(image,parts,counts,camera,cells,mode='strict',
                                           atlas_path=STABLE/'runs/26.3/atlas')
    # Keep the early decision and the later per-surface confirmations distinct.
    status['verified_height_decisions']=detail['height_decisions']
    status['verified_added']=len(final-cells)
    status['verified_removed']=len(cells-final)
    return final,status
