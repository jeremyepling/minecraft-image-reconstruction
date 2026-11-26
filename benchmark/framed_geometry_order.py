"""Compare geometry corrections in a fixed sequence on previously exported cells."""
from benchmark.texture_camera_eval import STABLE
from engine import grounded_depth_envelope, section_decision, vertical_atlas_faces
from engine.camera_atlas_floor import _atlas

ATLAS = STABLE/'runs/26.3/atlas'


def refine(image,parts,camera,prior,policy):
    grounded,info=grounded_depth_envelope.recover(image,parts,camera,prior,atlas_path=ATLAS)
    adjusted=grounded
    if policy != 'ground':
        proposed=section_decision.recover(image,parts,camera,grounded,atlas_path=ATLAS)
        adjusted=grounded & proposed if policy=='prune' else proposed
    verified=vertical_atlas_faces.recover(image,parts,camera,adjusted,_atlas('26.3',ATLAS))
    return verified,dict(ground_removed=info['removed'],section_added=len(adjusted-grounded),
                         section_removed=len(grounded-adjusted),vertical_added=len(verified-adjusted))


def ground_and_atlas(image,parts,counts,camera,prior):
    return refine(image,parts,camera,prior,'ground')


def shorten_and_verify(image,parts,counts,camera,prior):
    return refine(image,parts,camera,prior,'prune')


def relocate_and_verify(image,parts,counts,camera,prior):
    return refine(image,parts,camera,prior,'move')
