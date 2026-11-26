"""Select the roof plane from the photographed patch before assigning individual grid cells."""
from unittest.mock import patch

import numpy as np

from engine import atlas_plane_depth, roof_height_atlas
from engine.camera_atlas_floor import _atlas
from engine.foreground_conflicts import ForegroundConflicts
from engine.surface_model import intrinsic
from engine.thin_atlas_faces import _mips


def choose(image,parts,camera,atlas,require_contacts=False):
    status=dict(height_decisions=[],primary_height=None)
    if not parts or camera.get('P') is None or atlas is None:
        return {},status
    P=np.array(camera['P'],dtype=float,copy=True)
    P/=np.linalg.norm(P[2,:3])
    inv=np.linalg.inv(P[:,:3])
    eye=-inv@P[:,3]
    focal=np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    rot=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    rgb=np.asarray(image.convert('RGB'))
    maps=_mips(atlas)
    levels={}
    tips={}
    if require_contacts:
        from engine.roof_height_rim import tops, _surface_masks
        regions=[p for p in parts if p.get('support') and p.get('normal_camera') is not None]
        if not regions:
            return {},status
        xs=np.arange(.5,image.width,2)
        ys=np.arange(.5,image.height,2)
        masks=[_surface_masks([p],image.size,xs,ys,rot)[0] for p in regions]
        axes=[int(np.argmax(abs(rot.T@p['normal_camera']))) for p in regions]
        tips=tops(masks,axes,eye,inv,2)
    area=0
    for part in parts:
        pixels=roof_height_atlas.horizontal_pixels(part,image.size,rot)
        if pixels is None:
            continue
        x,y=pixels
        area+=len(x)
        rays=np.column_stack((x+.5,y+.5,np.ones(len(x)))) @ inv.T
        if np.min(abs(rays[:,1])) < 1e-6:
            continue
        # No prior block coordinates enter this search. Heights are relative to the measured floor.
        ranked=roof_height_atlas.rank_planes(x,y,rays,rgb[y,x],eye,range(1,65),atlas,maps)
        if len(ranked)<2:
            continue
        best,second=ranked[:2]
        margin=best['value']-second['value']
        strong=best['value']>=.38 and margin>=.08
        # A weak absolute match can still distinguish heights when both separated
        # halves fit and the next-best alternative has little support.
        distinctive=margin>=.20 and best['pixel_corr']>=.30 and min(best['halves'])>=.20
        take=(strong or distinctive) and best['phase']>.06 and best['color']<14 and best['texels']>=16
        h=best['plane_y']
        span=roof_height_atlas.support_span(rays,eye,h,tips) if require_contacts else None
        if require_contacts:
            take=take and span>=3 and margin>=.18
        decision=dict(part=part['id'],plane_y=h,block_y=h-int(eye[1]>h),accepted=bool(take),
                      area=len(x),support_span=span,pixel_corr=best['pixel_corr'],halves=best['halves'],
                      margin=round(margin,4),runner_up_plane_y=second['plane_y'],distinctive=bool(distinctive))
        status['height_decisions'].append(decision)
        if take:
            levels[part['id']]=h
    if require_contacts:
        for r in status['height_decisions']:
            if r['area']<.65*area:
                levels.pop(r['part'],None)
                r['accepted']=False
    status['primary_height']=max((r for r in status['height_decisions'] if r['accepted']),
                                 key=lambda r:(r['area']*r['pixel_corr'],r['margin']),default=None)
    return levels,status


class UnchangedPlanes(Exception):
    """The previous placer independently chose identical levels for every selected surface."""


def recover(image,parts,counts,camera,prior,*,require_contacts=False):
    levels,status=choose(image,parts,camera,_atlas('26.3'),require_contacts)
    if not levels:
        return prior,status
    original=ForegroundConflicts.resolve
    def resolve(policy):
        chosen=original(policy)
        changes={key:h for key,h in levels.items() if chosen.get(key)!=h}
        if require_contacts:
            changes={key:h for key,h in changes.items() if key in chosen and abs(chosen[key]-h)==1}
        status['early_overrides']={key:[chosen.get(key),h] for key,h in changes.items()}
        if not changes:
            raise UnchangedPlanes
        chosen.update(changes)
        policy.levels.update(changes)
        return chosen
    try:
        with patch.object(ForegroundConflicts,'resolve',resolve):
            cells,_=atlas_plane_depth._place_ranked_once(image,'26.3',parts,counts,camera)
    except UnchangedPlanes:
        cells=prior
    return cells,status


def supported(image,parts,counts,camera,prior):
    return recover(image,parts,counts,camera,prior,require_contacts=True)
