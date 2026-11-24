"""Measure dimensions and locations from a fitted lattice without material predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from engine.geometry import project


def measure(geometry, layout):
    if layout.get('image_sha256',geometry['image_sha256']) != geometry['image_sha256']:
        raise ValueError('Height landmarks belong to another image')
    spec=layout['dimensions']
    planes={p['name']:p for p in geometry['planes']}
    frame=spec['front_elevation']
    front=np.linalg.inv(planes[frame['reference_plane']]['grid_to_image'])
    origin=project(front,frame['origin_px'])
    def elevation(points):
        # Flatten sightlines onto the reference wall. Depth offsets in the
        # actual scene introduce parallax for surfaces on another plane.
        return ((project(front,points)-origin)*[1,-1]).round(3).tolist()
    measures={}
    for plane in layout['planes']:
        name=plane['name']
        grid=np.array(planes[name]['grid_to_image'])
        quad=project(np.linalg.inv(grid),plane['outline'])
        spans=quad[[3,2],1]-quad[[0,1],1]
        widths=quad[[1,2],0]-quad[[0,3],0]
        if np.any(spans<=0) or np.any(widths<=0):
            raise ValueError(f'Unexpected outline orientation for {name}')
        measures[name]=dict(left_rows=float(spans[0]),right_rows=float(spans[1]),
                            rows=float(spans.mean()),columns=float(widths.mean()),
                            outline_px=plane['outline'],local_grid_corners=quad.round(3).tolist(),
                            projected_elevation_corners=elevation(plane['outline']))
        locations=[]
        for c in planes[name]['cells']:
            center=project(grid,np.array(c['ij'])+.5)
            locations.append(dict(ij=c['ij'],center_px=center.round(2).tolist(),
                                  projected_elevation_center=elevation(center),
                                  visible_fraction=c['visible_fraction']))
        measures[name]['cells']=locations
    bands=spec['bands_bottom_to_top']
    if len(bands)!=len(set(bands)) or not bands:
        raise ValueError('Height needs one non-repeating vertical stack')
    raw=sum(measures[name]['rows'] for name in bands)
    def span(names):
        if len(names)!=len(set(names)) or not names:
            raise ValueError('A dimension needs a non-repeating span of faces')
        return sum(measures[n]['columns'] for n in names)
    width=span(spec['width_bands_left_to_right'])
    depth=span(spec['depth_bands_front_to_back']) if spec.get('depth_bands_front_to_back') else None
    return dict(image_sha256=geometry['image_sha256'],baseline=spec['baseline'],
                roof_edge=spec['roof_edge'],bands_bottom_to_top=bands,
                front_elevation=dict(**frame,axes=['right','up'],unit='blocks'),
                width_bands_left_to_right=spec['width_bands_left_to_right'],
                estimated_front_width=width,front_width=int(np.rint(width)),estimated_depth=depth,
                planes=measures,estimated_rows=raw,visible_height=int(np.rint(raw)),
                roof_detail_allowance=spec.get('roof_detail_allowance',0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('geometry','layout','out'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--image',type=Path)
    args=parser.parse_args()
    result=measure(json.loads(args.geometry.read_text()),json.loads(args.layout.read_text()))
    if args.image and hashlib.sha256(args.image.read_bytes()).hexdigest()!=result['image_sha256']:
        raise ValueError('Geometry belongs to another screenshot')
    result['geometry_sha256']=hashlib.sha256(args.geometry.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result,indent=2))
