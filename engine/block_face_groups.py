"""Group projected full-block faces at unambiguous outward cube edges."""
from collections import defaultdict
from itertools import combinations

import numpy as np

from engine.face_evidence import effective_width, MIN_FACE_WIDTH
from engine.surface_model import lens, lift


def _geometry(face, camera, inverse):
    points = np.asarray(face['quad'], dtype=float)
    rays = np.column_stack((points, np.ones(4))) @ inverse.T
    normal = np.asarray(face.get('normal_camera'), dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<.1:
        estimate = lift(points,camera,{})
        if estimate is None:
            return None
        normal = estimate[1]
    normal = normal / np.linalg.norm(normal)
    incidence = rays @ normal
    if np.all(incidence>0):
        normal, incidence = -normal, -incidence
    if np.any(incidence>=-1.e-4):
        return None
    return points, rays/(-incidence[:,None]), normal


def _convex_link(a, edge_a, b, edge_b):
    qa, xyz_a, na = a
    qb, xyz_b, nb = b
    if abs(na@nb)>.35:
        return None
    i,j = [edge_a,(edge_a+1)%4], [edge_b,(edge_b+1)%4]
    if np.linalg.norm(qa[i[0]]-qb[j[0]]) > np.linalg.norm(qa[i[0]]-qb[j[1]]):
        j.reverse()
    distance = np.linalg.norm(qa[i]-qb[j],axis=1).max()
    length = np.linalg.norm(qa[i[1]]-qa[i[0]])
    if length<3 or distance>min(2.5,.13*length):
        return None
    edge = qa[i[1]]-qa[i[0]]
    def cross_to_center(q):
        v=q.mean(0)-qa[i[0]]
        return (edge[0]*v[1]-edge[1]*v[0])/length
    if cross_to_center(qa)*cross_to_center(qb)>=0:
        return None

    # Anchor both planes at the shared edge and compare their 3D interiors.
    # A convex corner puts each interior behind the other's outward plane.
    anchor_a=xyz_a[i].mean(0)
    anchor_b=xyz_b[j].mean(0)
    a_local=(xyz_a-anchor_a)/anchor_a[2]
    b_local=(xyz_b-anchor_b)/anchor_b[2]
    edge_length=np.linalg.norm(a_local[i[0]]-a_local[i[1]])
    if np.linalg.norm(a_local[i]-b_local[j],axis=1).max()>.18*edge_length:
        return None
    inward_a=-float(a_local.mean(0)@nb)/edge_length
    inward_b=-float(b_local.mean(0)@na)/edge_length
    if not (.20<inward_a<.85 and .20<inward_b<.85):
        return None
    return dict(shared_edge_error_px=float(distance),inward_distances_in_edges=[inward_a,inward_b])


def group_faces(faces, size):
    """Input order indexes observations. Uncertain joins remain single faces."""
    if not faces:
        return []
    quads=[np.asarray(f['quad'],float) for f in faces]
    camera,_=lens(quads,size)
    inverse=np.linalg.inv(camera)
    geometry=[_geometry(f,camera,inverse) for f in faces]
    bins=defaultdict(list)
    links={}
    surface_ids=[f.get('surface_id') or f.get('proposal') for f in faces]
    for idx,g in enumerate(geometry):
        if g is None:
            continue
        for edge in range(4):
            center=(g[0][edge]+g[0][(edge+1)%4])/2
            bx,by=np.floor(center/3).astype(int)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for other,other_edge in bins.get((bx+dx,by+dy),()):
                        if idx==other or (surface_ids[idx] is not None and surface_ids[idx]==surface_ids[other]):
                            continue
                        link=_convex_link(g,edge,geometry[other],other_edge)
                        if link is not None:
                            links[other,idx]=link
            bins[bx,by].append((idx,edge))
    neighbors=defaultdict(set)
    for i,j in links:
        neighbors[i].add(j)
        neighbors[j].add(i)
    widths=[effective_width(q) for q in quads]
    areas=[abs(np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1)))/2 for q in quads]
    result=[]
    visited=set()
    for idx in range(len(faces)):
        if idx in visited:
            continue
        component={idx}
        queue=[idx]
        while queue:
            new=neighbors[queue.pop()]-component
            queue.extend(new)
            component.update(new)
        ordered=sorted(component)
        visited.update(component)
        # Three visible faces of one cube have three mutually supported edges.
        unambiguous=len(ordered)<=3 and all(pair in links for pair in combinations(ordered,2))
        for members in ([ordered] if unambiguous else [[i] for i in ordered]):
            eligible=[i for i in members if widths[i]>=MIN_FACE_WIDTH]
            selected=max(eligible,key=lambda i:(widths[i],areas[i],-i),default=None)
            result.append(dict(face_indices=members,selected_index=selected,
                               ambiguous_neighbors=not unambiguous,
                               evidence=[dict(face_indices=list(pair),**links[pair]) for pair in combinations(members,2)]))
    return result


def assign_block_materials(image, faces, rank_tile):
    """Make one material comparison per observed block and keep per-face evidence."""
    from engine.surface_tiles import tile_sample
    groups = group_faces(faces, image.size)
    for block_number, group in enumerate(groups):
        selected = group['selected_index']
        material = None
        if selected is not None:
            material = rank_tile(tile_sample(image,faces[selected]['quad']))
        group.update(block_index=block_number,material=material)
        for idx in group['face_indices']:
            row=faces[idx]
            row.update(block_index=block_number,material_source_face=selected)
            if idx == selected:
                row.update(material=material,material_status='read from this face')
            elif selected is None:
                row.update(material_status='unresolved: no face reaches five source pixels')
            else:
                row.update(material=dict(block_id=material['block_id']),material_status='inherited from another face')
    return groups
