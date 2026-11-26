"""Measure narrow, photographed horizontal strips at native pixel centers."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from engine.surface_model import intrinsic
from engine.atlas_phase import ATLAS_SHA256
from engine.visible_horizontal_faces import _surface_masks, _keep_prior_visible
from engine.visible_placement import _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE


def _mips(atlas):
    """Block-average the native sprite over a two- or four-texel footprint."""
    original = atlas.tiles.astype(np.float32).mean(3)
    maps = []
    for k in (1,2,4):
        plane = original.reshape(-1,16//k,k,16//k,k).mean((2,4)).repeat(k,1).repeat(k,2)
        rotated = np.stack([np.rot90(g,n,axes=(1,2)) for g in (plane,plane[:,:,::-1])
                            for n in range(4)],axis=1)
        maps.append(rotated.reshape(-1,8,256))
    return np.stack(maps,axis=1)


def _corr(a,b):
    centered = a-a.mean(axis=-1,keepdims=True)
    target = b-b.mean()
    return np.einsum('...k,k->...',centered,target)/np.maximum(.001,np.linalg.norm(centered,axis=-1)*np.linalg.norm(target))


def _match(uv, color, atlas, maps, palette=()):
    # Color is sampled once per visible native pixel. Multiple pixels falling
    # into one texel are averaged before matching, so one long texel cannot dominate.
    idx = (uv[:,1]%16)*16+uv[:,0]%16
    ids, group, n = np.unique(idx,return_inverse=True,return_counts=True)
    gray = (np.bincount(group,weights=color.mean(1))/n).astype(np.float32)
    if len(ids)<6 or gray.std()<.28:
        return None
    mean = color.mean(0)
    gain = np.clip((atlas.means@mean)/np.maximum(1,(atlas.means**2).sum(1)),.30,1.18)
    color_distance = abs(gain[:,None]*atlas.means-mean).mean(1)+2*abs(np.log(gain/.80))
    short = np.asarray(palette,int) if len(palette) else np.argpartition(color_distance,28)[:28]
    choices = maps[short]
    aligned = _corr(choices[...,ids],gray)
    # Compete on the same partition used to select the texture; use independent
    # halves below to guard against the lucky matches possible on a small strip.
    penalized = aligned-color_distance[short,None,None]/65.
    ti,mi,oi = np.unravel_index(penalized.argmax(),penalized.shape)
    template = choices[ti,mi,oi]
    best = float(aligned[ti,mi,oi])
    rng = np.random.default_rng(291)
    half = np.zeros(len(ids),bool)
    half[rng.permutation(len(ids))[:len(ids)//2]] = True
    split = min(_corr(template[ids[sel]],gray[sel]) for sel in (half,~half))
    competitors = [ids//16*16+(ids+d)%16 for d in (4,8)]
    competitors += [(ids+d*16)%256 for d in (4,8)]
    rival = max(_corr(template[ij],gray) for ij in competitors)
    tex_id = int(short[ti])
    per_sprite = penalized.max(axis=(1,2))
    runner = np.partition(per_sprite,-2)[-2] if len(short)>1 else per_sprite[ti]
    return dict(sprite=tex_id,turn=int(oi),mip=int(mi),corr=round(best,4),split=round(float(split),4),
                phase=round(best-float(rival),4),color=round(float(color_distance[tex_id]),2),
                margin=round(float(per_sprite[ti]-runner),4),texels=len(ids),std=round(float(gray.std()),2))


def _projection_margin(eye, rays, plane, uv, colors, templates):
    gray=colors.mean(1).astype(np.float32)
    def score(coords):
        ix=np.floor(coords*16).astype(int)
        idx=16*(ix[:,1]%16)+(ix[:,0]%16)
        return float(_corr(templates[...,idx],gray).max())
    current=score(uv)
    depths=[score(eye[[0,2]]+(p-eye[1])*rays[:,[0,2]]/rays[:,[1]]) for p in (plane-1,plane+1)]
    walls=[]
    for axis,free in [(0,[1,2]),(2,[0,1])]:
        if np.any(abs(rays[:,axis])<1e-5):
            continue
        constant=round(float(np.median(uv[:,0 if axis==0 else 1])))
        walls.append(score(eye[free]+(constant-eye[axis])*rays[:,free]/rays[:,[axis]]))
    return round(current-max(depths),4),round(current-max(walls,default=current),4)


class StripEvidence:
    def __init__(self,image,parts,camera,prior,atlas):
        self.size = image.size
        self.shape = image.size[::-1]
        self.rgb = np.asarray(image.convert('RGB'))
        self.prior = prior
        self.atlas = atlas
        self.maps = _mips(atlas)
        P = np.asarray(camera['P'],float)
        self.P = P = P/np.linalg.norm(P[2,:3])
        self.native = P.copy()
        self.native[:2] *= SAMPLE_STRIDE
        self.inverse = np.linalg.inv(P[:,:3])
        self.eye = -self.inverse@P[:,3]
        focal = np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
        rotation = np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
        self.union,self.masks = _surface_masks(parts,self.size,np.arange(image.width)+.5,
                                               np.arange(image.height)+.5,rotation)
        points = sorted(prior)
        owner,self.axes = _frontmost_cells(points,self.native,self.eye,self.shape,face_axes=True)
        self.depth = np.zeros(self.shape)
        v,u = np.nonzero(owner>=0)
        a = self.axes[v,u]
        rays = self._rays(u,v)
        level = np.asarray(points)[owner[v,u],a]
        plane = level+(self.eye[a]>level+1)
        self.depth[v,u] = rays[np.arange(len(u)),a]/(plane-self.eye[a])
        self.anchors = self._palette(points,owner)

    def _palette(self,points,owner):
        """Identify sprites on visible reference blocks before considering any additions."""
        # A stride-two sample here describes the palette. Candidate strip evidence
        # below still uses every actual native pixel, including one-pixel strips.
        v,u = np.nonzero(owner[::2,::2]>=0)
        v,u = v*2,u*2
        key = owner[v,u]*3+self.axes[v,u]
        order = np.argsort(key,kind='stable')
        groups = np.split(order,np.flatnonzero(np.diff(key[order]))+1)
        large = [g for g in groups if len(g)>=18]
        if len(large)>150:
            large = [large[i] for i in np.linspace(0,len(large)-1,150,dtype=int)]
        anchors = {}
        for g in large:
            idx,axis=divmod(int(key[g[0]]),3)
            cell=points[idx]
            xx,yy=u[g],v[g]
            take=np.linspace(0,len(g)-1,min(len(g),250),dtype=int)
            xx,yy=xx[take],yy[take]
            rays=self._rays(xx,yy)
            free=[d for d in range(3) if d!=axis]
            plane=cell[axis]+int(self.eye[axis]>cell[axis]+1)
            uv=self.eye[free]+(plane-self.eye[axis])*rays[:,free]/rays[:,[axis]]
            colors=self.rgb[yy,xx]
            match=_match(np.floor(uv*16).astype(int),colors,self.atlas,self.maps)
            if match and match['texels']>=16 and match['split']>=.58 and match['phase']>.05 and match['color']<9:
                key_id=match['sprite']
                anchors.setdefault(key_id,[]).append((cell,axis,colors.mean(0)))
        return anchors

    def _rays(self,x,y):
        return np.column_stack((x+.5,y+.5,np.ones(len(x))))@self.inverse.T

    def candidates(self):
        # Search nearby visible courses. The generating neighborhood supplies
        # neither a placement nor an occupancy target; every face is measured below.
        return {(x+dx,y,z+dz) for x,y,z in self.prior for dx in range(-3,4) for dz in range(-3,4)
                if 0<abs(dx)+abs(dz)<=3}-self.prior

    def _side_support(self,cell,face_box,face_front,joint):
        votes = []
        for axis in range(3):
            if axis==1:
                box,front = face_box,face_front
            else:
                if cell[axis]<=self.eye[axis]<=cell[axis]+1:
                    continue
                hit = _sample_face(self.native,cell,axis,1 if self.eye[axis]>cell[axis]+1 else -1,self.shape)
                if hit is None:
                    continue
                box,pixels,depth,*_ = hit
                front = pixels & (depth>self.depth[box]+1e-8)
                if joint is not None:
                    owner,axes,ids = joint
                    front &= (owner[box]==ids[cell]) & (axes[box]==axis)
            count = int(front.sum())
            if count:
                matches = np.count_nonzero(front & self.masks[axis][box])
                inside = np.count_nonzero(front & self.union[box])
                old = self.axes[box]
                same = np.count_nonzero(front & (old == axis))
                votes.append((count,matches,inside,same))
        total,aligned,inside,same = np.sum(votes,axis=0)
        return dict(aligned=round(aligned/total,4),inside=round(inside/total,4),
                    over_same=round(same/total,4))

    def measure(self,cell,joint=None):
        x,y,z = cell
        if y<=self.eye[1]<=y+1:
            return None
        above = self.eye[1]>y+1
        hit = _sample_face(self.native,cell,1,1 if above else -1,self.shape)
        if hit is None:
            return None
        box,face,depth,_,_ = hit
        front = face & (depth>self.depth[box]+1e-8)
        if joint is not None:
            owner,axes,ids = joint
            front &= (owner[box]==ids[cell]) & (axes[box]==1)
        witness = front & self.union[box] & (self.axes[box]!=1)
        n = int(witness.sum())
        if n<12 or n<.35*front.sum():
            return None
        yy,xx = np.nonzero(witness)
        step = max(1,len(xx)//200)
        xx,yy = xx[::step]+box[1].start,yy[::step]+box[0].start
        rays = self._rays(xx,yy)
        uv = self.eye[[0,2]] + (y+int(above)-self.eye[1]) * rays[:,[0,2]]/rays[:,[1]]
        bins=np.floor(16*uv).astype(int)
        colors=self.rgb[yy,xx]
        match = _match(bins,colors,self.atlas,self.maps)
        if match is None:
            return None
        palette=list(self.anchors)
        if palette and match['sprite'] not in self.anchors:
            known = _match(bins,colors,self.atlas,self.maps,palette)
            if known and known['split']-known['color']/60 >= match['split']-match['color']/60-.14:
                match=known
        refs=self.anchors.get(match['sprite'],[])
        same=[(abs(x-p[0])+abs(z-p[2]),abs(colors.mean(0)-color).mean()) for p,axis,color in refs
              if axis==1 and p[1]==y]
        match.update(donors=len(refs),course=len(same),donor_dist=min((d for d,_ in same),default=50),
                     donor_color=round(float(min((e for _,e in same),default=50)),2))
        match['depth_margin'],match['axis_margin']=_projection_margin(self.eye,rays,y+int(above),uv,colors,
                                                                      self.maps[match['sprite']])
        near = sum((x+i,y,z+j) in self.prior for i,j in ((1,0),(-1,0),(0,1),(0,-1)))
        horiz = self.masks[1][box] & witness
        coverage=self._side_support(cell,box,front,joint)
        return dict(match,**coverage,n=n,area=int(face.sum()),fraction=round(n/int(front.sum()),4),
                    oriented=round(int(horiz.sum())/n,4),near=near)

    def joint(self,selected):
        cells = sorted(self.prior|selected)
        owner,axes = _frontmost_cells(cells,self.native,self.eye,self.shape,face_axes=True)
        ids = {c:i for i,c in enumerate(cells)}
        return owner,axes,ids


FEATURES = ('correlation','half_match','phase','color','sprite_margin','brightness_spread','texel_count',
            'sample_count','face_coverage','orientation','coplanar_neighbors','mipmap',
            'all_axes','mask_coverage','occludes_matching_axis','other_faces','coherent_neighbor',
            'same_sprite','mask_purity','visible_fraction','sprite_support','course_support','donor_proximity',
            'donor_color','depth_margin','axis_margin')
WEIGHTS = Path(__file__).with_name('thin_face_weights.json')


def features(c,r,rows):
    x,y,z = c
    adjacent = [rows[k] for k in ((x+1,y,z),(x-1,y,z),(x,y,z+1),(x,y,z-1)) if k in rows]
    neighbors = [a for a in adjacent if a['split']>.4 and a['phase']>.03 and a['color']<12]
    return [r['corr'],r['split'],np.clip(r['phase'],-.5,1),np.log1p(r['color']),
            min(r['margin'],.6),np.log1p(r['std']),np.log(r['texels']),np.log(r['n']),
            r['fraction'],r['oriented'],min(3,r['near']),r['mip'],r['aligned'],r['inside'],
            r['over_same'],len(adjacent),len(neighbors),sum(a['sprite']==r['sprite'] for a in neighbors),
            float(r['aligned']>=.97),r['n']/max(1,r['area']),np.log1p(r['donors']),np.log1p(r['course']),
            np.log1p(r['donor_dist']),np.log1p(r['donor_color']),r['depth_margin'],r['axis_margin']]


@lru_cache(maxsize=2)
def _model(path,mtime):
    model = json.loads(path.read_text())
    if model['features']!=list(FEATURES):
        raise ValueError('Incompatible thin-face model')
    for k in ('mean','scale','weights'):
        vector=np.asarray(model[k],float)
        if vector.shape!=(len(FEATURES),) or not np.isfinite(vector).all():
            raise ValueError('Invalid thin-face weights')
        model[k]=vector
    if not np.all(model['scale']>0) or not np.isfinite(float(model['bias'])) or 'forest' in model:
        raise ValueError('Invalid thin-face model scale or bias')
    model['bias']=float(model['bias'])
    return model


def model_probabilities(values,model):
    if 'forest' in model:
        def score(tree):
            if isinstance(tree,list):
                k,t,left,right=tree
                return np.where(values[:,k]<=t,score(left),score(right))
            return tree
        linear=np.full(len(values),model['bias'])
        for tree in model['forest']:
            linear+=score(tree)
    else:
        linear = (values-np.array(model['mean']))/model['scale']@model['weights']+model['bias']
    return 1/(1+np.exp(-linear.clip(-30,30)))


def probabilities(rows,model,context=None):
    if not rows:
        return {}
    values = np.asarray([features(c,r,rows if context is None else context) for c,r in rows.items()])
    return dict(zip(rows,model_probabilities(values,model)))


def choose(rows,mode,context=None):
    if mode.startswith('rank'):
        try:
            model=_model(WEIGHTS,WEIGHTS.stat().st_mtime_ns)
        except (OSError,KeyError,TypeError,ValueError):
            return set()
        threshold=float(mode[4:])/100
        return {c for c,p in probabilities(rows,model,context).items() if p>=threshold
                and rows[c]['split']>=.25 and rows[c]['phase']>.02 and rows[c]['texels']>=8}
    accepted = set()
    for c,r in rows.items():
        if r['texels']<10 or r['n']<16 or r['color']>10 or r['fraction']<.60:
            continue
        score = r['split']-.010*r['color']
        if mode=='strong':
            good = score>=.78 and r['phase']>=.16 and r['oriented']>=.55
        elif mode=='moderate':
            good = score>=.60 and r['phase']>=.10 and r['oriented']>=.70
        elif mode=='union':
            good = score>=.80 and r['phase']>=.20
        else:
            raise ValueError(mode)
        if good:
            accepted.add(c)
    return accepted


def recover(image,parts,camera,prior,atlas,mode='rank85'):
    provenance=getattr(atlas,'provenance',None) or {}
    if (not prior or camera['P'] is None or not any(p.get('support') for p in parts)
            or provenance.get('version')!='26.3' or provenance.get('atlas_sha256')!=ATLAS_SHA256):
        return prior
    if mode.startswith('rank'):
        try:
            _model(WEIGHTS,WEIGHTS.stat().st_mtime_ns)
        except (OSError,KeyError,TypeError,ValueError,OverflowError):
            return prior
    ctx = StripEvidence(image,parts,camera,prior,atlas)
    rows = {c:r for c in ctx.candidates() if (r:=ctx.measure(c)) is not None}
    selected = choose(rows,mode)
    if selected:
        joint = ctx.joint(selected)
        verified = {c:r for c in selected if (r:=ctx.measure(c,joint)) is not None}
        selected = choose(verified,mode,rows)
    if not selected:
        return prior
    return _keep_prior_visible(image.size,ctx.P,ctx.eye,prior,selected)
