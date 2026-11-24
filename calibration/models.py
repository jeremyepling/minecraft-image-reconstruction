"""Resolve the selected default model to a south-face texture when unambiguous."""
import json
from functools import lru_cache

DIRECTIONS = dict(north=(0,0,-1),south=(0,0,1),up=(0,1,0),down=(0,-1,0),east=(1,0,0),west=(-1,0,0))
INVISIBLE = {'air','cave_air','void_air','barrier','light','structure_void','moving_piston'}


class Models:
    def __init__(self, jar):
        self.jar = jar

    @lru_cache(None)
    def model(self, name):
        try:
            raw = json.loads(self.jar.read('assets/minecraft/models/'+name.removeprefix('minecraft:')+'.json'))
        except KeyError:
            return {}, []
        textures,elements=self.model(raw['parent']) if 'parent' in raw else ({},[])
        return {**textures,**raw.get('textures',{})},raw.get('elements',elements)

    def front(self, block, properties, direction='south'):
        """Only label an unambiguous full, untinted front of one cube model."""
        try:
            state=json.loads(self.jar.read('assets/minecraft/blockstates/'+block+'.json'))
        except KeyError:
            return None
        for key, variants in state.get('variants',{}).items():
            if key and any(properties.get(k)!=v for k,v in (p.split('=') for p in key.split(','))):
                continue
            textures=set()
            for variant in variants if isinstance(variants,list) else [variants]:
                mapping,elements=self.model(variant['model'])
                if len(elements)!=1:
                    return None
                elem=elements[0]
                if elem['from']!=[0,0,0] or elem['to']!=[16,16,16] or 'rotation' in elem:
                    return None
                for side,(x,y,z) in DIRECTIONS.items():
                    for _ in range((variant.get('x',0)//90)%4):
                        y,z=z,-y
                    for _ in range((variant.get('y',0)//90)%4):
                        x,z=-z,x
                    if (x,y,z)!=DIRECTIONS[direction]:
                        continue
                    face=elem['faces'].get(side,{})
                    texture=face.get('texture')
                    u0,v0,u1,v1=face.get('uv',[0,0,16,16])
                    if face.get('tintindex',-1)>=0 or {u0,u1}!={0,16} or {v0,v1}!={0,16}:
                        return None
                    while isinstance(texture,str) and texture.startswith('#'):
                        texture=mapping.get(texture[1:])
                    if isinstance(texture,dict):
                        texture=texture.get('sprite')
                    if not texture:
                        return None
                    textures.add(texture.split('/')[-1])
            return next(iter(textures)) if len(textures)==1 else None
        return None
