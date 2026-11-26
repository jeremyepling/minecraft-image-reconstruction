"""Write a private, clear-noon superflat Java calibration world."""
import gzip
import struct

GROUND=64
PACK="palette_lab.zip"


def nbt_string(value):
    data=value.encode('utf-8')
    return struct.pack('>H',len(data))+data


def tag(value):
    if isinstance(value,bool): return 1,struct.pack('>b',value)
    if isinstance(value,int): return 3,struct.pack('>i',value)
    if isinstance(value,str): return 8,nbt_string(value)
    if isinstance(value,tuple):
        kind,val=value
        if kind==11: return kind,struct.pack('>i',len(val))+b''.join(struct.pack('>i',i) for i in val)
        return kind,struct.pack({4:'>q',5:'>f'}[kind],val)
    if isinstance(value,list):
        entries=[tag(v) for v in value]
        kind=entries[0][0] if entries else 0
        assert all(k==kind for k,_ in entries)
        return 9,bytes([kind])+struct.pack('>i',len(entries))+b''.join(blob for _,blob in entries)
    out=bytearray()
    for key,val in value.items():
        kind,blob=tag(val)
        out.extend(bytes([kind])+nbt_string(key)+blob)
    return 10,bytes(out)+b'\0'


def save_nbt(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(b'\x0a\x00\x00'+tag(value)[1],mtime=0))


def create_world(world, metadata, extra_rules=None, level_name='Palette Lab', *, floor='smooth_stone'):
    data_version=metadata['world_version']
    over,nether,end=['minecraft:'+s for s in ('overworld','the_nether','the_end')]
    spawn=dict(pos=(11,[0,GROUND+1,29]),pitch=(5,0.),dimension=over,yaw=(5,180.))
    level=dict(difficulty_settings=dict(difficulty='peaceful',hardcore=False,locked=False),
               Time=(4,6000),version_history=[data_version],GameType=1,ServerBrands=['vanilla'],
               version=19133,spawn=spawn,Version=dict(Snapshot=not metadata.get('stable',True),
                   Series=metadata.get('series_id','main'),Id=data_version,Name=metadata['id']),
               LevelName=level_name,initialized=True,WasModded=False,DataVersion=data_version,
               allowCommands=True,DataPacks=dict(Enabled=['vanilla','file/'+PACK],Disabled=[]))
    save_nbt(world/'level.dat',{'Data':level})
    flat=dict(features=False,biome='minecraft:plains',structure_overrides=[],lakes=False,layers=[
        dict(block='minecraft:bedrock',height=1),dict(block='minecraft:stone',height=127),
        dict(block='minecraft:'+floor,height=1)])
    dims={over:dict(type=over,generator=dict(type='minecraft:flat',settings=flat)),
          nether:dict(type=nether,generator=dict(type='minecraft:noise',settings='minecraft:nether',
                                                 biome_source=dict(type='minecraft:multi_noise',preset='minecraft:nether'))),
          end:dict(type=end,generator=dict(type='minecraft:noise',settings='minecraft:end',biome_source=dict(type=end)))}
    for file,data in [('world_gen_settings',dict(bonus_chest=False,seed=(4,732431),generate_structures=False,dimensions=dims)),
                      ('world_clocks',{over:{'total_ticks':(4,6000)},end:{'total_ticks':(4,6000)}}),
                      ('weather',dict(raining=False,thundering=False,rain_time=0,thunder_time=0,clear_weather_time=1000000)),
                      ('game_rules',{'minecraft:advance_time':False,'minecraft:advance_weather':False,
                                     'minecraft:spawn_mobs':False,'minecraft:respawn_radius':0,**(extra_rules or {})})]:
        save_nbt(world/f'data/minecraft/{file}.dat',dict(data=data,DataVersion=data_version))
