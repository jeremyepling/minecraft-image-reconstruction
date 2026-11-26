"""Package the original full-size captures, recorded references, image-only stage outputs and results."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark.roof_height_transfer_panel import SELECTION
from benchmark import visible_depth_transfer as transfer
from benchmark.texture_camera_eval import checked, digest, read, save


def archive_inputs(data,output,selection):
    manifest=read(data/'builds.json')
    selected=read(checked(selection,manifest['selection_sha256']))['selected']
    builds={r['id']:r['files'] for r in manifest['rows']}
    files={'builds.json':data/'builds.json',selection.name:selection}
    for f in selected:
        for name,sha in builds[f['id']].items():
            rel=Path('upstream')/f['id']/name
            files[rel.as_posix()]=checked(data/rel,sha)
        rgb=transfer.rgb_path(f)
        for path,n in [(rgb,'rgb.png'),(rgb.with_suffix('.json'),'camera.json'),
                       (rgb.with_name(f['source_id']+'-blocks.json'),'world-readback.json')]:
            files[(Path('sources')/f['directory']/'captures'/path.name).as_posix()]=checked(path,f['files'][n])
    with ZipFile(output,'w',ZIP_DEFLATED,compresslevel=4) as z:
        for rel,path in files.items():
            z.write(path,rel)
    with ZipFile(output) as z:
        assert set(z.namelist())==set(files) and z.testzip() is None
    return dict(archive=output.name,sha256=digest(output),members=len(files),bytes=output.stat().st_size)


def archive_results(runs,names,out):
    summaries={}
    with ZipFile(out/'scores.zip','w',ZIP_DEFLATED) as z:
        for name in names:
            here=runs/name
            m=read(here/'summary.json')
            records=checked(here/'cases.json',m['cases_sha256'])
            for path in (records,here/'summary.json'):
                z.write(path,name+'/'+path.name)
            for row in read(records):
                folder=here/row['id']
                pack=checked(folder/'reconstruction.zip',row['pack_sha256'])
                for path in (pack,folder/'cells.json'):
                    z.write(path,name+'/'+row['id']+'/'+path.name)
            summaries[name]=m
    with ZipFile(out/'scores.zip') as z:
        assert z.testzip() is None
    save(out/'scores.json',dict(archive_sha256=digest(out/'scores.zip'),methods=summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['inputs','results'])
    p.add_argument('--data',type=Path,required=True,help='Inputs folder or runs folder')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--selection',type=Path,default=SELECTION)
    p.add_argument('--name',default='framed-canopy-heights-v1.zip')
    p.add_argument('methods',nargs='*')
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    if a.stage=='inputs':
        save(a.out/'inputs.json',archive_inputs(a.data,a.out/a.name,a.selection))
    else:
        archive_results(a.data,a.methods,a.out)
