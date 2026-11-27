"""Grade full ordinary pipeline exports from the material feedback component switches."""
import argparse
from pathlib import Path

from benchmark import material_feedback_geometry as exp, typed_structures as typed
from benchmark.mask_blocks import decoded_pack
from benchmark.material_feedback_bypasses import metrics
from benchmark.pavement_opening_eval import visible_faces
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.thin_face_placement_run import ref_inputs
from benchmark.texture_camera_eval import digest, read

EXAMPLES={'colonnade':('inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right',('control','skip_edges')),
          'loadingdock':('holdout-loadingdock-timber-high-right',('control','skip_edges','skip_repeat_choice'))}


def grade(root):
    jobs={j[1]['id']:j for j in typed.SUITE.jobs(['fresh','new'])}
    results=[]
    for label,(key,methods) in EXAMPLES.items():
        job=jobs[key]; split,frame,_,_=job
        _,_,transform,_=ref_inputs(job)
        labels,pixels=visible_faces(split,frame)
        truth={tuple(r['xyz_world']):r['block_id'] for r in labels['blocks']}
        optional=horizontal_slivers(labels,pixels)
        results_by_mode={}
        for mode in methods:
            path=root/(label+'-'+mode)
            manifest=read(path/'build-manifest.json')
            if manifest['image_sha256']!=frame['files']['rgb.png']:
                raise ValueError(f'Mismatched raw screenshot for {key}')
            actual=decoded_pack(path/'reconstruction.zip')
            rows=read(path/'placements.json')['blocks']
            ids={tuple(r[a] for a in 'xyz'):r['block_id'] for r in rows}
            evidence={tuple(r[a] for a in 'xyz'):r['material_source'] for r in rows}
            if actual!={p:'minecraft:'+v for p,v in ids.items()}:
                raise ValueError(f'Export and manifest disagree for {key}')
            score,_=metrics(ids,evidence,transform,truth,optional)
            results_by_mode[mode]=dict(score,ordinary_pack_sha256=digest(path/'reconstruction.zip'),
                                       final_refinement=manifest['placement']['final_refinement'])
        baseline=exp.published()[1][key]
        results.append(dict(id=key,source_sha256=frame['files']['rgb.png'],methods=results_by_mode,
                            published_pack_sha256=baseline['pack_sha256'],
                            published_geometry=baseline['geometry'],published_material=baseline['result']))
    typed.compact(exp.OUT/'full-entry-bypasses.json',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=exp.HERE/'runs/full-bypass')
    grade(parser.parse_args().root)
