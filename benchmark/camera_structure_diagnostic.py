"""Force the first tentative ground fit per frame through the ordinary build for diagnosis."""
from concurrent.futures import ProcessPoolExecutor
from benchmark.camera_structure_run import BASE,COHORTS,ROOT,REPORT,SOURCE,paths,frames,read,save
import sys


def measure(entry):
    sys.path.append(str(ROOT.parent/'camera-horizon-floor/benchmark'))
    from camera_horizon_replay import replay
    base,out=paths(entry)
    option=read(out/'source.json.gz')['tentative'][0]
    camera=dict(P=option['P'],reason='diagnostic force first tentative wall fit',
                camera_source='scaffold-diagnostic',scale_status='supported')
    case=replay(entry,camera,base,out/'forced')
    case['kind']=option['kind']
    save(out/'forced-grade.json',case)
    return case


if __name__=='__main__':
    entries=[e for e in frames(SOURCE,COHORTS) if (paths(e)[1]/'source.json.gz').exists()
             and read(paths(e)[1]/'source.json.gz').get('tentative')]
    with ProcessPoolExecutor(2) as pool:
        cases=list(pool.map(measure,entries))
    save(REPORT/'forced-tentative.json',cases)
