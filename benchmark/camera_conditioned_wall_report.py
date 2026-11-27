"""Summarize and verify the decoded packs from the frozen depth-gate pilot."""
from collections import Counter, defaultdict
from benchmark.camera_conditioned_wall_separation import OUTPUT, CASES
from benchmark.camera_model_controls import inventory_hash
from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.expanded_scores import pooled
from benchmark.mask_blocks import decoded_pack

def main():
    results = []
    cohorts = defaultdict(list)
    for path in sorted((OUTPUT / 'scores').glob('*-depth.json')):
        row = read(path)
        before = row['baseline']
        pack = OUTPUT / 'runs/wall-color-partition-01' / row['id'] / 'depth/reconstruction.zip'
        assert digest(pack) == row['pack_sha256']
        decoded = inventory_hash(decoded_pack(pack))
        assert before['source_sha256'] == row['source_sha256']
        proposals = [child for p in row['handoff'] for child in p['children'] if 'proposed' in child]
        moved = sum(c['level'] != p['initial'] for p in row['handoff'] for c in p['children'])
        if not moved:
            assert before['decoded_sha256'] == decoded, row['id']
        outcomes = Counter()
        for c in proposals:
            if min(c['interior'], c['neighbor']) < 16:
                reason = 'small_interior'
            elif 'offset' not in c:
                reason = 'unusable_slope'
            elif c['offset'] * c['predicted_change'] <= 0:
                reason = 'wrong_direction'
            elif abs(c['offset']) <= 2*c['noise']:
                reason = 'offset_within_noise'
            elif c['residual_gain'] <= c['noise']:
                reason = 'insufficient_residual_gain'
            else:
                reason = 'accepted'
            outcomes[reason] += 1
        assert moved == outcomes['accepted']
        result = {k: row[k] for k in ('id','split','dataset','source_sha256','pack_sha256','depth_source','paired','geometry','scores')}
        result.update(prior_pack=before['pack_sha256'], decoded=decoded, prior_decoded=before['decoded_sha256'],
                      proposed=len(proposals), moved=moved, decisions=outcomes,
                      handoff_added=sum(h['added'] for h in row['handoff']),
                      handoff_removed=sum(h['removed'] for h in row['handoff']),
                      prior_geometry=before['geometry'], prior_scores=before['scores'])
        results.append(result)
        cohorts[row['dataset']].append(row)
    summary = {}
    for dataset, group in cohorts.items():
        n = {'visible-builds-v3':144, 'decorative-builds-v1':40}.get(dataset,104)
        summary[dataset] = dict(prior=pooled([r['baseline'] for r in group],n), candidate=pooled(group,n))
    save(OUTPUT/'summary.json', dict(branch='experiment/camera-conditioned-wall-separation',base='e3bd220',
         current_camera_cases=digest(CASES), run_code=digest(ROOT/'benchmark/camera_conditioned_wall_separation.py'),
         trial_code=digest(ROOT/'benchmark/wall_color_partition.py'), cases=results, cohorts=summary))
    for r in results:
        print(r['id'],r['proposed'],r['moved'],r['paired'],flush=True)

if __name__ == '__main__':
    main()
