"""Record which neighboring cells share texture across a change in brightness."""
import numpy as np
from engine.geometry import face
from engine.illumination import MATERIAL_THRESHOLD, edge_evidence


def boundary_evidence(rgb,cells):
    lookup={(c['plane'],*c['ij']):i for i,c in enumerate(cells) if c.get('facing')=='south'}
    pairs=[]
    for (plane,x,y),i in lookup.items():
        for key in ((plane,x+1,y),(plane,x,y+1)):
            if key in lookup:
                pairs.append((i,lookup[key]))
    rows=[]
    for start in range(0,len(pairs),128):
        batch=pairs[start:start+128]
        crops={i:face(rgb,cells[i]['quad'],16) for pair in batch for i in pair}
        evidence=edge_evidence(np.stack([crops[i] for i,_ in batch]),np.stack([crops[j] for _,j in batch]))
        for k,(i,j) in enumerate(batch):
            score=float(evidence['material_score'][k])
            if not evidence['observable'][k]:
                label='ambiguous'
            elif score>=MATERIAL_THRESHOLD:
                label='material_change_likely'
            elif abs(evidence['log_gain'][k])>.1:
                label='consistent_with_lighting'
            else:
                label='similar_appearance'
            rows.append(dict(cell_indices=[i,j],assessment=label,material_score=round(score,4),
                             shading_field=str(evidence['shading_model'][k]),
                             log_gain=round(float(evidence['log_gain'][k]),4)))
    return rows
