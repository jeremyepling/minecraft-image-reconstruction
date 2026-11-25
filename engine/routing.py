"""Measure spatial agreement with the exposed match before trusting a whole-cell crop."""
import numpy as np

from calibration.retrieval import distances, exposure_distance, features


def alignment_scores(crops, train, labels, method):
    """Return exposure ranking and spatial agreement with exemplars of the winner."""
    order = np.argsort(labels, kind='stable')
    names, starts = np.unique(labels[order], return_index=True)
    table = []
    evidence = []
    for start in range(0,len(crops),48):
        query = features(crops[start:start+48])
        baseline = distances(query,train)[method]
        adjusted = exposure_distance(query,train,method,baseline)
        d = np.minimum.reduceat(adjusted[:,order],starts,axis=1)
        for i, row in enumerate(d):
            indices = np.argsort(row,kind='stable')
            target = indices[0]
            exemplar = np.flatnonzero(labels==names[target])
            # Compare the actual winning template, never a second version
            # admitted solely on shape. Pearson correlation resists shadows.
            best = exemplar[np.argmin(adjusted[i,exemplar])]
            a = query[1][i].reshape(14,14,3).mean(axis=2).ravel()
            b = train[1][best].reshape(14,14,3).mean(axis=2).ravel()
            variance = np.linalg.norm(a)*np.linalg.norm(b)
            agreement = np.dot(a,b)/max(float(variance),1e-6)
            table.append((names[indices[:5]].tolist(),row[indices[:5]].tolist(),
                          float((row[indices[1]]-row[target])/max(row[indices[1]],1e-6))))
            evidence.append(float(agreement))
    return table,np.asarray(evidence)


def trusted_cells(evidence, face_width_px=None):
    """A plane with many strongly aligned faces supports a common grid phase."""
    # A sub-texel screenshot suppresses spatial detail. Keep the calibrated
    # retriever when several tiny faces still independently support the grid.
    tiny = face_width_px is not None and np.median(face_width_px)<12
    minimum = .04 if tiny else .20
    return (evidence >= .8) | (np.mean(evidence >= .8) >= minimum)
