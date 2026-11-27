"""Select a metric camera from seams, texels, then verified floor tile projections."""
from engine import camera_atlas_floor, camera_native_floor, camera_repeat_floor, camera_seam


def predict(image, version='26.3', *, atlas_path=None):
    seams = camera_seam.predict(image, version)
    if seams['P'] is not None:
        return dict(seams, camera_source='seam-fit')
    atlas = camera_atlas_floor.predict(image, version, atlas_path=atlas_path)
    if atlas['P'] is not None:
        return dict(atlas, camera_source='atlas-floor', seam_reason=seams['reason'])
    repeat = camera_repeat_floor.predict(image, version, atlas_path=atlas_path)
    if repeat['P'] is not None:
        return dict(repeat, seam_reason=seams['reason'], atlas_reason=atlas['reason'])
    native = camera_native_floor.predict(image, version, atlas_path=atlas_path)
    if native['P'] is not None:
        return dict(native, seam_reason=seams['reason'], atlas_reason=atlas['reason'], repeat_reason=repeat['reason'])
    return dict(seams, camera_source=None, atlas_reason=atlas['reason'], repeat_reason=repeat['reason'],
                native_floor_reason=native['reason'], native_floor_evidence=native['evidence'])
