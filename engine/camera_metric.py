"""Select a metric camera using visible ground seams, then registered floor art."""
from engine import camera_atlas_floor, camera_seam


def predict(image, version='26.3', *, atlas_path=None):
    seams = camera_seam.predict(image, version)
    if seams['P'] is not None:
        return dict(seams, camera_source='seam-fit')
    atlas = camera_atlas_floor.predict(image, version, atlas_path=atlas_path)
    if atlas['P'] is not None:
        return dict(atlas, camera_source='atlas-floor', seam_reason=seams['reason'])
    return dict(seams, camera_source=None, atlas_reason=atlas['reason'])
