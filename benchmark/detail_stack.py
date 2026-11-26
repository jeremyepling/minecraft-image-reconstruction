"""Evaluate whether frozen Planarity additions improve the photo-supported boundary."""
from benchmark import edge_detail, plane_detail


SETTINGS = {'photo-then-plane': ('native4-count-guard','add2')}


def predict(photo, parts, frame_sha, setting):
    photo_setting, plane_setting = SETTINGS[setting]
    refined = edge_detail.predict(photo,parts,frame_sha,photo_setting)
    return plane_detail.predict(photo,refined,frame_sha,plane_setting)


def provenance(setting):
    photo_setting, plane_setting = SETTINGS[setting]
    return dict(photo=edge_detail.provenance(photo_setting),plane=plane_detail.provenance(plane_setting),
                field_manifests=plane_detail.provenance(plane_setting)['field_manifests'])
