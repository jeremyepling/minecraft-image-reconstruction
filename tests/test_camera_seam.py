import unittest

import numpy as np
from PIL import Image

from engine.camera_seam import predict


def pavement_image(remove_one_family=False):
    """Analytic pinhole projection with a clipped far plane and dark unit seams."""
    w, h, focal = 720, 460, 330
    pitch, yaw = np.radians([9, 23])
    right = np.array([np.cos(yaw), 0., -np.sin(yaw)])
    forward = np.array([np.sin(yaw) * np.cos(pitch), -np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    up = np.cross(forward, right)
    eye = np.array([0.37, 6.42, 0.13])
    y, x = np.mgrid[:h, :w]
    rays = (forward + ((x + 0.5 - w / 2) / focal)[:, :, None] * right -
            ((y + 0.5 - h / 2) / focal)[:, :, None] * up)
    distance = -eye[1] / rays[:, :, 1]
    xyz = eye + rays * distance[:, :, None]
    offset = np.abs(xyz[:, :, (0, 2)] - np.round(xyz[:, :, (0, 2)]))
    seam_distance = offset[:, :, 0] if remove_one_family else np.min(offset, axis=2)
    gray = np.where(seam_distance < 1 / 16, 112, 163)
    rgb = np.where(((distance > 0) & (distance < 150))[:, :, None],
                   np.repeat(gray[:, :, None], 3, axis=2), [107, 169, 230])
    # A foreground object masks the far ground and overlaps the estimation band.
    rgb[100:330, 230:490] = [82, 86, 91]
    return Image.fromarray(rgb.astype("uint8")), focal, eye[1]


class SeamCameraTests(unittest.TestCase):
    def test_metric_recovery_from_oblique_seams(self):
        image, focal, height = pavement_image()
        result = predict(image)
        self.assertEqual(result["scale_status"], "supported")
        self.assertLess(abs(result["evidence"]["focal_px"] - focal), 4)
        self.assertLess(abs(result["evidence"]["height_blocks"] - height), 0.08)

    def test_blank_ground_abstains(self):
        self.assertIsNone(predict(Image.new("RGB", (300, 200), "gray"))["P"])

    def test_single_family_does_not_establish_metric_camera(self):
        image, _, _ = pavement_image(remove_one_family=True)
        self.assertIsNone(predict(image)["P"])
