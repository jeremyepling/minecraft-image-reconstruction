from contextvars import ContextVar
from functools import wraps

# Private to this benchmark. An unset value preserves the ordinary ground policy.
scene_distance = ContextVar('floating_scene_distance', default=None)


def active():
    return scene_distance.get() is not None


def ranked_policy(function):
    @wraps(function)
    def run(image, version, parts, counts, camera, **options):
        distance = float(camera['build_grid_distance']) if camera.get('floating_grid') else None
        token = scene_distance.set(distance)
        try:
            return function(image, version, parts, counts, camera, **options)
        finally:
            scene_distance.reset(token)
    return run


def roof_heights(prior, cap, margin):
    low = min(cell[1] for cell in prior) if active() else 0
    return range(low + 1, min(low + cap, max(cell[1] for cell in prior) + margin) + 1)
