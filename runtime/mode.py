"""Keep screenshot-only commands and their child processes out of the Minecraft runtime."""
from contextlib import contextmanager
import os


IMAGE_ONLY = 'MCIR_IMAGE_ONLY'


def mark_image_only():
    """Also used as the initializer of each scoring worker."""
    os.environ[IMAGE_ONLY] = '1'


@contextmanager
def image_only():
    previous = os.environ.get(IMAGE_ONLY)
    mark_image_only()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(IMAGE_ONLY, None)
        else:
            os.environ[IMAGE_ONLY] = previous


def require_game_runtime(operation):
    if os.environ.get(IMAGE_ONLY) == '1':
        raise RuntimeError(f'Image-only run cannot start Minecraft for {operation}. '
                           'Use the saved fixture files for this comparison.')
