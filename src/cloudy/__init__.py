from ._common import Pipeline, cloud_transform, latent_transform
from ._modeling import sample_slice

try:
    from ._rendering import (
        camera_poses,
        oct_camera_poses,
        transmittance,
        scattered,
        accumulate,
        save_video
    )
except:
    pass


def create_pipeline(workspace: str = '.', **settings):
    return Pipeline(workspace, **settings)