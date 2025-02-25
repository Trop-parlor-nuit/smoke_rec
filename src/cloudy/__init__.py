from ._common import Pipeline, cloud_transform, latent_transform
from ._modeling import sample_slice, ema_diff, dclamp

try:
    from ._rendering import (
        camera_poses,
        environment_objects,
        oct_camera_poses,
        transmittance,
        scattered,
        accumulate,
        save_video
    )
except:
    pass


def create_pipeline(workspace: str = '.', **settings) -> Pipeline:
    """
    Creates a pipeline object bound to a specific folder as workspace.

    Paramters
    ---------
    workspace: str
        Folder path that will be used to store intermediate results, decoder, encoded latents, enhanced latents.
    settings: Dict
        Update values for default settings. If a value is a dict, replacement will only update the new keys. For example
        use `create_pipeline('./test', decoder=dict(train_steps=40_000))` to change the train_steps for the decoder training.
    """
    return Pipeline(workspace, **settings)