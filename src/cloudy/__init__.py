from ._common import Pipeline, Recorder, cloud_transform, latent_transform, CallbackInfo
from ._modeling import (
    sample_slice,
    ema_diff,
    dclamp,
    reconstruct_grid3d,
    Volume,
    resampling,
    resample_grid,
    sample_grid2d,
    sample_grid2d_batch,
    sample_grid3d,
    sample_kplanar,
    total_variation_2D,
    total_variation_3D,
    total_variation_2D_abs,
    gaussian_filter
)
"""
# try:
from ._rendering import (
    camera_poses,
    environment_objects,
    oct_camera_poses,
    transmittance,
    scattered,
    scattered_environment,
    reconstruct_environment,
    background_radiance,
    accumulate,
    display_postprocess,
    gamma_correction,
    log_tone_mapping,
    save_video
)
# except Exception as e:
#     print(e)
#     pass

"""
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
