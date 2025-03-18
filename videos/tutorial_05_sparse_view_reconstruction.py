import matplotlib.pyplot as plt

import cloudy
import numpy as np
import torch
import vulky.datasets as datasets
import os


pipeline = cloudy.create_pipeline('./test')
# pipeline.download_pretrained()

recorder = pipeline.create_recorder()

# General settings for the rendering
settings = dict(
    camera_positions=[
        [np.cos(0.5) * 2.7, np.sin(0.5), np.sin(0.5) * 2.7],
        [np.cos(0.5 + 1) * 2.7, -2.0, np.sin(0.5 + 1) * 2.7],
        [np.cos(0.5 + 2) * 2.7, -2.0, np.sin(0.5 + 2) * 2.7]
    ],
    environment = datasets.Images.environment_example.to('cuda'),
    density_scale = 300,
    phase_g = 0.0,
    scattering_albedo = (0.99, 0.98, 0.94),
)

environment, environment_sampler = cloudy.environment_objects(settings['environment'])
env_id = recorder.add_environment(settings['environment'])

ref_latent = pipeline.get_test_latent(8)

def render_grid(g: torch.Tensor, samples: int = 128):
    return cloudy.scattered(g * settings['density_scale'],
                     camera_poses=cloudy.camera_poses(torch.tensor(settings['camera_positions'], dtype=torch.float32)),
                     scattering_albedo=settings[
                         'scattering_albedo'],
                     environment=environment,
                     environment_sampler=environment_sampler,
                     phase_g=settings['phase_g'],
                     majorant=g.max() * settings[
                         'density_scale'],
                     # kwargs
                     width=512,
                     height=512,
                     jittered=True,
                     samples=samples,
                     samples_bw=8,
                     mode='sps'
                     )

with torch.no_grad():
    ref_grid = pipeline.decode_latent(ref_latent)
    ref_grid = pipeline.clean_volume(ref_grid)
    # Generate reference image
    if os.path.exists('./reference_image3.pt'):
        print("Loading cached reference_image3.pt")
        reference_image = torch.load('./reference_image3.pt', map_location='cuda', weights_only=True)
        plt.imshow(reference_image[2].cpu()**(1.0/2.2))
        plt.show()
    else:
        print("Creating cached reference_image3.pt")
        reference_image = cloudy.accumulate(lambda: render_grid(ref_grid), times=16)
        print("Saved cached reference_image3.pt")
        torch.save(reference_image, './reference_image3.pt')

ref_capture_id = recorder.add_capture_image(reference_image[0])

torch.manual_seed(101)

grid = pipeline.sample_volume(
    resolution=128,
    samples=100,
    scheduler_gamma=1.0,
    y=reference_image,
    A=lambda v: render_grid(v, samples=32),
    ema_factor=0.2,
    guiding_strength=1.0,
    callback=lambda ci: recorder.new_keyframe(
        ref_capture_id,
        recorder.add_capture_latent(
            ci.latent,
            render_mode='ms',
            environment=env_id,
            density_scale=settings['density_scale'],
            scattering_albedo=settings['scattering_albedo'],
            camera_position=tuple(settings['camera_positions'][0]),
            phase_g=settings['phase_g'],
            samples=max(1, 128*ci.step//ci.total_steps)
        )
    ))

recorder.show_clip(1, width=512, height=512, samples_multiplier=4)
# exit()
# recorder.save_video('test.webp', samples_multiplier=32)

ref_vol_id = recorder.add_volume(ref_grid)
smp_vol_id = recorder.add_volume(grid)

for i in range(100):
    alpha = 3.141593 * 2 * i / 99
    camera_position = (np.cos(0.5+alpha)*2.7, np.sin(0.5+alpha), np.sin(0.5+alpha)*2.7)
    recorder.new_keyframe(
        recorder.add_capture_volume(
            ref_vol_id,
            render_mode='ms',
            environment=env_id,
            density_scale=settings['density_scale'],
            scattering_albedo=settings['scattering_albedo'],
            camera_position=camera_position,
            phase_g=settings['phase_g'],
            samples=128
        ),
        recorder.add_capture_volume(
            smp_vol_id,
            render_mode='ms',
            environment=env_id,
            density_scale=settings['density_scale'],
            scattering_albedo=settings['scattering_albedo'],
            camera_position=camera_position,
            phase_g=settings['phase_g'],
            samples=128
        )
    )

recorder.save_video('tutorial_05_sparse_view_reconstruction.webp', samples_multiplier=4)
