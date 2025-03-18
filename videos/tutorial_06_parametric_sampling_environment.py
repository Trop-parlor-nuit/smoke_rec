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
    camera_position = (np.cos(0.5) * 2.7, np.sin(0.5), np.sin(0.5) * 2.7),
    environment = datasets.Images.environment_example.to('cuda'),
    density_scale = 300,
    phase_g = 0.0,
    scattering_albedo = (0.99, 0.98, 0.94),
)

environment, environment_sampler = cloudy.environment_objects(settings['environment'])
env_id = recorder.add_environment(settings['environment'])

ref_latent = pipeline.get_test_latent(8)

def render_grid(g: torch.Tensor, environment, environment_sampler, samples: int = 128):
    return cloudy.scattered(g * settings['density_scale'],
                     camera_poses=cloudy.camera_poses(settings['camera_position']),
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
    if os.path.exists('./reference_image.pt'):
        print("Loading cached reference_image.pt")
        reference_image = torch.load('./reference_image.pt', map_location='cuda', weights_only=True)
    else:
        reference_image = cloudy.accumulate(lambda: render_grid(ref_grid, environment, environment_sampler), times=16)
        torch.save(reference_image, './reference_image.pt')

ref_capture_id = recorder.add_capture_image(reference_image[0])

# environment parameters
sun_trainable = torch.nn.Parameter(torch.ones(16, 32, 3, device=pipeline.get_device())*-3.0, requires_grad=True)
sky_estimation = reference_image[0][-30:].mean(dim=0).mean(dim=0)
environment_trainable = torch.nn.Parameter(torch.ones(2, 1, 3, device=pipeline.get_device())*-1.0, requires_grad=True)
with torch.no_grad():
    environment_trainable[0, 0] = torch.log(sky_estimation / (1 - sky_estimation))

# environment map model
def compute_environment_map():
    sky = cloudy.resampling(torch.sigmoid(environment_trainable).unsqueeze(0), (16, 32), mode='nearest-exact', align_corners=None)[0].contiguous()
    sun = torch.exp(sun_trainable) # modeling.gaussian_filter(torch.exp(sun_trainable).unsqueeze(0), sigma=1, kernel_size=3)[0].contiguous()  # let always a chance to sample
    return sun + sky


def create_A(samples: int = 64):
    with torch.no_grad():
        environment_tensor = compute_environment_map()
    environment, environment_sampler = cloudy.environment_objects(environment_tensor)
    return lambda g: render_grid(g, environment, environment_sampler, samples=samples)


def create_L(g: torch.Tensor, y: torch.Tensor, samples: int = 64):
    density_scale = settings['density_scale']
    phase_g = settings['phase_g']
    scattering_albedo = settings['scattering_albedo']
    camera_poses = cloudy.camera_poses(settings['camera_position'])
    with torch.no_grad():
        g = g * density_scale
    def L():
        yhat = cloudy.scattered_environment(
             environment=compute_environment_map(),
             grid=g,
             camera_poses=camera_poses,
             majorant=g.max().item(),
             scattering_albedo=scattering_albedo,
             phase_g=phase_g,
             width=512,
             height=512,
             jittered=True,
             samples=samples,
             bw_samples=max(1, samples // 8),
             )
        loss = torch.nn.functional.mse_loss(yhat, y, reduction='sum')
        sun = torch.exp(sun_trainable)
        loss += 0.001 * cloudy.total_variation_2D(sun.unsqueeze(0))
        loss += 0.001 * sun.abs().sum()
        return loss
    return L


def callback(ci: cloudy.CallbackInfo):
    with torch.no_grad():
        if ci.subpass == 'dps':
            if ci.step % 10 != 0:
                return
            samples = max(1, 128 * ci.step // ci.total_steps)
        else:
            if ci.step % 20 != 0:
                return
            samples = 128
        cap_id = recorder.add_capture_latent(
            ci.latent,
            compute_environment_map(),
            settings['density_scale'],
            settings['scattering_albedo'],
            settings['phase_g'],
            settings['camera_position'],
            samples = samples,
            render_mode='ms'
        )
        recorder.new_keyframe(
            ref_capture_id,
            cap_id
        )


# create optimizer
opt = torch.optim.NAdam([sun_trainable, environment_trainable], lr=0.01)

torch.manual_seed(101)

grid = pipeline.reconstruct_volume(
    y = reference_image,
    A_factory = lambda p: create_A(),
    L_factory = lambda p, grid, y: create_L(grid, y),  # no physical parameter unknown
    optimizer=opt,
    resolution=128,
    ema_factor=0.2,
    samples=100,
    scheduler_gamma=.6,
    weights=[0.2, 0.4, 0.6, 0.8, 1.0],
    decoding_resolution=[32, 32, 64, 64, 128],
    decoding_noise=[1.0, 1.0, 1.0, 1.0, 0.0],
    optimization_steps=200,
    optimization_passes=5,
    callback=callback
)


recorder.show_clip(2, 6, width=512, height=512, samples_multiplier=1)
# recorder.save_video('test.webp', samples_multiplier=32)

ref_vol_id = recorder.add_volume(ref_grid)
smp_vol_id = recorder.add_volume(grid)
rec_env_id = recorder.add_environment(compute_environment_map().detach())

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
            environment=rec_env_id,
            density_scale=settings['density_scale'],
            scattering_albedo=settings['scattering_albedo'],
            camera_position=camera_position,
            phase_g=settings['phase_g'],
            samples=128
        )
    )

recorder.save_video('tutorial_06_parametric_sampling_environment.webp', samples_multiplier=4)
