import os.path

import matplotlib.pyplot as plt
import torch
import cloudy
import numpy as np
from tqdm import tqdm
from vulky import datasets


pipeline = cloudy.create_pipeline('./test')
# pipeline.download_pretrained()

camera_position = (np.cos(0.5) * 2.7, np.sin(0.5), np.sin(0.5) * 2.7)
camera_poses = cloudy.camera_poses(camera_position)
environment = datasets.Images.environment_example.to('cuda') # torch.zeros(64, 128, 3, device='cuda')
# environment[10, 0] = 2000

settings = dict(
    camera_position = camera_position,
    environment = environment,
    density_scale = 300,
    phase_g = 0.0,
    scattering_albedo = (0.99, 0.98, 0.94),
)


def render_grid(g: torch.Tensor, width: int = 512, height: int = 512, samples: int = 128):
    with torch.no_grad():
        return cloudy.scattered(torch.clamp(g, 0, 1) * settings['density_scale'],
                         camera_poses=camera_poses,
                         majorant=g.max().item() * settings['density_scale'],
                         scattering_albedo=settings['scattering_albedo'],
                         phase_g=settings['phase_g'],
                         environment=environment,
                         width=width,
                         height=height,
                         jittered=True,
                         samples=samples
                         )[0]



def create_single_view_criteria(reference_image: torch.Tensor, **settings):
    camera_poses = cloudy.camera_poses(settings['camera_position'])
    environment_tensor = settings['environment']
    environment, environment_sampler = cloudy.environment_objects(environment_tensor)
    density_scale = settings['density_scale']
    phase_g = settings['phase_g']
    scattering_albedo = settings['scattering_albedo']

    ema_image = None

    def criteria(latent: torch.Tensor)->torch.Tensor:
        g = pipeline.from_latent_to_grid(latent)
        # g = g * (g >= 0.0).float()
        g = cloudy.dclamp(g, 0.0, .8)

        inf_image = cloudy.scattered(g * density_scale,
                         camera_poses=camera_poses,
                         majorant=g.max().item() * density_scale,
                         scattering_albedo=scattering_albedo,
                         phase_g=phase_g,
                         environment=environment,
                         environment_sampler=environment_sampler,
                         width=reference_image.shape[1],
                         height=reference_image.shape[0],
                         jittered=True,
                         samples=64,
                         bw_samples=8,
                         )[0]

        nonlocal ema_image
        if ema_image is not None:
            inf_image = cloudy.ema_diff(inf_image, ema_image, 0.2)
        loss = ((inf_image - reference_image)**2).sum()
        ema_image = inf_image.detach()
        return loss
    return criteria


### Loading reference volume

ref_latent = pipeline.get_test_latent(8)

with torch.no_grad():
    ref_rep = pipeline.from_latent_to_rep(ref_latent)
    ref_grid = pipeline.from_rep_to_grid(ref_rep, 128)
    ref_grid = ref_grid * (ref_grid >= 0.0).float()

plt.imshow(ref_grid[:, :, ref_grid.shape[2]//2, 0].cpu().T, cmap='gist_heat_r', vmin=0.0, vmax=0.6)
plt.gca().invert_yaxis()
plt.show()


### Generating reference view

if os.path.exists('./reference_image.pt'):
    reference_image = torch.load('./reference_image.pt', map_location='cuda', weights_only=True)
else:
    with torch.no_grad():
        reference_image = cloudy.accumulate(lambda: cloudy.scattered(ref_grid * settings['density_scale'],
                              camera_poses=camera_poses,
                              scattering_albedo=settings['scattering_albedo'],
                              environment=settings['environment'],
                              phase_g=settings['phase_g'],
                              majorant=ref_grid.max()*settings['density_scale'],
                              # kwargs
                              width=512,
                              height=512,
                              jittered=True,
                              samples=128
                              ), times=32)
    torch.save(reference_image, './reference_image.pt')
reference_image = reference_image[0]

plt.figure(figsize=(1, 1), dpi=512)
plt.imshow(reference_image.cpu() ** (1.0/2.2))
plt.gca().invert_yaxis()
plt.gca().axis('off')
plt.tight_layout(pad=0.0)
plt.show()


### Storing rendered images for intermediate states
images = []
def on_new_state2(i, latent):
    with torch.no_grad():
        rep = pipeline.from_latent_to_rep(latent)
        grid = pipeline.from_rep_to_grid(rep, 128)
        grid = grid * (grid >= 0.0).float()
        im = render_grid(grid,
                         width=reference_image.shape[1],
                         height=reference_image.shape[0],
                         samples=max(1, 64 * (1000 - i) // 1000))
        fr = torch.zeros_like(im)
        h = reference_image.shape[0] // 4
        fr[h*2:] = reference_image[h:-h]
        fr[:h*2] = im[h:-h]
        images.append(fr)
        print(f"Generated image for step {i}")
        if np.random.rand() < 0.125:
            plt.figure(figsize=(1,1), dpi=512)
            plt.imshow(latent.cpu()[:,:,[8, 16, 24]]*2.0+0.5)
            plt.gca().axis('off')
            plt.tight_layout(pad=0.0)
            plt.show()

            plt.figure(figsize=(1,1), dpi=512)
            plt.imshow(fr.cpu() ** (1.0/2.2))
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            plt.tight_layout(pad=0.0)
            plt.show()


def on_new_state(i, latent):
    # if np.random.rand() < 0.9:
    #     return
    with torch.no_grad():
        rep = pipeline.from_latent_to_rep(latent)
        grid = pipeline.from_rep_to_grid(rep, 128)
        grid = torch.clamp(grid, 0., 0.8)
        im = render_grid(grid,
                         width=reference_image.shape[1],
                         height=reference_image.shape[0],
                         samples=max(1, 64 * (1000 - i) // 1000))
        fr = torch.zeros_like(im)
        h = reference_image.shape[0] // 4
        fr[h*2:] = reference_image[h:-h]
        fr[:h*2] = im[h:-h]
        images.append(fr)
        # plt.figure(figsize=(1,1), dpi=512)
        # plt.imshow(latent.cpu()[:,:,[8, 16, 24]]*2.0+0.5)
        # plt.gca().axis('off')
        # plt.tight_layout(pad=0.0)
        # plt.show()
        # plt.figure(figsize=(1,1), dpi=512)
        # plt.imshow(fr.cpu() ** (1.0/2.2))
        # plt.gca().axis('off')
        # plt.gca().invert_yaxis()
        # plt.tight_layout(pad=0.0)
        # plt.show()


latent = pipeline.generate_latent(
    samples=300,
    scheduler_gamma=.4,
    criteria=create_single_view_criteria(reference_image, **settings),
    criteria_scale=2.0,
    callback=on_new_state
)

with torch.no_grad():
    rep = pipeline.from_latent_to_rep(latent)
    grid = pipeline.from_rep_to_grid(rep, 256)
    grid = torch.clamp(grid, 0., 0.8)

plt.imshow(grid[:, :, grid.shape[2]//2, 0].cpu().T, cmap='gist_heat_r', vmin=0.0, vmax=0.6)
plt.gca().invert_yaxis()
plt.show()


with torch.no_grad():
    im = cloudy.accumulate(lambda: render_grid(grid,
                     width=512,
                     height=512,
                     samples=64
                     ), 16)
    plt.figure(figsize=(1, 1), dpi=512)
    plt.imshow(im.cpu() ** (1.0/2.2))
    plt.gca().invert_yaxis()
    plt.gca().axis('off')
    plt.tight_layout(pad=0.0)
    plt.show()


with torch.no_grad():
    FRAMES = 100
    frames = torch.zeros(100 + len(images), 512, 512, 3)

    for i, im in enumerate(images):
        frames[i] = torch.flip(im, dims=[0])

    for f in tqdm(range(FRAMES), desc="Generating video"):
        alpha = f/(FRAMES - 1)
        camera_poses = cloudy.camera_poses((np.cos(0.5 + alpha*6.28)*2.7, np.sin(0.5 + alpha*6.28), np.sin(0.5 + alpha*6.28)*2.7))
        ims = cloudy.scattered(grid * 300,
                          camera_poses=camera_poses,
                          scattering_albedo=(0.99, 0.98, 0.94),
                          environment=environment,
                          phase_g=0.0,
                          majorant=grid.max()*300,
                          # kwargs
                          width=512,
                          height=512,
                          jittered=True,
                          samples=128
                          )[0]
        imc = cloudy.scattered(ref_grid * 300,
                              camera_poses=camera_poses,
                              scattering_albedo=(0.99, 0.98, 0.94),
                              environment=environment,
                              phase_g=0.0,
                              majorant=ref_grid.max() * 300,
                              # kwargs
                              width=512,
                              height=512,
                              jittered=True,
                              samples=128
                              )[0]
        im = torch.zeros_like(ims)
        im[256:] = imc[128:-128]
        im[:256] = ims[128:-128]
        frames[f + len(images)] = torch.flip(im, dims=[0])
    cloudy.save_video(frames, 'generated_cloud_single_view.webp')