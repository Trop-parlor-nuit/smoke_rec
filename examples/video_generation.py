import matplotlib.pyplot as plt
import torch
import cloudy
import numpy as np
from tqdm import tqdm



pipeline = cloudy.create_pipeline('./test')
# pipeline.download_pretrained()


camera_poses = cloudy.camera_poses((np.cos(0.5) * 2.7, np.sin(0.5), np.sin(0.5) * 2.7))
environment = torch.zeros(64, 128, 3, device='cuda')
environment[10, 0] = 2000

images = []

grid = None

def on_new_state(i, latent):
    global grid

    with torch.no_grad():
        print(f"Generated image for step {i}")
        rep = pipeline.from_latent_to_rep(latent)
        grid = pipeline.from_rep_to_grid(rep, 128)
        grid = grid * (grid >= 0.0).float()
        im = cloudy.scattered(grid * 300,
                              camera_poses=camera_poses,
                              scattering_albedo=(0.99, 0.98, 0.94),
                              environment=environment,
                              phase_g=0.0,
                              majorant=grid.max() * 300,
                              # kwargs
                              width=512,
                              height=512,
                              jittered=True,
                              samples=max(1, 128 * (1000 - i) // 1000)
                              )[0]
        images.append(im)
        if np.random.rand() < 0.125:
            plt.figure(figsize=(1, 1), dpi=512)
            plt.imshow(latent.cpu()[:, :, [8, 16, 24]] + 0.5)
            plt.gca().axis('off')
            plt.tight_layout(pad=0.0)
            plt.show()

            plt.figure(figsize=(1, 1), dpi=512)
            plt.imshow(im.cpu() ** (1.0 / 2.2))
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            plt.tight_layout(pad=0.0)
            plt.show()


latent = pipeline.generate_latent(samples=50, callback=on_new_state)

with torch.no_grad():
    rep = pipeline.from_latent_to_rep(latent)
    grid = pipeline.from_rep_to_grid(rep, 256)
    grid = grid * (grid >= 0.0).float()

plt.imshow(grid[:, :, grid.shape[2]//2, 0].cpu().T, cmap='gist_heat_r', vmin=0.0, vmax=0.6)
plt.gca().invert_yaxis()
plt.show()


with torch.no_grad():
    im = cloudy.scattered(grid * 300,
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
                          )
    plt.figure(figsize=(1, 1), dpi=512)
    plt.imshow(im[0].cpu() ** (1.0/2.2))
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
        im = cloudy.scattered(grid * 300,
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
        frames[f + len(images)] = torch.flip(im, dims=[0])
    cloudy.save_video(frames, 'generated_cloud.webp')