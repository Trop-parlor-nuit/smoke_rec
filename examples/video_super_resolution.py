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

def downsample_grid(g: torch.Tensor):
    cg = torch.nn.functional.interpolate(
        g.unsqueeze(0).permute(0, 4, 1, 2, 3),
        scale_factor=0.5,
        mode='trilinear',
        align_corners=True
    )
    return cg.permute(0, 2, 3, 4, 1)[0]


def coarser_grid(g: torch.Tensor, levels: int):
    for _ in range(levels):
        g = downsample_grid(g)
    return g


def create_super_resolution_criteria (coarse_grid: torch.Tensor):
    p = int(np.log2(coarse_grid.shape[0]))
    assert coarse_grid.shape[0] == (1 << p), "coarse_grid shape[0] must be power of 2"
    assert coarse_grid.shape[0] == coarse_grid.shape[2] and coarse_grid.shape[0] == coarse_grid.shape[1] * 2, "Dimensions should be Nx(N//2)xN"
    assert coarse_grid.shape[0] <= 128, "coarse grid shape[0] should not be larger than 128."

    def criteria(latent):
        g = pipeline.from_latent_to_grid(latent, resolution=128)
        g = g * (g >= 0.0).float()
        yhat = coarser_grid(g, 7 - p)
        return ((coarse_grid - yhat) ** 2).sum() * (8 ** (7 - p))
    return criteria


ref_latent = pipeline.get_test_latent(6)

with torch.no_grad():
    ref_rep = pipeline.from_latent_to_rep(ref_latent)
    ref_grid = pipeline.from_rep_to_grid(ref_rep, 128)
    ref_grid = ref_grid * (ref_grid >= 0.0).float()

plt.imshow(ref_grid[:, :, ref_grid.shape[2]//2, 0].cpu().T, cmap='gist_heat_r', vmin=0.0, vmax=0.6)
plt.gca().invert_yaxis()
plt.show()

with torch.no_grad():
    coarse_grid = coarser_grid(ref_grid, 3)

plt.imshow(coarse_grid[:, :, coarse_grid.shape[2]//2, 0].cpu().T, cmap='gist_heat_r', vmin=0.0, vmax=0.6)
plt.gca().invert_yaxis()
plt.show()

with torch.no_grad():
    im = cloudy.scattered(ref_grid * 300,
                          camera_poses=camera_poses,
                          scattering_albedo=(0.99, 0.98, 0.94),
                          environment=environment,
                          phase_g=0.0,
                          majorant=ref_grid.max()*300,
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
    coarse_im = cloudy.scattered(coarse_grid * 300,
                          camera_poses=camera_poses,
                          scattering_albedo=(0.99, 0.98, 0.94),
                          environment=environment,
                          phase_g=0.0,
                          majorant=coarse_grid.max()*300,
                          # kwargs
                          width=512,
                          height=512,
                          jittered=True,
                          samples=128
                          )
    plt.figure(figsize=(1, 1), dpi=512)
    plt.imshow(coarse_im[0].cpu() ** (1.0/2.2))
    plt.gca().invert_yaxis()
    plt.gca().axis('off')
    plt.tight_layout(pad=0.0)
    plt.show()


def on_new_state(i, latent):
    # if np.random.rand() < 0.9:
    #     return

    with torch.no_grad():
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
        fr = torch.zeros_like(im)
        fr[256:] = coarse_im[0][128:-128]
        fr[:256] = im[128:-128]
        images.append(fr)
        print(f"Generated image for step {i}")
        # plt.figure(figsize=(1,1), dpi=512)
        # plt.imshow(latent.cpu()[:,:,[8, 16, 24]]+0.5)
        # plt.gca().axis('off')
        # plt.tight_layout(pad=0.0)
        # plt.show()
        #
        # plt.figure(figsize=(1,1), dpi=512)
        # plt.imshow(fr.cpu() ** (1.0/2.2))
        # plt.gca().axis('off')
        # plt.gca().invert_yaxis()
        # plt.tight_layout(pad=0.0)
        # plt.show()


latent = pipeline.generate_latent(
    samples=300,
    scheduler_gamma=.4,
    criteria=create_super_resolution_criteria(coarse_grid),
    criteria_scale=1.0,
    callback=on_new_state
)

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
                          samples=32
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
        imc = cloudy.scattered(coarse_grid * 300,
                              camera_poses=camera_poses,
                              scattering_albedo=(0.99, 0.98, 0.94),
                              environment=environment,
                              phase_g=0.0,
                              majorant=coarse_grid.max() * 300,
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
    cloudy.save_video(frames, 'generated_cloud_super_resolution.webp')