import cloudy
import numpy as np
import torch


pipeline = cloudy.create_pipeline('./test')
# pipeline.download_pretrained()

recorder = pipeline.create_recorder()

ref_latent = pipeline.get_test_latent(8)

def coarser_grid(g: torch.Tensor, levels: int) -> torch.Tensor:
    g = g.unsqueeze(0).permute(0, 4, 1, 2, 3)  # channels first
    for _ in range(levels):
        g = torch.nn.functional.interpolate(g, scale_factor=0.5, mode='trilinear', align_corners=True)
    return g.permute(0, 2, 3, 4, 1)[0]  # back to channels last


with torch.no_grad():
    ref_grid = pipeline.decode_latent(ref_latent)
    ref_grid = pipeline.clean_volume(ref_grid)
    coarse_grid = coarser_grid(ref_grid, 2)

ref_cap = recorder.add_capture_volume(coarse_grid, samples=128)

torch.manual_seed(101)

grid = pipeline.sample_volume(
    resolution=128,
    samples=200,
    scheduler_gamma=1.0,
    y=coarse_grid,
    A=lambda v: coarser_grid(v, 2),
    callback=lambda ci: recorder.new_keyframe(
        ref_cap,
        recorder.add_capture_latent(
            ci.latent,
            samples=max(1, 128 * ci.step // ci.total_steps)
    )))

# recorder.show_clip(4, 2, width=512, height=256, samples_multiplier=64)
# recorder.save_video('test.webp', samples_multiplier=32)

ref_vol_id = recorder.add_volume(coarse_grid)
smp_vol_id = recorder.add_volume(grid)

for i in range(100):
    alpha = 3.141593 * 2 * i / 99
    camera_position = (np.cos(0.5+alpha)*2.7, np.sin(0.5+alpha), np.sin(0.5+alpha)*2.7)
    ref_id = recorder.add_capture_volume(ref_vol_id, camera_position=camera_position, samples=128)
    smp_id = recorder.add_capture_volume(smp_vol_id, camera_position=camera_position, samples=128)
    recorder.new_keyframe(ref_id, smp_id)

recorder.save_video('tutorial_02_super_resolution.webp', samples_multiplier=4)

