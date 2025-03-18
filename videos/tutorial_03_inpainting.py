import cloudy
import numpy as np
import torch


pipeline = cloudy.create_pipeline('./test')
# pipeline.download_pretrained()

recorder = pipeline.create_recorder()

ref_latent = pipeline.get_test_latent(8)

with torch.no_grad():
    ref_grid = pipeline.decode_latent(ref_latent)
    ref_grid = pipeline.clean_volume(ref_grid)
    mask = torch.ones_like(ref_grid)
    mask[:64] = 0.0  # clear a half
    masked_grid = ref_grid * mask

ref_capture_id = recorder.add_capture_volume(masked_grid, samples=128)

torch.manual_seed(101)

grid = pipeline.sample_volume(
    resolution=128,
    samples=200,
    scheduler_gamma=1.0,
    y=masked_grid,
    A=lambda v: v*mask,
    guiding_strength=2.0,
    callback=lambda ci: recorder.new_keyframe(
        ref_capture_id,
        recorder.add_capture_latent(ci.latent, samples=max(1, 128*ci.step//ci.total_steps))
    ))

# recorder.show_clip(1, width=512, height=512, samples_multiplier=4)
# exit()
# recorder.save_video('test.webp', samples_multiplier=32)


ref_vol_id = recorder.add_volume(masked_grid)
smp_vol_id = recorder.add_volume(grid)

for i in range(100):
    alpha = 3.141593 * 2 * i/99
    camera_position = (np.cos(0.5+alpha)*2.7, np.sin(0.5+alpha), np.sin(0.5+alpha)*2.7)
    ref_id = recorder.add_capture_volume(ref_vol_id, camera_position=camera_position, samples=128)
    smp_id = recorder.add_capture_volume(smp_vol_id, camera_position=camera_position, samples=128)
    recorder.new_keyframe(ref_id, smp_id)

recorder.save_video('tutorial_03_inpainting.webp', samples_multiplier=4)
