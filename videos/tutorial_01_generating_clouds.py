import cloudy
import numpy as np
import torch


pipeline = cloudy.create_pipeline('./test')
# pipeline.download_pretrained()

recorder = pipeline.create_recorder()

# ref_latent = pipeline.get_test_latent(8)
# ref_capture_id = recorder.add_capture_latent(ref_latent)
# recorder.new_keyframe(ref_capture_id)

torch.manual_seed(101)

grid = pipeline.sample_volume(
    resolution=128,
    samples=200,
    scheduler_gamma=1.0,
    callback=lambda ci: recorder.new_keyframe(recorder.add_capture_latent(
        ci.latent,
        samples=max(1, 128 * ci.step // ci.total_steps)
    )))

grid_id = recorder.add_volume(grid)

recorder.show_clip(3, 2, width=512, height=256, samples_multiplier=1)

for i in range(100):
    alpha = 3.141593 * 2 * i / 99
    recorder.new_keyframe(recorder.add_capture_volume(
        grid_id,
        camera_position=(np.cos(0.5 + alpha)*2.7, np.sin(0.5 + alpha), np.sin(0.5 + alpha)*2.7),
        samples=128
    ))

recorder.save_video('tutorial_01_generating_clouds.webp', samples_multiplier=4)

