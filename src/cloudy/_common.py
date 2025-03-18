import copy
import os
import time
from datetime import timedelta
from typing import Any

import torch.cuda
import typing
from . import _modeling as modeling
from . import _rendering as rendering
import numpy as np
from tqdm import tqdm
import rendervous as rdv
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    __PREFERRED_DEVICE__ = torch.device('cuda')
else:
    __PREFERRED_DEVICE__ = torch.device('cpu')


def preferred_device():
    return __PREFERRED_DEVICE__


def random_subset(N: int, K: int):
    batch_ids = [i for i in range(N)]
    np.random.shuffle(batch_ids)
    batch_ids = batch_ids[:K]
    return batch_ids


def last_file_id(path, file_prefix, file_posfix) -> int:
    max_id = 0
    for file in os.listdir(path):
        if file.startswith(file_prefix) and file.endswith(file_posfix):
            number_str = file[len(file_prefix):len(file)-len(file_posfix)]
            if number_str.isnumeric():
                max_id = max(max_id, int(number_str))
    return max_id


def sample_slice (vol, step_size: float, umin=-1.0, umax=1.0, vmin=-1.0, vmax=1.0, w=0.0, axis='z', device: typing.Optional[torch.device] = None):
    if device is None:
        device = preferred_device()
    u = torch.arange(umin, umax + 0.0000001, step_size, device=device)
    v = torch.arange(vmin, vmax + 0.0000001, step_size, device=device)
    w = torch.tensor([w], device=device)
    p = torch.cartesian_prod(w, v, u)
    if axis == 'z':
        p = p[:, [2, 1, 0]]
    elif axis == 'y':
        p = p[:, [2, 0, 1]]
    else:
        p = p[:, [0, 1, 2]]
    values = vol(p)
    return values.view(len(v), len(u), -1)


def eval_monoplanar_representation(x: torch.Tensor, latent: torch.Tensor, decoder: torch.nn.Module,
                    *,
                    features: typing.Optional[typing.Union[int, typing.List[int]]] = None,
                    samples: typing.Optional[typing.Union[int, typing.List[int]]] = None,
                    window_sizes: typing.Optional[typing.Union[float, typing.List[float]]] = None,
                    fourier_levels: int = 0
                    ):
    xz = x[:, [0, 2]]
    y = x[:, [1]]
    g = modeling.sample_grid2d(latent, xz, mode='bicubic')
    f = modeling.sample_monoplanar(
        g,
        y,
        features=features,
        samples=samples,
        window_sizes=window_sizes
    )
    if fourier_levels == 0:
        return decoder(f)
    return decoder(torch.cat([f, modeling.fourier_encode(y, fourier_levels)], dim=-1))


def eval_monoplanar_128x32_X(x: torch.Tensor, latent: torch.Tensor, decoder: torch.nn.Module):
    return eval_monoplanar_representation(
        x,
        latent,
        decoder,
        features=[16, 48],
        samples=[16, 48],
        window_sizes=[1.0, .5],
        fourier_levels=0
    )


def eval_monoplanar_128x32(x: torch.Tensor, latent: torch.Tensor, decoder: torch.nn.Module):
    return eval_monoplanar_representation(
        x,
        latent,
        decoder,
        features=[64],
        samples=[64],
        window_sizes=[1.0],
        fourier_levels=0
    )


def regularizer_l1(model):
    return sum (p.abs().sum() for p in model.parameters() if p.requires_grad)


def regularizer_l2(model):
    return sum ((p ** 2).sum() for p in model.parameters() if p.requires_grad)


def regularizer_latent_l1(latent):
    return latent.abs().sum()


def regularizer_latent_l2(latent):
    return (latent ** 2).sum()


def regularizer_monoplanar_latent_tv(latent):
    return modeling.total_variation_2D(latent.unsqueeze(0))


def cloud_transform(rep_model, scale, rotation, device):
    if scale == 1.0 and rotation == 0.0:
        return rep_model
    T = rdv.mat3.scale(rdv.vec3(1.0/scale, 1.0, 1.0/scale)) @ rdv.mat3.rotation(rdv.vec3(0., 1., 0.), rotation)
    T = T.to(device)
    def t_rep_model(x):
        return rep_model(x @ T)
    return t_rep_model


def latent_transform(latent, scale, rotation):
    if abs(scale - 1.0) <= .0001 and abs(rotation) <= 0.00001:
        return latent  # no sampling needed
    x = torch.arange(-1.0, 1.00001, 2.0 / (latent.shape[1] - 1), device=latent.device)
    z = torch.arange(-1.0, 1.00001, 2.0 / (latent.shape[1] - 1), device=latent.device)
    grid_xz = torch.cartesian_prod(z, x)[..., [1, 0]]
    inv_scale = 1.0/scale
    # Apply here a transform to grid_xz
    M = float(inv_scale)*torch.tensor([
        [np.cos(rotation), np.sin(rotation)],
        [-np.sin(rotation), np.cos(rotation)]], dtype=torch.float32, device=latent.device)

    new_latent = modeling.sample_grid2d(latent, grid_xz @ M.unsqueeze(0), mode='bicubic')
    return new_latent.view(len(z), len(x), -1)


def _update_recursive(dst: dict, src: dict):
    for k, v in src.items():
        assert k in dst.keys(), f'Key {k} is not present in default settings'
        if not isinstance(v, dict):
            dst[k] = v
        else:
            _update_recursive(dst[k], v)


class RepresentationModes:
    monoplanar_128_32 = 'monoplanar_128_32'

class Optimizers:
    adam = 'adam'
    nadam = 'nadam'
    adamw = 'adamw'


def create_optimization_objects(parameters, **settings):
    lr = settings['lr']
    lr_decay = settings['lr_decay']
    lr_restart = settings.get('lr_restart', 0)
    lr_scheduler = settings.get('lr_scheduler', 'exp')
    steps = settings['train_steps']
    optimizer_id = settings['optimizer']
    betas = settings.get('betas', (0.9, 0.999))
    if optimizer_id == Optimizers.adam:
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas, eps=1e-14)
    elif optimizer_id == Optimizers.nadam:
        optimizer = torch.optim.NAdam(parameters, lr=lr, betas=betas, eps=1e-14)
    elif optimizer_id == Optimizers.adamw:
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=1e-14)
    else:
        raise NotImplementedError()
    if lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, np.exp(np.log(lr_decay)/steps))
    elif lr_scheduler == 'cosine':
        if lr_restart == 0:
            lr_restart = steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, lr_restart, 2, lr * lr_decay)
    elif lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=steps,
            pct_start=0.05,
            div_factor=10,
            final_div_factor=1/lr_decay
        )
    else:
        raise NotImplementedError()
    return optimizer, scheduler


class Grid3DDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        latent, decoder, resx, resy, resz, noise, fw_batch_size, bw_batch_size = args
        device = latent.device
        ctx.decoder = decoder
        ctx.bw_batch_size = bw_batch_size
        stepx = 2.0 / (resx - 1)
        stepy = 2.0 / (resy - 1)
        stepz = 2.0 / (resz - 1)
        cell_size = torch.tensor([stepx, stepy, stepz], device=device, dtype=torch.float32)
        x = torch.arange(-1.0, 1.0000001, stepx, device=device)
        y = torch.arange(-1.0, 1.0000001, stepy, device=device)
        z = torch.arange(-1.0, 1.0000001, stepz, device=device)
        p = torch.cartesian_prod(z, y, x)[:, [2, 1, 0]]
        if noise > 0.0:
            p += (torch.rand_like(p) - .5) * cell_size.view(-1,3) * noise
        ctx.save_for_backward(latent, p)
        number_of_samples = p.shape[0]
        with torch.no_grad(): # just in case...
            if fw_batch_size >= number_of_samples:
                o = decoder(latent, p)
            else:
                o = None
                offset = 0
                import math
                number_of_batches = int(math.ceil(number_of_samples / fw_batch_size))
                for bp in torch.chunk(p, number_of_batches):
                    bo = decoder(latent, bp)
                    if o is None:
                        o = torch.zeros(number_of_samples, bo.shape[-1], device=device)
                    o[offset:offset + bo.shape[0],:].copy_(bo)
                    offset += bo.shape[0]
        return o.view(resz, resy, resx, o.shape[-1])

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grads, = grad_outputs
        latent, p = ctx.saved_tensors
        num_samples = p.shape[0]
        grads = grads.view(num_samples, -1)
        decoder = ctx.decoder
        batch_size = ctx.bw_batch_size
        import math
        number_of_chunks = int(math.ceil(num_samples / batch_size))
        batched_p = torch.chunk(p, number_of_chunks)
        batched_grads = torch.chunk(grads, number_of_chunks)
        latent_grad = torch.zeros_like(latent)
        for bp, bg in zip(batched_p, batched_grads):
            with torch.enable_grad():
                bo = decoder(latent, bp)
                bl_grad = torch.autograd.grad([bo], [latent], [bg])[0]
            latent_grad.add_(bl_grad)
        return latent_grad, None, None, None, None, None, None, None


class LatentDecoder(torch.nn.Module):
    def __init__(self, upsampler, decoder):
        super().__init__()
        upsampler = copy.deepcopy(upsampler)
        decoder = copy.deepcopy(decoder)
        for p in list(upsampler.parameters()) + list(decoder.parameters()):
            p.requires_grad_(False)
        self.upsampler = upsampler
        self.decoder = decoder

    def forward(self, *args, **kwargs):
        latent, = args
        resx = kwargs.get('resx')
        resy = kwargs.get('resy')
        resz = kwargs.get('resz')
        noise = kwargs.get('noise', 0.0)
        batch_size = kwargs.get('batch_size', None)
        if batch_size is None:
            batch_size = resx * resy * resz
        bw_batch_size = max(1, batch_size // 8)

        up_latent = self.upsampler(latent.unsqueeze(0))[0]

        return Grid3DDecode.apply(
            up_latent,
            lambda ltn, x: eval_monoplanar_128x32(x, ltn, self.decoder),
            resx, resy, resz,
            noise,
            batch_size,
            bw_batch_size)


class CallbackInfo:

    def __init__(self,
                 pipeline: 'Pipeline',
                 step: int,
                 total_steps: int,
                 timestep: int,
                 latent: torch.Tensor,
                 normalized: bool,
                 pass_index: int = 0,
                 subpass: str = 'dps'
                 ):
        self._pipeline = pipeline
        self._step = step
        self._total_steps = total_steps
        self._sampled_timestep = timestep
        self._latent = pipeline.denormalize_latent(latent) if normalized else latent
        self._normalized_latent = latent if normalized else pipeline.normalize_latent(latent)
        self._pass_index = pass_index
        self._subpass = subpass

    @property
    def sampled_timestep(self):
        """
        The timestep index within the whole timesteps (T...0).
        """
        return self._sampled_timestep

    @property
    def step(self):
        """
        The index of the current sample in the DDIM subsampling scheme [0...samples),
        or step within an optimization process.
        """
        return self._step

    @property
    def total_steps(self):
        """
        The final total steps of the DDIM subsampling scheme [0...samples),
        or number of steps in an optimization process.
        """
        return self._total_steps

    @property
    def latent(self):
        """
        The latent denormalized ready for decoding with channels last.
        """
        return self._latent

    @property
    def normalized_latent(self):
        """
        The latent normalized with channels first.
        """
        return self._normalized_latent

    @property
    def pass_index(self):
        """
        The current pass during optimization. -1 represents the initial unconditional sampling
        """
        return self._pass_index

    @property
    def subpass(self):
        """
        The current sub-pass during optimization.
        'dps' - sampling
        'opt' - optimizing
        'ref' - refining
        """
        return self._subpass

    def volume(self, resolution: int = 128):
        """
        Decodes the latent as a clean volume (0...1)
        """
        with torch.no_grad():
            return self._pipeline.clean_volume(self._pipeline.decode_latent(self._latent, resolution=resolution))


class Recorder:
    def __init__(self, pipeline: 'Pipeline'):
        self.__pipeline = pipeline
        self.__environments = []
        self.__latents = []
        self.__volumes = []
        self.__captures = []
        self.__frames = []
        environment = 0.4 * torch.ones(64, 128, 3, device='cuda')
        # environment[:, :, 0] = (59 / 255) ** 2.2
        # environment[:, :, 1] = (94 / 255) ** 2.2
        # environment[:, :, 2] = (134 / 255) ** 2.2
        environment[32:] *= 0.2
        environment[12, 0] = 2500
        self.add_environment(environment) # adds a default environment
        self.__default_environment_objects = rendering.environment_objects(environment)

    @property
    def num_frames(self):
        return len(self.__frames)

    def equal_tensors(self, t1, t2):
        return t1.shape == t2.shape and torch.equal(t1, t2)

    def _resolve_environment(self, environment):
        environment = environment.detach().clone().cpu()
        if len(self.__environments) == 0 or not self.equal_tensors(self.__environments[-1], environment):
            self.__environments.append(environment)
        return len(self.__environments) - 1

    def _resolve_latent(self, latent):
        latent = latent.detach().clone().cpu()
        if len(self.__latents) == 0 or not self.equal_tensors(self.__latents[-1], latent):
            self.__latents.append(latent)
        return len(self.__latents) - 1

    def _resolve_volume(self, volume):
        volume = volume.detach().clone().cpu()
        if len(self.__volumes) == 0 or not self.equal_tensors(self.__volumes[-1], volume):
            self.__volumes.append(volume)
        return len(self.__volumes) - 1

    def add_environment(self, environment: torch.Tensor):
        self.__environments.append(environment.detach().clone().cpu())
        return len(self.__environments) - 1

    def add_latent(self, latent: torch.Tensor):
        self.__latents.append(latent.detach().clone().cpu())
        return len(self.__latents) - 1

    def add_volume(self, volume: torch.Tensor):
        self.__volumes.append(volume.detach().clone().cpu())
        return len(self.__volumes) - 1

    def add_capture_latent(self,
                    latent: typing.Union[torch.Tensor, int],
                    environment: typing.Union[torch.Tensor, int] = 0,
                    density_scale: float = 300,
                    scattering_albedo: typing.Tuple[float, float, float] = (0.99, 0.98, 0.94),
                    phase_g: float = 0.0,
                    camera_position: typing.Tuple[float, float, float] = (np.cos(0.5) * 2.7, np.sin(0.5), np.sin(0.5) * 2.7),
                    samples: int = 1,
                    render_mode: typing.Literal['tr', 'ms', 'msw'] = 'msw'
                    ):
        if isinstance(latent, torch.Tensor):
            latent = self._resolve_latent(latent)
        if isinstance(environment, torch.Tensor):
            environment = self._resolve_environment(environment)
        self.__captures.append(dict(
            capture_mode='latent',
            index=latent,
            density_scale=density_scale,
            environment_index=environment,
            scattering_albedo=scattering_albedo,
            phase_g=phase_g,
            camera_position=camera_position,
            samples=samples,
            render_mode=render_mode
        ))
        return len(self.__captures) - 1

    def add_capture_volume(self,
                    volume: typing.Union[torch.Tensor, int],
                    environment: typing.Union[torch.Tensor, int] = 0,
                    density_scale: float = 300,
                    scattering_albedo: typing.Tuple[float, float, float] = (0.99, 0.98, 0.94),
                    phase_g: float = 0.0,
                    camera_position: typing.Tuple[float, float, float] = (np.cos(0.5) * 2.7, np.sin(0.5), np.sin(0.5) * 2.7),
                    samples: int = 1,
                    render_mode: typing.Literal['tr', 'ms', 'msw'] = 'msw'
                    ):
        if isinstance(volume, torch.Tensor):
            volume = self._resolve_volume(volume)
        if isinstance(environment, torch.Tensor):
            environment = self._resolve_environment(environment)
        self.__captures.append(dict(
            capture_mode='volume',
            index=volume,
            density_scale=density_scale,
            environment_index=environment,
            scattering_albedo=scattering_albedo,
            phase_g=phase_g,
            camera_position=camera_position,
            samples=samples,
            render_mode=render_mode
        ))
        return len(self.__captures) - 1

    def add_capture_image(self, rendered_image: torch.Tensor):
        self.__captures.append(dict(
            capture_mode='image',
            image=rendered_image
        ))
        return len(self.__captures) - 1

    def new_keyframe(self, *capture_indices):
        self.__frames.append(list(capture_indices))

    def save(self, file_name: str):
        torch.save({
            'captures': self.__captures,
            'frames': self.__frames,
            'environments': self.__environments,
            'latents': self.__latents,
        }, file_name)

    def load(self, file_name: str):
        data = torch.load(file_name)
        self.__latents = data['latents']
        self.__frames = data['frames']
        self.__captures = data['captures']
        self.__environments = data['environments']

    def _create_environment_objects(self, *ids):
        environments = {}
        for env_id in ids:
            if env_id in environments:
                continue
            env, env_sampler = rendering.environment_objects(self.__environments[env_id].to(self.__pipeline.get_device()))
            environments[env_id] = (env, env_sampler)
        return environments

    def _render_grid_ms(self,
        grid: torch.Tensor,
        density_scale: float,
        scattering_albedo,
        phase_g,
        environment,
        environment_sampler,
        camera_position,
        width: int,
        height: int,
        samples: int
    ):
        return rendering.accumulate(lambda: rendering.scattered(
                    grid * density_scale,
                    camera_poses=rendering.camera_poses(camera_position),
                    scattering_albedo=scattering_albedo,
                    environment=environment,
                    phase_g=phase_g,
                    majorant=grid.max() * density_scale,
                    environment_sampler=environment_sampler,
                    width=width,
                    height=height,
                    jittered=samples > 16,
                    samples=min(32, samples)
                ), times=max(1, samples // 32))[0]

    def _render_grid_msw(self,
        grid: torch.Tensor,
        density_scale: float,
        scattering_albedo,
        phase_g,
        environment,
        environment_sampler,
        camera_position,
        width: int,
        height: int,
        samples: int
    ):
        T = rendering.transmittance(grid*density_scale,
                                    camera_poses=rendering.camera_poses(camera_position),
                                    width=width, height=height, samples=1, jittered=False
                                    )[0]
        return T + (1 - T) * rendering.accumulate(lambda: rendering.scattered(
                    grid * density_scale,
                    camera_poses=rendering.camera_poses(camera_position),
                    scattering_albedo=scattering_albedo,
                    environment=rdv.ONE,
                    phase_g=phase_g,
                    majorant=grid.max() * density_scale,
                    environment_sampler=environment_sampler,
                    width=width,
                    height=height,
                    jittered=samples > 16,
                    samples=min(32, samples)
                ), times=max(1, samples // 32))[0]


    def render_image(self,
                     grid: typing.Union[int, torch.Tensor],
                     environment: typing.Union[int, torch.Tensor] = 0,
                     density_scale: float = 300,
                     scattering_albedo: typing.Tuple[float, float, float] = (0.99, 0.98, 0.94),
                     phase_g: float = 0.0,
                     camera_position: typing.Tuple[float, float, float] = (
                     np.cos(0.5) * 2.7, np.sin(0.5), np.sin(0.5) * 2.7),
                     width: int = 512,
                     height: int = 512,
                     samples: int = 32,
                     render_mode: typing.Literal['tr', 'ms', 'msw'] = 'msw'
    ):
        if isinstance(grid, int):
            grid = self.__volumes[grid]
        if isinstance(environment, int):
            environment = self.__environments[environment]
        if environment is self.__environments[0]:
            environment, environment_sampler = self.__default_environment_objects
        else:
            environment, environment_sampler = rendering.environment_objects(environment)
        if render_mode == 'ms':
            return self._render_grid_ms(grid, density_scale, scattering_albedo, phase_g, environment, environment_sampler, camera_position, width, height, samples)
        if render_mode == 'msw':
            return self._render_grid_msw(grid, density_scale, scattering_albedo, phase_g, environment,
                                        environment_sampler, camera_position, width, height, samples)
        raise NotImplementedError()


    def render_captures(self, *ids, width: int = 512, height: int = 512, samples_multiplier: int = 1, max_samples: int = 4096):
        device = self.__pipeline.get_device()
        captures = {}
        env_ids = [self.__captures[i]['environment_index'] for i in ids if self.__captures[i]['capture_mode'] != 'image']
        environment_objects = self._create_environment_objects(*env_ids)
        prev_grid = None
        prev_grid_index = -1
        prev_grid_source = ''
        ids = set(ids)
        for cap_id in tqdm(ids, "Rendering captures"):
            c = self.__captures[cap_id]
            if c['capture_mode'] == 'image':
                im = modeling.resampling(c['image'].unsqueeze(0), (height, width), align_corners=False)[0]
            else:
                if prev_grid_source != c['capture_mode'] or prev_grid_index != c['index']:
                    prev_grid_index = c['index']
                    prev_grid_source = c['capture_mode']
                    if prev_grid_source == 'latent':
                        latent = self.__latents[prev_grid_index]
                        prev_grid = self.__pipeline.decode_latent(latent.to(device))
                        prev_grid = self.__pipeline.clean_volume(prev_grid)
                    else:  # explicit volume
                        prev_grid = self.__volumes[prev_grid_index]
                grid = prev_grid
                env, env_sampler = environment_objects[c['environment_index']]
                samples = min(c['samples'] * samples_multiplier, max_samples)
                if c['render_mode'] == 'ms':
                    im = self._render_grid_ms(
                        grid, c['density_scale'], c['scattering_albedo'], c['phase_g'],
                        env, env_sampler, c['camera_position'],
                        width, height, samples
                    )
                elif c['render_mode'] == 'msw':
                    im = self._render_grid_msw(
                        grid, c['density_scale'], c['scattering_albedo'], c['phase_g'],
                        env, env_sampler, c['camera_position'],
                        width, height, samples
                    )
                else:
                    raise NotImplementedError()
            captures[cap_id] = torch.flip(im.cpu(), dims=[0])
        return captures

    def render_frames(self, *ids, width: int = 512, height: int = 512, samples_multiplier: int = 1, max_samples: int = 4096):
        cap_ids = []
        for i in ids:
            for c in self.__frames[i]:
                cap_ids.append(c)
        cap_ids = set(cap_ids)
        captured_images = self.render_captures(
            *cap_ids,
            width=width,
            height=height,
            samples_multiplier=samples_multiplier,
            max_samples=max_samples
        )
        frames = torch.zeros(len(ids), height, width, 3)
        for i, id in enumerate(ids):
            captures = self.__frames[id]
            if len(captures) == 1:
                frames[i] = captured_images[captures[0]]
            elif len(captures) == 2:
                h = height // 4
                c0 = captured_images[captures[0]][h:3 * h]
                c1 = captured_images[captures[1]][h:-h]
                frames[i][2 * h:] = c1
                frames[i][:2 * h] = c0
            elif len(captures) == 3:
                h = height // 3
                c0 = captured_images[captures[0]][height//2 - h//2:height//2+h-(h//2)]
                frames[i][:h] = c0
                c1 = captured_images[captures[1]][height//2 - h//2:height//2+h-(h//2)]
                frames[i][h:2*h] = c1
                h = height - 2 * h
                c2 = captured_images[captures[2]][height//2 - h//2:height//2+h-(h//2)]
                frames[i][-h:] = c2

        return frames

    def save_video(self,
                     file_name: str,
                     width: int = 512,
                     height: int = 512,
                     samples_multiplier: int = 1,
                     max_samples: int = 4096):
        frames = self.render_frames(
            *list(range(len(self.__frames))),
            width=width,
            height=height,
            samples_multiplier=samples_multiplier,
            max_samples=max_samples)
        rendering.save_video(frames, file_name)

    def show_clip(self,
                  cols: int,
                  rows: int = 1,
                  width: int = 512,
                  height: int = 512,
                  samples_multiplier: int = 1,
                  frame_border: bool = False
                  ):
        num_frames = cols * rows
        if num_frames == 0:
            return
        if num_frames == 1:
            ids = [self.num_frames - 1]
        else:
            ids = (np.arange(0.0, 1.00000001, 1.0/(num_frames-1)) * (self.num_frames - 1)).astype(np.int32)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * width/height, rows), dpi=height)
        frames = self.render_frames(*ids,
                        width=width,
                        height=height,
                        samples_multiplier=samples_multiplier)
        for i, f in enumerate(frames):
            a = axes if num_frames == 1 else (axes[i] if rows == 1 else axes[i//cols, i%cols])
            a.imshow(torch.clamp(f ** (1.0/2.2), 0.0, 1.0))
            if frame_border:
                a.get_xaxis().set_ticks([])
                a.get_yaxis().set_ticks([])
            else:
                a.axis('off')
        fig.tight_layout(pad=0.0)
        fig.show()


class Pipeline:
    def __init__(self, workplace: str, **settings):
        self.workplace = workplace
        final_settings = self.default_settings()
        _update_recursive(final_settings, settings)
        self.settings = final_settings

    def default_settings(self):
        return dict(
            data_path ='../../../cloudy_project_prev',
            clouds_folder = 'cloudy',
            representation_mode = RepresentationModes.monoplanar_128_32,
            decoder = dict(
                train_N=64,
                train_K=22,
                train_steps=20_000,
                optimizer=Optimizers.nadam,
                # lr=0.002,  # first 5000
                # lr_decay=0.05,
                lr=0.001,  # second 5000
                lr_decay=.05,
                # lr_restart=5000,
                cp_every=1000,
                decoder_l2 = 0.0, # 1e-7,
                # first decoder_l2 = 1e-8, # 1e-7,
                decoder_l1 = 0.0,
                latent_l2 = 1e-8,
                latent_l1 = 0.0, # 1e-7,
                latent_tv = 1e-3, # 1e-4
            ),
            diffuser=dict(
                train_N=1000*14,
                train_K=1000,
                train_B=16,
                train_steps=1_000_000,
                optimizer=Optimizers.adamw,
                # first 1% with 0.00001,
                # next, 0.0001 with exp decay to 1%
                # lr=0.00001,
                lr=0.0001,
                lr_decay=.01,
                # lr_restart=5000,
                betas=(0.9, 0.999),
                cp_every=5_000
            ),
            diffuser_acc=dict(
                train_N=1000 * 14,
                train_K=1000,
                train_B=32,
                train_steps=400_000,
                optimizer=Optimizers.adamw,
                # first 1% with 0.00001,
                # next, 0.0001 with exp decay to 1%
                # lr=0.00001,
                ## First 200_000
                # lr = 0.0001,
                # lr_decay = 0.01,
                # lr_scheduler='onecycle',
                lr = 1e-7,
                lr_decay = 1.0,
                lr_scheduler='exp',
                # lr_restart=5000,
                betas=(0.9, 0.999),
                cp_every=5_000
            ),
            encoder = dict(
                train_steps=2_000,
                optimizer=Optimizers.nadam,
                lr=0.004,
                betas=(0.9, 0.9),
                lr_decay=.1,
                latent_l2=1e-8, #1e-6,
                latent_l1=0.0, # 0.0, #1e-4,
                latent_tv=1e-2 # 1e-3  #1e-3
            ),
            enhancer=dict(
                train_steps=800,
                optimizer=Optimizers.nadam,
                lr=0.004,
                lr_decay=.1,
                betas=(0.9, 0.9),
                latent_l2=1e-8,
                latent_l1=1e-8,
                latent_tv=1e-4
                # latent_l1=1e-4,
                # latent_tv=1e-3
            )
        )

    def create_recorder(self):
        return Recorder(self)

    def get_latent_shape(self):
        if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:
            return (128, 128, 32)
        raise NotImplementedError()

    def get_device(self):
        return preferred_device()

    def get_clouds_path(self):
        return f"{self.settings['data_path']}/{self.settings['clouds_folder']}"

    def get_number_of_clouds(self):
        if not hasattr(self, 'number_of_clouds'):
            clouds_path = self.get_clouds_path()
            id = 0
            while os.path.exists(clouds_path+f"/cloud_{id}.pt"):
                id += 1
            print(f"[INFO] Not found {clouds_path+f'/cloud_{id}.pt'}")
            self.number_of_clouds = id
            print(f"[INFO] Number of clouds {id}")
        return self.number_of_clouds

    def get_cloud(self, cloud_id, map_location = None):
        if map_location is None:
            map_location = self.get_device()
        f = self.get_clouds_path() + f"/cloud_{cloud_id}.pt"
        return torch.load(f, map_location=map_location, weights_only=True)

    def get_volume(self, cloud_id):
        return self.from_grid_to_volume(self.get_cloud(cloud_id))

    def get_encoded_latent(self, cloud_id):
        return torch.load(self.workplace +f"/encoded/latent_{cloud_id}.pt", map_location=self.get_device(), weights_only=True)

    def get_enhanced_latent(self, latent_id):
        return torch.load(self.workplace +f"/enhanced/latent_{latent_id}.pt", map_location=self.get_device(), weights_only=True)

    def get_test_latent(self, latent_id):
        return torch.load(self.workplace +f"/test/latent_{latent_id}.pt", map_location=self.get_device(), weights_only=True)

    def get_number_of_latents_for_decoder_training(self):
        return self.settings['decoder']['train_N']

    def get_number_of_enhanced_transforms(self):
        return 14

    def get_enhanced_transform_r_s(self, transform_idx):
        r = np.pi * 0.5 * transform_idx / self.get_number_of_enhanced_transforms()
        # s = 1.0 - 0.2 * (0.5 - np.cos(transform_idx * 2 * np.pi / self.get_number_of_enhanced_transforms()) * 0.5)
        scales = [1.0, 0.85, 0.95, 0.8, 0.95, 0.8, 0.9] * 2
        s = scales[transform_idx]
        return r, s

    def get_latents_for_decoder_training(self, cp: typing.Optional[int] = None):
        if cp is None:
            cp = last_file_id(self.workplace + "/decoder", "latents_", ".pt")
        if cp == 0:  # No cp
            raise Exception("No latents saved yet.")
        return torch.load(self.workplace+f"/decoder/latents_{cp}.pt", map_location=self.get_device(), weights_only=True)

    def get_volume_for_decoder_training(self, id: int):
        try:
            ids = torch.load(self.workplace + "/decoder/batch_used.pt", weights_only=True)
        except:
            raise Exception("No trained decoder yet.")
        return self.from_grid_to_volume(torch.load(
                self.get_clouds_path() + f"/cloud_{ids[id]}.pt", map_location=self.get_device(), weights_only=True
            ))

    def get_decoder(self, cp: typing.Optional[int] = None):
        if not hasattr(self, 'decoding'):
            # Create decoder and upsampler models
            rep_mode = self.settings['representation_mode']
            if rep_mode == RepresentationModes.monoplanar_128_32:
                decoder = modeling.MLP(64, 1, 256, 6, activation_scheme='leaky_relu').to(self.get_device())
                upsampler = modeling.FeatureUpsampler(32, 64, activation='leaky_relu').to(self.get_device())
            else:
                raise NotImplementedError(f'Can not create decoder for mode {rep_mode}')
            # load the last cp state
            if cp is None:
                cp = last_file_id(self.workplace + "/decoder", "decoder_",".pt")
            if cp > 0:  # some checkpoint found
                decoder.load_state_dict(torch.load(self.workplace+f"/decoder/decoder_{cp}.pt", weights_only=True))
                upsampler.load_state_dict(torch.load(self.workplace + f"/decoder/upsampler_{cp}.pt", weights_only=True))
            self.decoding = decoder, upsampler, cp
        assert cp is None or self.decoding[2] == cp
        self.decoding[0].eval()
        self.decoding[1].eval()
        return self.decoding  # this is a tuple decoder, upsampler, checkpoint_index

    def get_latent_decoder(self, cp = None):
        if not hasattr(self, 'latent_decoding'):
            decoder, upsampler, _ = self.get_decoder(cp)
            self.latent_decoding = LatentDecoder(upsampler, decoder)
        return self.latent_decoding

    def get_diffuser(self, cp: typing.Optional[int] = None, use_ema: bool = True):
        if not hasattr(self, 'diffuser'):
            # Create decoder and upsampler models
            rep_mode = self.settings['representation_mode']
            if rep_mode == RepresentationModes.monoplanar_128_32:
                diffuser = modeling.GaussianDiffuser2D(32, 128).to(self.get_device())
            else:
                raise NotImplementedError(f'Can not create decoder for mode {rep_mode}')
            # load the last cp state
            if cp is None:
                cp = last_file_id(self.workplace + "/diffuser", "denoiser_", ".pt")
            if cp > 0:  # some checkpoint found
                if use_ema:
                    diffuser.denoiser.load_state_dict(torch.load(self.workplace + f"/diffuser/denoiser_{cp}.pt", weights_only=True))
                else:
                    diffuser.load_state_dict(torch.load(self.workplace + f"/diffuser/optimizing_{cp}.pt", weights_only=True)['model'])
            self.diffuser = diffuser, cp
        self.diffuser[0].eval()
        return self.diffuser

    def get_normalization_stats(self):
        if not hasattr(self, 'norm_stats'):
            self.norm_stats = torch.load(self.workplace + "/enhanced/stats.pt", map_location=self.get_device(), weights_only=True)
        return self.norm_stats

    def load_cloud_grid(self, cloud_id):
        return torch.load(self.get_clouds_path() + f"/cloud_{cloud_id}.pt", map_location=preferred_device(), weights_only=True)

    def run_train_decoder_batched(self):
        settings = self.settings['decoder']
        rep_mode = self.settings['representation_mode']
        # ensure output folder
        output_path = self.workplace + "/decoder"
        os.makedirs(output_path, exist_ok=True)
        # select N clouds
        N = settings['train_N']
        K = settings['train_K']
        if not os.path.exists(output_path + "/batch_used.pt"):
            torch.save(random_subset(self.get_number_of_clouds(), N), output_path + "/batch_used.pt")
        ids = torch.load(output_path + "/batch_used.pt", weights_only=True)
        # get decoder
        decoder, upsampler, cp = self.get_decoder()
        decoder.train()
        upsampler.train()
        cp_every = settings['cp_every']
        # create latents
        if rep_mode == RepresentationModes.monoplanar_128_32:
            latents = torch.nn.Parameter(torch.zeros(N, 128, 128, 32, device=self.get_device()))
        else:
            raise NotImplementedError()
        # load latent current state
        if cp > 0:
            with torch.no_grad():
                latents.copy_(torch.load(self.workplace + f"/decoder/latents_{cp}.pt", weights_only=True))
        # get optimizer objects
        decoder_opt, decoder_sch = create_optimization_objects(
            list(decoder.parameters())+list(upsampler.parameters()),
            **settings
        )
        latents_opt, latents_sch = create_optimization_objects([latents], **settings)

        if cp > 0:  # load optimizer states if checkpoint exists
            decoder_opt.load_state_dict(torch.load(self.workplace+f"/decoder/decoder_opt_{cp}.pt", weights_only=True))
            latents_opt.load_state_dict(torch.load(self.workplace+f"/decoder/latents_opt_{cp}.pt", weights_only=True))

        # Set enhancing transforms
        if rep_mode == RepresentationModes.monoplanar_128_32:
            latent_transforms = [  # Z x X x F  Single latent
                lambda l: l,  # x z
                lambda l: torch.flip(l, dims=[0]),  # x negz
                lambda l: torch.flip(l, dims=[1]),  # negx z
                lambda l: torch.flip(l, dims=[0, 1]),  # negx negz
                lambda l: torch.transpose(l, dim0=0, dim1=1),  # z x
                lambda l: torch.flip(torch.transpose(l, dim0=0, dim1=1), dims=[0]),  # z negx
                lambda l: torch.flip(torch.transpose(l, dim0=0, dim1=1), dims=[1]),  # negz x
                lambda l: torch.flip(torch.transpose(l, dim0=0, dim1=1), dims=[0, 1]),  # negz negx
            ]
            x_transforms = [
                lambda xyz: xyz,
                lambda xyz: xyz * torch.tensor([[1.0, 1.0, -1.0]], device=xyz.device),
                lambda xyz: xyz * torch.tensor([[-1.0, 1.0, 1.0]], device=xyz.device),
                lambda xyz: xyz * torch.tensor([[-1.0, 1.0, -1.0]], device=xyz.device),
                lambda xyz: xyz[..., [2, 1, 0]],
                lambda xyz: xyz[..., [2, 1, 0]] * torch.tensor([[1.0, 1.0, -1.0]], device=xyz.device),
                lambda xyz: xyz[..., [2, 1, 0]] * torch.tensor([[-1.0, 1.0, 1.0]], device=xyz.device),
                lambda xyz: xyz[..., [2, 1, 0]] * torch.tensor([[-1.0, 1.0, -1.0]], device=xyz.device),
            ]
        else:
            raise NotImplementedError()

        # Regularization factors
        decoder_l2_reg = settings['decoder_l2']
        decoder_l1_reg = settings['decoder_l1']
        latent_l2_reg = settings['latent_l2']
        latent_l1_reg = settings['latent_l1']
        latent_tv_reg = settings['latent_tv']

        # training loop
        steps = settings['train_steps']
        start_step = cp * cp_every
        # # stratified sampling
        # corners_x = torch.arange(-1.0, .99999999, 2.0 / 127, device=self.get_device())
        # corners_z = torch.arange(-1.0, .99999999, 2.0 / 127, device=self.get_device())
        # corners_y = torch.arange(-.5, .4999999, 1.0 / 3, device=self.get_device())
        # corners = torch.cartesian_prod(corners_x, corners_y, corners_z)
        # cell_size = torch.tensor([[2.0 / 127, 1.0 / 3, 2.0 / 127]], device=self.get_device())

        cloud_grids_cpu = [self.get_cloud(id, map_location='cpu') for id in ids]
        steps_iterator = tqdm(range(start_step, steps), desc="Loss: -")
        x_scaler = torch.tensor([[1.0, 2.0, 1.0]], device=self.get_device())  # Used to map -0.5...0.5 in y to full range -1.0...1.0 in the rep
        for step in steps_iterator:
            # clear gradients
            decoder_opt.zero_grad()
            latents_opt.zero_grad()
            step_loss_value = 0
            # generate sampling positions
            # x = corners + torch.rand_like(corners) * cell_size
            with torch.no_grad():
                x = torch.rand(64 * 1024, 3, device=self.get_device()) * 2 - 1.0
                x[:,1] *= 0.5

            offset = 0
            while offset < N:
                clouds = [
                    modeling.Volume(cloud_grids_cpu[i].to(self.get_device())
                        , fit_box=(1.0, 0.5, 1.0)) for i in range(offset, min(N, offset+K))
                ]
                for i, v in enumerate(clouds):
                    latent_id = offset + i
                    with torch.no_grad():
                        ref_values = v(x)  # sample once from a single cloud
                    # assume 8 different transforms
                    for latent_transform, x_transform in zip(latent_transforms, x_transforms):
                        tl = latent_transform(latents[latent_id])
                        tx = x_transform(x)
                        # random_size = np.random.randint(64, 128)
                        # random_size = (random_size, random_size)#, 32 * random_size // 128)
                        # tl = tl.unsqueeze(0) #.unsqueeze(-1)
                        # tl = modeling.resampling(tl, random_size, mode='bicubic')
                        # # # tl = modeling.resample_grid(tl, random_size)
                        # tl = modeling.resampling(tl, (128, 128), mode='bicubic')[0]
                        # # tl = modeling.resample_grid(tl, (128, 128, 32))
                        # tl = tl.squeeze(-1).squeeze(0)
                        if rep_mode == RepresentationModes.monoplanar_128_32:
                            inf_values = eval_monoplanar_128x32(
                                tx * x_scaler, upsampler(tl.unsqueeze(0))[0], decoder
                            )
                        else:
                            raise NotImplementedError()
                        loss = torch.nn.functional.mse_loss(inf_values, ref_values, reduction='sum')
                        loss_value = loss.item()

                        if decoder_l2_reg > 0:
                            loss += regularizer_l2(decoder) * decoder_l2_reg
                            loss += regularizer_l2(upsampler) * decoder_l2_reg
                        if decoder_l1_reg > 0:
                            loss += regularizer_l1(decoder) * decoder_l1_reg
                            loss += regularizer_l1(upsampler) * decoder_l1_reg
                        if latent_l1_reg > 0:
                            loss += regularizer_latent_l1(tl) * latent_l1_reg
                        if latent_l2_reg > 0:
                            loss += regularizer_latent_l2(tl) * latent_l2_reg
                        if latent_tv_reg > 0:
                            if rep_mode == RepresentationModes.monoplanar_128_32:
                                loss += regularizer_monoplanar_latent_tv(tl) * latent_tv_reg
                            else:
                                raise NotImplementedError()
                        loss.backward()

                        step_loss_value += loss_value
                offset += len(clouds)
            ave_loss = step_loss_value / N / len(latent_transforms)
            steps_iterator.set_description(f"Loss: {ave_loss}")
            decoder_opt.step()
            latents_opt.step()
            decoder_sch.step()
            latents_sch.step()

            if (step + 1) % cp_every == 0:  # save checkpoint
                current_cp = (step + 1) // cp_every
                torch.save(decoder.state_dict(), self.workplace +f"/decoder/decoder_{current_cp}.pt")
                torch.save(upsampler.state_dict(), self.workplace +f"/decoder/upsampler_{current_cp}.pt")
                torch.save(latents, self.workplace +f"/decoder/latents_{current_cp}.pt")
                torch.save(decoder_opt.state_dict(), self.workplace + f"/decoder/decoder_opt_{current_cp}.pt")
                torch.save(latents_opt.state_dict(), self.workplace + f"/decoder/latents_opt_{current_cp}.pt")

    def run_train_decoder(self):
        settings = self.settings['decoder']
        rep_mode = self.settings['representation_mode']
        # ensure output folder
        output_path = self.workplace + "/decoder"
        os.makedirs(output_path, exist_ok=True)
        # select N clouds
        N = settings['train_N']
        if not os.path.exists(output_path + "/batch_used.pt"):
            torch.save(random_subset(self.get_number_of_clouds(), N), output_path + "/batch_used.pt")
        ids = torch.load(output_path + "/batch_used.pt", weights_only=True)
        # get decoder
        decoder, upsampler, cp = self.get_decoder()
        decoder.train()
        upsampler.train()
        cp_every = settings['cp_every']
        # create latents
        if rep_mode == RepresentationModes.monoplanar_128_32:
            latents = torch.nn.Parameter(torch.zeros(N, 128, 128, 32, device=self.get_device()))
        else:
            raise NotImplementedError()
        # load latent current state
        if cp > 0:
            with torch.no_grad():
                latents.copy_(torch.load(self.workplace + f"/decoder/latents_{cp}.pt", weights_only=True))
        # get optimizer objects
        decoder_opt, decoder_sch = create_optimization_objects(
            list(decoder.parameters())+list(upsampler.parameters()),
            **settings
        )
        latents_opt, latents_sch = create_optimization_objects([latents], **settings)

        if cp > 0:  # load optimizer states if checkpoint exists
            decoder_opt.load_state_dict(torch.load(self.workplace+f"/decoder/decoder_opt_{cp}.pt", weights_only=True))
            latents_opt.load_state_dict(torch.load(self.workplace+f"/decoder/latents_opt_{cp}.pt", weights_only=True))

        # Set enhancing transforms
        if rep_mode == RepresentationModes.monoplanar_128_32:
            latent_transforms = [  # Z x X x F  Single latent
                lambda l: l,  # x z
                lambda l: torch.flip(l, dims=[0]),  # x negz
                lambda l: torch.flip(l, dims=[1]),  # negx z
                lambda l: torch.flip(l, dims=[0, 1]),  # negx negz
                lambda l: torch.transpose(l, dim0=0, dim1=1),  # z x
                lambda l: torch.flip(torch.transpose(l, dim0=0, dim1=1), dims=[0]),  # z negx
                lambda l: torch.flip(torch.transpose(l, dim0=0, dim1=1), dims=[1]),  # negz x
                lambda l: torch.flip(torch.transpose(l, dim0=0, dim1=1), dims=[0, 1]),  # negz negx
            ]
            x_transforms = [
                lambda xyz: xyz,
                lambda xyz: xyz * torch.tensor([[1.0, 1.0, -1.0]], device=xyz.device),
                lambda xyz: xyz * torch.tensor([[-1.0, 1.0, 1.0]], device=xyz.device),
                lambda xyz: xyz * torch.tensor([[-1.0, 1.0, -1.0]], device=xyz.device),
                lambda xyz: xyz[..., [2, 1, 0]],
                lambda xyz: xyz[..., [2, 1, 0]] * torch.tensor([[1.0, 1.0, -1.0]], device=xyz.device),
                lambda xyz: xyz[..., [2, 1, 0]] * torch.tensor([[-1.0, 1.0, 1.0]], device=xyz.device),
                lambda xyz: xyz[..., [2, 1, 0]] * torch.tensor([[-1.0, 1.0, -1.0]], device=xyz.device),
            ]
        else:
            raise NotImplementedError()

        # Regularization factors
        decoder_l2_reg = settings['decoder_l2']
        decoder_l1_reg = settings['decoder_l1']
        latent_l2_reg = settings['latent_l2']
        latent_l1_reg = settings['latent_l1']
        latent_tv_reg = settings['latent_tv']

        # training loop
        steps = settings['train_steps']
        start_step = cp * cp_every
        # # stratified sampling
        # corners_x = torch.arange(-1.0, .99999999, 2.0 / 127, device=self.get_device())
        # corners_z = torch.arange(-1.0, .99999999, 2.0 / 127, device=self.get_device())
        # corners_y = torch.arange(-.5, .4999999, 1.0 / 3, device=self.get_device())
        # corners = torch.cartesian_prod(corners_x, corners_y, corners_z)
        # cell_size = torch.tensor([[2.0 / 127, 1.0 / 3, 2.0 / 127]], device=self.get_device())

        clouds = [self.get_volume(id) for id in ids]
        steps_iterator = tqdm(range(start_step, steps), desc="Loss: -")
        x_scaler = torch.tensor([[1.0, 2.0, 1.0]], device=self.get_device())  # Used to map -0.5...0.5 in y to full range -1.0...1.0 in the rep
        for step in steps_iterator:
            alpha_progress = (step / (steps - 1))**2
            # clear gradients
            decoder_opt.zero_grad()
            latents_opt.zero_grad()
            step_loss_value = 0
            # generate sampling positions
            # x = corners + torch.rand_like(corners) * cell_size
            with torch.no_grad():
                x = torch.rand(64 * 1024, 3, device=self.get_device()) * 2 - 1.0
                x[:,1] *= 0.5

            for i, v in enumerate(clouds):
                latent_id = i
                with torch.no_grad():
                    ref_values = v(x)  # sample once from a single cloud
                # assume 8 different transforms
                for latent_transform, x_transform in zip(latent_transforms, x_transforms):
                    tl = latent_transform(latents[latent_id])
                    tx = x_transform(x)
                    # random_size = np.random.randint(64, 128)
                    # random_size = (random_size, random_size)#, 32 * random_size // 128)
                    # tl = tl.unsqueeze(0) #.unsqueeze(-1)
                    # tl = modeling.resampling(tl, random_size, mode='bicubic')
                    # # # tl = modeling.resample_grid(tl, random_size)
                    # tl = modeling.resampling(tl, (128, 128), mode='bicubic')[0]
                    # # tl = modeling.resample_grid(tl, (128, 128, 32))
                    # tl = tl.squeeze(-1).squeeze(0)
                    if rep_mode == RepresentationModes.monoplanar_128_32:
                        inf_values = eval_monoplanar_128x32(
                            tx * x_scaler, upsampler(tl.unsqueeze(0))[0], decoder
                        )
                    else:
                        raise NotImplementedError()
                    loss = alpha_progress * torch.nn.functional.l1_loss(inf_values, ref_values, reduction='sum') + \
                            (1 - alpha_progress) * torch.nn.functional.mse_loss(inf_values, ref_values, reduction='sum')
                    loss_value = loss.item()

                    if decoder_l2_reg > 0:
                        loss += regularizer_l2(decoder) * decoder_l2_reg
                        loss += regularizer_l2(upsampler) * decoder_l2_reg
                    if decoder_l1_reg > 0:
                        loss += regularizer_l1(decoder) * decoder_l1_reg
                        loss += regularizer_l1(upsampler) * decoder_l1_reg
                    if latent_l1_reg > 0:
                        loss += regularizer_latent_l1(tl) * latent_l1_reg
                    if latent_l2_reg > 0:
                        loss += regularizer_latent_l2(tl) * latent_l2_reg
                    if latent_tv_reg > 0:
                        if rep_mode == RepresentationModes.monoplanar_128_32:
                            loss += regularizer_monoplanar_latent_tv(tl) * latent_tv_reg
                        else:
                            raise NotImplementedError()
                    loss.backward()

                    step_loss_value += loss_value
            ave_loss = step_loss_value / N / len(latent_transforms)
            steps_iterator.set_description(f"Loss: {ave_loss}")
            decoder_opt.step()
            latents_opt.step()
            decoder_sch.step()
            latents_sch.step()

            if (step + 1) % cp_every == 0:  # save checkpoint
                current_cp = (step + 1) // cp_every
                torch.save(decoder.state_dict(), self.workplace +f"/decoder/decoder_{current_cp}.pt")
                torch.save(upsampler.state_dict(), self.workplace +f"/decoder/upsampler_{current_cp}.pt")
                torch.save(latents, self.workplace +f"/decoder/latents_{current_cp}.pt")
                torch.save(decoder_opt.state_dict(), self.workplace + f"/decoder/decoder_opt_{current_cp}.pt")
                torch.save(latents_opt.state_dict(), self.workplace + f"/decoder/latents_opt_{current_cp}.pt")

    def download_pretrained(self):
        """
        https://drive.google.com/file/d/1UTf-S8sM6m8t5IuCsbyvdZuuFs7cs9LE/view?usp=sharing
        """
        import gdown
        import zipfile

        url = f"https://drive.google.com/file/d/"
        os.makedirs(self.workplace, exist_ok=True)
        zip_path = self.workplace + "/pretrained.zip"
        # Download the ZIP file
        print(f"Downloading file from Google Drive: {url}")
        file_id="1gtdgT9R4ZpxxC-8Id77N5MSw0j7jops6"
        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}",
            zip_path
        )
        #gdown.download(id="1gtdgT9R4ZpxxC-8Id77N5MSw0j7jops6", output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.workplace)
        # Optionally, remove the ZIP file after extraction
        os.remove(zip_path)

    def encode_to_latent(self, rep, initial_latent: typing.Optional[torch.Tensor] = None):
        decoder, upsampler, cp = self.get_decoder()
        settings = self.settings['encoder']
        rep_mode = self.settings['representation_mode']
        latent_l2_reg = settings['latent_l2']
        latent_l1_reg = settings['latent_l1']
        latent_tv_reg = settings['latent_tv']
        steps = settings['train_steps']
        if initial_latent is None:
            if rep_mode == RepresentationModes.monoplanar_128_32:
                latent = torch.nn.Parameter(torch.zeros(128, 128, 32, device=self.get_device()))
            else:
                raise NotImplementedError()
        else:
            latent = torch.nn.Parameter(initial_latent)
        latent_opt, latent_sch = create_optimization_objects([latent], **settings)
        steps_iterator = tqdm(range(steps))
        corners_x = torch.arange(-1.0, .99999999, 2.0 / 127, device=self.get_device())
        corners_z = torch.arange(-1.0, .99999999, 2.0 / 127, device=self.get_device())
        corners_y = torch.arange(-.5, .4999999, 1.0 / 7, device=self.get_device())
        corners = torch.cartesian_prod(corners_x, corners_y, corners_z)
        cell_size = torch.tensor([[2.0 / 127, 1.0 / 7, 2.0 / 127]], device=self.get_device())
        x_scaler = torch.tensor([[1.0, 2.0, 1.0]], device=self.get_device())  # Used to map -0.5...0.5 in y to full range -1.0...1.0 in the rep
        for s in steps_iterator:
            latent_opt.zero_grad()
            # x = torch.rand_like(corners) * 2 - 1.0
            # x[:,1] *= 0.5
            x = corners + torch.rand_like(corners) * cell_size
            with torch.no_grad():
                ref_values = rep(x)
            if rep_mode == RepresentationModes.monoplanar_128_32:
                inf_values = eval_monoplanar_128x32(
                    x * x_scaler, upsampler(latent.unsqueeze(0))[0], decoder
                )
            else:
                raise NotImplementedError()
            loss = torch.nn.functional.l1_loss(inf_values, ref_values, reduction='sum')
            loss_value = loss.item()
            # loss += inf_values.abs().sum() * 1e-4
            if latent_l1_reg > 0:
                loss += regularizer_latent_l1(latent) * latent_l1_reg
            if latent_l2_reg > 0:
                loss += regularizer_latent_l2(latent) * latent_l2_reg
            if latent_tv_reg > 0:
                if rep_mode == RepresentationModes.monoplanar_128_32:
                    loss += regularizer_monoplanar_latent_tv(latent) * latent_tv_reg
                else:
                    raise NotImplementedError()
            loss.backward()
            steps_iterator.set_description_str(f"Loss: {loss_value}")
            latent_opt.step()
            latent_sch.step()
        return latent.detach()

    def enhance_latent(self, rep, encoded_latent: torch.Tensor):
        decoder, upsampler, cp = self.get_decoder()
        settings = self.settings['enhancer']
        rep_mode = self.settings['representation_mode']
        latent_l2_reg = settings['latent_l2']
        latent_l1_reg = settings['latent_l1']
        latent_tv_reg = settings['latent_tv']
        steps = settings['train_steps']
        latent = torch.nn.Parameter(encoded_latent, requires_grad=True)
        # get sampling weights from opt_latent in this initial state
        with torch.no_grad():
            weights = ((latent.detach().abs() ** 2.0)).mean(-1, keepdim=True) + .0001
            weights = modeling.gaussian_filter(weights.unsqueeze(0), sigma=2.0, kernel_size=5)[0]
            weights = weights.flatten()
        latent_opt, latent_sch = create_optimization_objects([latent], **settings)
        steps_iterator = tqdm(range(steps))
        x_scaler = torch.tensor([[1.0, 2.0, 1.0]], device=self.get_device())  # Used to map -0.5...0.5 in y to full range -1.0...1.0 in the rep
        cell_size = torch.tensor([[2.0 / 127, 1.0 / 1.0, 2.0 / 127]], device=self.get_device())
        offset = torch.tensor([[-1.0, 0., -1.0]], device=self.get_device())
        for s in steps_iterator:
            alpha = s / (steps - 1)
            latent_opt.zero_grad()
            corners_id = torch.multinomial(weights, 128*1024, replacement=True)
            sel_z = corners_id // 128
            sel_x = corners_id % 128
            sel_y = torch.zeros_like(sel_z)
            x = (torch.cat([sel_x.view(-1, 1), sel_y.view(-1, 1), sel_z.view(-1, 1)], dim=-1) + torch.rand(
                len(sel_z), 3, device=self.get_device()) - 0.5) * cell_size + offset
            # x = 2 * torch.rand(64 * 1024, 3, device=self.prefered_device()) - 1
            # x[..., 1] *= 0.5
            with torch.no_grad():
                ref_values = rep(x)
            if rep_mode == RepresentationModes.monoplanar_128_32:
                inf_values = eval_monoplanar_128x32(
                    x * x_scaler, upsampler(latent.unsqueeze(0))[0], decoder
                )
            loss_mse = torch.nn.functional.mse_loss(inf_values, ref_values, reduction='sum')
            loss_l1 = torch.nn.functional.l1_loss(inf_values, ref_values, reduction='sum')
            loss = (1 - alpha) * loss_mse + alpha * loss_l1
            # loss += inf_values.abs().sum() * 0.001  # sparsity
            if latent_tv_reg > 0:
                loss += regularizer_monoplanar_latent_tv(latent) * latent_tv_reg
            if latent_l1_reg > 0:
                loss += regularizer_latent_l1(latent) * latent_l1_reg
            if latent_l2_reg > 0:
                loss += regularizer_latent_l2(latent) * latent_l2_reg
            loss.backward()
            latent_opt.step()
            latent_sch.step()
        return latent.detach()

    def run_encoding(self, start_id = None, end_id = None):
        if end_id is None:
            end_id = self.get_number_of_clouds()
        os.makedirs(self.workplace + "/encoded/", exist_ok=True)
        if start_id is None:
            start_id = last_file_id(self.workplace + "/encoded", "latent_", ".pt")
        for id in (range(start_id, end_id)):
            print(f"Encoding {id}")
            vol = self.get_volume(id)
            latent = self.encode_to_latent(vol)
            torch.save(latent, self.workplace + f"/encoded/latent_{id}.pt")

    def run_enhancing(self, start_cloud : typing.Optional[int] = None, end_cloud = None):
        if end_cloud is None:
            end_cloud = self.get_number_of_clouds()
        # N_T = 14
        N_T = 14
        start_time = time.perf_counter()
        os.makedirs(self.workplace + "/enhanced/", exist_ok=True)
        start_id = start_cloud * N_T #  max(start_cloud * 14, last_file_id(self.workplace + "/enhanced", "latent_", ".pt"))
        start_id //= N_T

        # rotations = [np.pi * 0.5 * r / N_T for r in range(N_T)]
        # scales = [1.0 - 0.2 * (0.5 - np.cos(s * 2 * np.pi / N_T) * 0.5) for s in range(N_T)]

        scales = [1.0, 0.85, 0.95, 0.8, 0.95, 0.8, 0.9] * 2
        rotations = [np.pi * 0.5 * r / N_T for r in range(N_T)]

        eid = start_id * N_T
        for id in (range(start_id, end_cloud)):
            d = (time.perf_counter() - start_time) / max(0.001, id - start_id)
            print(f"Enhancing {id} ETA: {timedelta(seconds=self.get_number_of_clouds() - id) * d}")
            vol = self.get_volume(id)
            for scale, rotation in zip(scales, rotations):
                latent = self.get_encoded_latent(id)
                t_latent = latent_transform(latent, scale, rotation)
                t_vol = cloud_transform(vol, scale, rotation, self.get_device())
                e_latent = self.enhance_latent(t_vol, t_latent)
                torch.save(e_latent, self.workplace + f"/enhanced/latent_{eid}.pt")
                eid += 1

    def run_compute_normalization_stats(self):
        augmented_dataset_path = self.workplace + "/enhanced"
        result_file = f"{augmented_dataset_path}/stats.pt"
        rep_mode = self.settings['representation_mode']
        if rep_mode == RepresentationModes.monoplanar_128_32:
            latent_mean = torch.zeros(32, device=self.get_device())
            latent_max = 0.1 * torch.ones(32, device=self.get_device())
            latent_min = -0.1 * torch.ones(32, device=self.get_device())
            cloud_id = 0
            while os.path.exists(
                    augmented_dataset_path + f"/latent_{cloud_id}.pt"):
                l = torch.load(augmented_dataset_path + f"/latent_{cloud_id}.pt", map_location=self.get_device(), weights_only=True)
                latent_mean += (l[0, 0] + l[0, 127] + l[127, 0] + l[127, 127]) / 4
                latent_max = torch.maximum(latent_max, l.max(dim=0)[0].max(dim=0)[0])
                latent_min = torch.minimum(latent_min, l.min(dim=0)[0].min(dim=0)[0])
                cloud_id += 1
                if cloud_id % 1000 == 0:
                    print(f"[INFO] Analized {cloud_id} latents.")
        else:
            raise NotImplementedError()
        # latent_min -= 0.001
        # latent_max += 0.001
        latent_mean /= cloud_id
        latent_scale = torch.maximum(latent_max - latent_mean, latent_mean - latent_min)
        torch.save({
            'minimum': latent_min.cpu(),
            'maximum': latent_max.cpu(),
            'mean': latent_mean.cpu(),
            'scale': latent_scale.cpu()
        }, result_file)
        print(f"[INFO] Computed stats for {cloud_id} original latents.")

    def run_train_diffuser(self):
        device = self.get_device()
        settings = self.settings['diffuser']
        # Clear if rerun
        result_path = f"{self.workplace}/diffuser"
        augmented_dataset_path = self.workplace + "/enhanced"
        clouds_count = 0
        while os.path.exists(augmented_dataset_path + f"/latent_{clouds_count}.pt"):
            clouds_count += 1
        assert clouds_count >= settings['train_N']  # leave the rest for testing
        os.makedirs(f"{result_path}", exist_ok=True)
        clouds_count = settings['train_N']
        print(f"[INFO] Using {clouds_count} clouds to train the diffuser")
        diffuser, cp = self.get_diffuser()
        cp_every = settings['cp_every']
        start_step = cp * cp_every
        steps = settings.get('train_steps')
        if start_step >= steps:
            return
        print(f"[INFO] Starting at step {start_step}")
        stats = self.get_normalization_stats()
        scale = stats['scale'].view(1, -1, 1, 1)  # B, C, H, W
        mean = stats['mean'].view(1, -1, 1, 1)  # B, C, H, W

        def load_latents_batch(images: torch.Tensor, ids: typing.List[int]):
            assert len(ids) == images.shape[0]
            with torch.no_grad():
                for i, id in enumerate(ids):
                    images[i].copy_(torch.load(augmented_dataset_path + f"/latent_{id}.pt",
                                               map_location=self.get_device(), weights_only=True).permute(2, 0, 1))  # channels first
                images.add_(other=mean, alpha=-1.0)
                images.mul_(1.0 / scale)

        # storage for 1000 clouds on GPU (~2.6GB)
        loaded_images = settings['train_K']
        images = torch.zeros(loaded_images, diffuser.channels, diffuser.resolution, diffuser.resolution, device=device)

        optimizer, scheduler = create_optimization_objects(diffuser.diffusion.parameters(), **settings)
        if cp > 0:
            if os.path.exists(f"{result_path}/optimizer_{cp}.pt"):
                optimizer.load_state_dict(torch.load(f"{result_path}/optimizer_{cp}.pt", weights_only=True))
                print(f"[INFO] Starting from lr: {optimizer.param_groups[0]['lr']}")
        batch_size = settings['train_B']
        z_0 = torch.randn(1, diffuser.channels, diffuser.resolution, diffuser.resolution, device=device)
        loss_value = 0  # loss during the inner-batch
        steps_iterator = tqdm(range(start_step, steps))
        for s in steps_iterator:
            if s % 100 == 0:
                load_latents_batch(images, random_subset(clouds_count, loaded_images))
            diffuser.train()
            ids = random_subset(loaded_images, batch_size)
            tr = np.random.rand() < 0.5
            fx = np.random.rand() < 0.5
            fy = np.random.rand() < 0.5
            with torch.no_grad():
                batch = images[ids]
                if tr:
                    batch = batch.transpose(dim0=-1, dim1=-2)
                if fx:
                    batch = torch.flip(batch, dims=[2])
                if fy:
                    batch = torch.flip(batch, dims=[3])
            optimizer.zero_grad()
            loss = diffuser(batch)
            loss_value += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffuser.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (s + 1) % 100 == 0:
                steps_iterator.set_description_str(f"Loss: {loss_value/100}")
                loss_value = 0  # restart
            if s == 0 or (s + 1) % cp_every == 0:
                if s > 0:
                    print("[INFO] Saving cached")
                    torch.save(diffuser.denoiser.state_dict(),
                               f'{result_path}/denoiser_{(s + 1) // cp_every}.pt')
                    torch.save(optimizer.state_dict(), f"{result_path}/optimizer_{(s + 1) // cp_every}.pt")
                if True:
                    with torch.no_grad():
                        diffuser.eval()
                        # 1, C=FEATURESx3, H, W
                        generated_latent = diffuser.eval_generator(z_0)
                        plt.imshow(generated_latent[0, 32 // 2, :, :].detach().cpu(), cmap='seismic', vmin=-1.0,
                                   vmax=1.0)
                        plt.gca().axis('off')
                        plt.show()
                        plt.imshow(generated_latent[0, :, :, generated_latent.shape[-1] // 2].detach().cpu(),
                                   cmap='seismic', vmin=-1.0,
                                   vmax=1.0, aspect=generated_latent.shape[-2] / 32)
                        plt.gca().axis('off')
                        plt.show()

                        latent = self.denormalize_latent(generated_latent)
                        rec_rep = self.from_latent_to_rep(latent[0])
                        rec_slice = sample_slice(rec_rep, 2/256, vmin=-0.5, vmax=0.5, device=self.get_device())
                        plt.figure(figsize=(2, 1), dpi=512)
                        plt.imshow(rec_slice.cpu(), vmin=0, vmax=0.6, cmap='gist_heat_r')
                        plt.gca().axis('off')
                        plt.gca().invert_yaxis()
                        plt.show()

    def run_train_diffuser_acc(self):
        device = self.get_device()
        settings = self.settings['diffuser_acc']
        # Clear if rerun
        result_path = f"{self.workplace}/diffuser"
        augmented_dataset_path = self.workplace + "/enhanced"
        clouds_count = 0
        while os.path.exists(augmented_dataset_path + f"/latent_{clouds_count}.pt"):
            clouds_count += 1
        assert clouds_count >= settings['train_N']  # leave the rest for testing
        os.makedirs(f"{result_path}", exist_ok=True)
        clouds_count = settings['train_N']
        print(f"[INFO] Using {clouds_count} clouds to train the diffuser")
        diffuser, cp = self.get_diffuser()
        print(f"[INFO] Training diffuser with {sum(p.numel() for p in diffuser.parameters())} parameters")

        parallel_diffuser = torch.nn.DataParallel(diffuser)

        ema_diffuser = modeling.EMA(diffuser, decay=0.99)

        cp_every = settings['cp_every']
        start_step = cp * cp_every
        steps = settings.get('train_steps')
        if start_step >= steps:
            return
        print(f"[INFO] Starting at step {start_step}")
        stats = self.get_normalization_stats()
        scale = stats['scale'].view(1, -1, 1, 1)  # B, C, H, W
        mean = stats['mean'].view(1, -1, 1, 1)  # B, C, H, W

        def load_latents_batch(images: torch.Tensor, ids: typing.List[int]):
            assert len(ids) == images.shape[0]
            with torch.no_grad():
                for i, id in enumerate(ids):
                    images[i].copy_(torch.load(augmented_dataset_path + f"/latent_{id}.pt",
                                               map_location=self.get_device(), weights_only=True).permute(2, 0, 1))  # channels first
                images.add_(other=mean, alpha=-1.0)
                images.mul_(1.0 / scale)

        # storage for 1000 clouds on GPU (~2.6GB)
        loaded_images = settings['train_K']
        images = torch.zeros(loaded_images, diffuser.channels, diffuser.resolution, diffuser.resolution, device=device)

        optimizer, scheduler = create_optimization_objects(parallel_diffuser.parameters(), **settings)
        if cp > 0:
            if os.path.exists(f"{result_path}/optimizing_{cp}.pt"):
                opt_data = torch.load(f"{result_path}/optimizing_{cp}.pt", weights_only=True)
                parallel_diffuser.module.load_state_dict(opt_data['model'])
                optimizer.load_state_dict(opt_data['opt'])
                # only to fix the restart lr problem
                optimizer.param_groups[0]['lr'] = settings['lr'] #0.0001
                try:
                    scheduler.load_state_dict(opt_data['sch'])
                except:
                    print("[WARNING] Scheduler was changed!")
        batch_size = settings['train_B']
        z_0 = torch.randn(1, diffuser.channels, diffuser.resolution, diffuser.resolution, device=device)
        loss_value = 0  # loss during the inner-batch
        steps_iterator = tqdm(range(start_step, steps))
        scheduler.step()
        print(f"[INFO] Starting from lr: {optimizer.param_groups[0]['lr']}")
        for s in steps_iterator:
            if s % 100 == 0:
                load_latents_batch(images, random_subset(clouds_count, loaded_images))
            parallel_diffuser.train()
            ids = random_subset(loaded_images, batch_size)
            tr = np.random.rand() < 0.5
            fx = np.random.rand() < 0.5
            fy = np.random.rand() < 0.5
            with torch.no_grad():
                batch = images[ids]
                if tr:
                    batch = batch.transpose(dim0=-1, dim1=-2)
                if fx:
                    batch = torch.flip(batch, dims=[2])
                if fy:
                    batch = torch.flip(batch, dims=[3])
            optimizer.zero_grad()
            loss = parallel_diffuser(batch).sum()
            loss_value += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parallel_diffuser.module.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema_diffuser.update()
            if (s + 1) % 100 == 0:
                steps_iterator.set_description_str(f"Loss: {loss_value/100} LR: {optimizer.param_groups[0]['lr']}")
                loss_value = 0  # restart
            if s == 0 or (s + 1) % cp_every == 0:
                diffuser = ema_diffuser.ema_model
                if s > 0:
                    print("[INFO] Saving cached")
                    torch.save(diffuser.denoiser.state_dict(),
                               f'{result_path}/denoiser_{(s + 1) // cp_every}.pt')
                    torch.save({
                        'model': parallel_diffuser.module.state_dict(),
                        'opt': optimizer.state_dict(),
                        'sch': scheduler.state_dict()
                    }, f"{result_path}/optimizing_{(s + 1) // cp_every}.pt")
                if True:
                    with torch.no_grad():
                        diffuser.eval()
                        # 1, C=FEATURESx3, H, W
                        generated_latent = diffuser.eval_generator(z_0)
                        plt.imshow(generated_latent[0, 32 // 2, :, :].detach().cpu(), cmap='seismic', vmin=-1.0,
                                   vmax=1.0)
                        plt.gca().axis('off')
                        plt.show()
                        plt.imshow(generated_latent[0, :, :, generated_latent.shape[-1] // 2].detach().cpu(),
                                   cmap='seismic', vmin=-1.0,
                                   vmax=1.0, aspect=generated_latent.shape[-2] / 32)
                        plt.gca().axis('off')
                        plt.show()

                        latent = self.denormalize_latent(generated_latent)
                        rec_rep = self.from_latent_to_rep(latent[0])
                        rec_slice = sample_slice(rec_rep, 2/256, vmin=-0.5, vmax=0.5, device=self.get_device())
                        plt.figure(figsize=(2, 1), dpi=512)
                        plt.imshow(rec_slice.cpu(), vmin=0, vmax=0.6, cmap='gist_heat_r')
                        plt.gca().axis('off')
                        plt.gca().invert_yaxis()
                        plt.show()

    def from_latent_to_rep(self, latent: torch.Tensor):
        decoder, upsampler, cp = self.get_decoder()
        if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:

            def rep(x):
                up_latent = upsampler(latent.unsqueeze(0))[0]
                x_scaler = torch.tensor([1.0, 2.0, 1.0],
                                        device=self.get_device())  # Used to map -0.5...0.5 in y to full range -1.0...1.0 in the rep

                return eval_monoplanar_128x32(
                    x * x_scaler.view(*([1]*(len(x.shape)-1)), 3),
                    up_latent,
                    decoder
                )
        else:
            raise NotImplementedError()
        return rep

    def from_grid_to_volume(self, grid: torch.Tensor):
        return modeling.Volume(
            grid,
            fit_box=(1.0, 0.5, 1.0)
        )

    def from_rep_to_grid(self, rep, resolution: int, *, noise: float = 0.0):
        return modeling.reconstruct_grid3d(
            rep,
            ymin=-0.5, ymax=0.5,
            resx = resolution,
            resy = resolution // 2,
            resz = resolution,
            noise=noise,
            device=self.get_device())

    def from_latent_to_grid(self, latent: torch.Tensor, *,
                            resolution: int = 128,
                            noise: float = 0.0
                            ):
        return self.from_rep_to_grid(
            self.from_latent_to_rep(latent),
            resolution=resolution,
            noise=noise
        )

    def decode_latent(self, latent: torch.Tensor,
                      resolution: int = 128,
                      noise: float = 0.0,
                      batch_size : typing.Optional[int] = 128*64*128
                      ):
        latent_decoding = self.get_latent_decoder()
        return latent_decoding(latent,
                               resx=resolution,
                               resy=resolution//2,
                               resz=resolution,
                               noise=noise,
                               batch_size=batch_size)

    def normalize_latent(self, latent: torch.Tensor):
        stats = self.get_normalization_stats()
        if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:
            if len(latent.shape) == 3:
                latent = (latent - stats['mean'].view(1,1,-1))/stats['scale'].view(1,1,-1)
                return latent.permute(2, 0, 1)
            elif len(latent.shape) == 4:
                latent = (latent - stats['mean'].view(1,1,1,-1))/stats['scale'].view(1,1,1,-1)
                return latent.permute(0, 3, 1, 2)
            else:
                raise NotImplemented()
        else:
            raise NotImplemented()

    def denormalize_latent(self, latent: torch.Tensor):
        stats = self.get_normalization_stats()
        if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:
            if len(latent.shape) == 3:
                latent = latent.permute(1, 2, 0)
                return latent * stats['scale'].view(1, 1, -1) + stats['mean'].view(1, 1, -1)
            elif len(latent.shape) == 4:
                latent = latent.permute(0, 2, 3, 1)
                return latent * stats['scale'].view(1, 1, 1, -1) + stats['mean'].view(1, 1, 1, -1)
            else:
                raise NotImplemented()
        else:
            raise NotImplemented()

    def random_gaussian_latent(self, batch_size: typing.Optional[int] = None):
        if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:
            if batch_size is None:
                return torch.randn(32, 128, 128, device=self.get_device())
            return torch.randn(batch_size, 32, 128, 128, device=self.get_device())
        else:
            raise NotImplemented()

    def denoise(self,
                noise: torch.Tensor,
                steps: typing.List[int], *,
                eta: float = 1.0,
                callback: typing.Optional[typing.Callable[[CallbackInfo], None]] = None
                ):
        """
        Denoises the initial noise (assumed at step steps[0]) transitioning from steps[i] to steps[i+1] using DDIM sampling method.

        Returns
        -------
        torch.Tensor
            The state after all steps.

        Parameters
        ----------
        noise: torch.Tensor
            The noise corresponding to the step index steps[0].
        steps: typing.List[int]
            The sequence of time steps will be transitioned. Steps must appear decreasingly, all in range [0..T]
        eta: float
            Hyper-parameter of the DDIM method. 0 determined solely from x_T, 1.0 equivalent to DDPM.
        callback:
            Function that will be called after each transition with the current step index and estimated
        """
        if len(steps) <= 1:  # nothing to do
            return noise
        diffuser, _ = self.get_diffuser()
        diffuser.eval()
        with torch.no_grad():
            if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:
                if len(noise.shape) == 3:  # no batch
                    # return diffuser.eval_generator_ddim(noise.unsqueeze(0), diffuser.timesteps)[0]
                    def wrap_callback(step, total_steps, timestep, l):
                        callback(CallbackInfo(
                            self,
                            step,
                            total_steps,
                            timestep,
                            l[0], True))
                    return diffuser.reverse_diffusion_DDIM(
                        noise.unsqueeze(0), steps=steps, eta=eta, callback=wrap_callback if callback is not None else None)[0]
                if len(noise.shape) == 4:  # batch
                    def wrap_callback(step, total_steps, timestep, l):
                        callback(CallbackInfo(self, step, total_steps, timestep, l, True))
                    # return diffuser.eval_generator_ddim(noise, diffuser.timesteps)
                    return diffuser.reverse_diffusion_DDIM(noise, steps=steps, eta=eta, callback=wrap_callback if callback is not None else None)
                raise Exception()
            else:
                raise NotImplemented()

    def posterior_sample(self,
                         noise: torch.Tensor,
                         steps: typing.List[int],
                         y: torch.Tensor,
                         A: typing.Callable[[torch.Tensor], torch.Tensor],
                         *,
                         eta: float = 1.0,
                         weight: float = 1.0,
                         ema_factor: float = 1.0,
                         callback: typing.Optional[typing.Callable[[CallbackInfo], None]] = None
                         ):
        if len(steps) <= 1:  # nothing to do
            return noise
        assert len(noise.shape) == 3
        diffuser, _ = self.get_diffuser()
        diffuser.eval()
        with torch.no_grad():
            if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:
                def wrap_callback(step, total_steps, timestep, l):
                    callback(CallbackInfo(self, step, total_steps, timestep, l[0], True))
                return diffuser.posterior_sampling_DPS_DDIM(
                    noise,
                    steps,
                    y=y,
                    A=A,
                    eta=eta,
                    weight=weight,
                    ema_factor=ema_factor,
                    callback=wrap_callback if callback is not None else None)[0]
            else:
                raise NotImplemented()

    def diffuse(self, latent: torch.Tensor, t: int, steps: int, *, noise: typing.Optional[torch.Tensor] = None):
        diffuser, _ = self.get_diffuser()
        diffuser.eval()
        if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:
            if len(latent.shape) == 3:  # no batch
                return diffuser.forward_diffusion(latent.unsqueeze(0), t, steps, noise=noise)[0]
            if len(latent.shape) == 4:  # batch
                return diffuser.forward_diffusion(latent, t, steps, noise=noise)
            raise Exception()
        else:
            raise NotImplemented()

    def sample_normalized_latent(self,
                                   start_noise: typing.Optional[torch.Tensor] = None,
                                   start_step: typing.Optional[int] = None, *,
                                   samples: int = 1000,
                                   scheduler_gamma: float = 0.5,
                                   eta: float = 1.0,
                                   y: typing.Optional[torch.Tensor] = None,
                                   A: typing.Optional[typing.Callable[[torch.Tensor],torch.Tensor]] = None,
                                   guiding_strength: float = 1.0,
                                   ema_factor: float = 1.0,
                                   callback: typing.Optional[typing.Callable[[CallbackInfo], None]] = None
                                   ):
        """
        Generates a normalized latent from the diffuser, x~p(x) or x~p(x|y) if y is provided.
        Parameters
        ----------
        start_noise: torch.Tensor | None
            The initial noise. If None, a proper gaussian noise is used.
            If the shape is BxFxRxR, the batch size B is used to generate a batch of latents instead of a single latent.
            Posterior sampling does not support batches.
        start_step: int | None
            If start_noise is not None, this is the corresponding step index. If None, the step T (timesteps) is assumed.
        samples: int
            Number of step samples from start_step to 0, distributed with a power scheduler. T*((t/T) ** gamma)
        scheduler_gamma: float
            gamma value used for the scheduler of the steps.
        eta: float
            eta parameter for the DDIM sampling. eta = 0 is solely determined by x_T, eta=1 is equivalent to DDPM.
        y: torch.Tensor
            Measurement as sampling condition p(x|y) assuming a forward model exists: y = A(x) + gaussian_noise
            If measurement y is provided, the gradient of A is used to guide the posterior sampling towards an optimization goal.
        A: typing.Callable[[torch.Tensor], torch.Tensor] | None
            Represents the function emulating the measurement.
            The latent for the gradient computation is provided (normalized).
            Default (None) case is assumed the identity y = x.
        guiding_strength: float
            Scaler for the gradient guiding the posterior sampling.
            0 will represent an unconditional sampling. Recommended 1. Greater than 1 could overshoot and create artifacts.
        callback:
            If callback is provided, the estimated latent for each denoising step it is sent to the function with the step index.

        Returns
        -------
        torch.Tensor
            The generated latent or latents normalized. (FxRxR) or (BxFxRxR)

        Examples
        --------
        >>> import cloudy
        >>> pipeline = cloudy.create_pipeline('./test')
        >>> pipeline.download_pretrained()
        >>> # Unconditional sampling
        >>> latent = pipeline.generate_normalized_latent(samples=50)
        >>> # Conditional sampling (Inpainting)
        >>> # load reference cloud
        >>> mask = ...., reference_cloud = ...
        >>> masked_reference = mask * reference_cloud
        >>> latent = pipeline.generate_normalized_latent(samples=50, criteria=lambda l: (pipeline.from_latent_to_grid(pipeline.denormalize_latent(l)) * mask - masked_reference)**2)
        """
        if A is None:
            A = lambda l: l
        assert y is None or start_noise is None or len(start_noise.shape) == 3, \
            "Posterior sampling is not supported for batch generation. Generate a single latent at a time."
        assert start_step is None or start_noise is not None, \
            "start_step requires a existing noise state to be provided."
        if start_noise is None:
            start_noise = self.random_gaussian_latent()
        diffuser, _ = self.get_diffuser()
        if start_step is None:
            start_step = diffuser.timesteps
        x = np.arange(0.0, 1.00001, 1.0/samples)
        x = x ** scheduler_gamma  # scheduler
        x = (x * start_step).astype(np.int32)
        steps = list(x)
        steps[1] = (steps[0] + steps[2])//2
        steps.reverse()
        if y is None:
            return self.denoise(
                start_noise,
                steps=steps,
                eta=eta,
                callback=callback
            )
        else:
            return self.posterior_sample(
                start_noise,
                steps=steps,
                y=y,
                A=A,
                eta=eta,
                weight=guiding_strength,
                ema_factor=ema_factor,
                callback=callback
            )

    def sample_latent(self,
                        start_noise: typing.Optional[torch.Tensor] = None,
                        start_step: typing.Optional[int] = None,
                        *,
                        samples: int = 200,
                        scheduler_gamma: float = 0.5,
                        eta: float = 1.0,
                        y: typing.Optional[torch.Tensor] = None,
                        A: typing.Optional[typing.Callable[[torch.Tensor],torch.Tensor]] = None,
                        guiding_strength: float = 1.0,
                        ema_factor: float = 1.0,
                        callback: typing.Optional[typing.Callable[[CallbackInfo], None]] = None
        ):
        """
        Generates a latent from the diffuser, x~p(x) or x~p(x|y) if y is provided.
        Parameters
        ----------
        start_noise: torch.Tensor | None
            The initial noise. If None, a proper gaussian noise is used.
            If the shape is BxRxRxF, the batch size B is used to generate a batch of latents instead of a single latent.
            Posterior sampling does not support batches.
        start_step: int | None
            If start_noise is not None, this is the corresponding step index. If None, the step T (timesteps) is assumed.
        samples: int
            Number of step samples from start_step to 0, distributed with a power scheduler. T*((t/T) ** gamma)
        scheduler_gamma: float
            gamma value used for the scheduler of the steps.
        eta: float
            eta parameter for the DDIM sampling. eta = 0 is solely determined by x_T, eta=1 is equivalent to DDPM.
        y: torch.Tensor
            Measurement as sampling condition p(x|y) assuming a forward model exists: y = A(x) + gaussian_noise
            If measurement y is provided, the gradient of A is used to guide the posterior sampling towards an optimization goal.
        A: typing.Callable[[torch.Tensor], torch.Tensor] | None
            Represents the function emulating the measurement.
            The latent for the gradient computation is provided (denormalized).
            Default (None) case is assumed the identity y = x.
        guiding_strength: float
            Scaler for the gradient guiding the posterior sampling.
            0 will represent an unconditional sampling. Recommended 1. Greater than 1 could overshoot and create artifacts.
        callback:
            If callback is provided, the estimated latent for each denoising step it is sent to the function with the step index.

        Returns
        -------
        torch.Tensor
            The generated latent or latents denormalized. (RxRxF) or (BxRxRxF)

        Examples
        --------
        >>> import cloudy
        >>> pipeline = cloudy.create_pipeline('./test')
        >>> pipeline.download_pretrained()
        >>> # Unconditional sampling
        >>> latent = pipeline.generate_latent(samples=50)
        >>> # Conditional sampling (Inpainting)
        >>> # load reference cloud
        >>> mask = ...., reference_cloud = ...
        >>> masked_reference = mask * reference_cloud
        >>> latent = pipeline.generate_latent(samples=50, y=masked_reference, A=lambda l: pipeline.from_latent_to_grid(l) * mask)
        """
        assert y is None or start_noise is None or len(start_noise.shape) == 3, \
            "Posterior sampling is not supported for batch generation. Generate a single latent at a time."

        if y is not None and A is None:
            A = lambda x0hat: x0hat

        def wrap_A(l):
            return A(self.denormalize_latent(l))

        latent = self.sample_normalized_latent(
            start_noise=start_noise,
            start_step=start_step,
            samples=samples,
            scheduler_gamma=scheduler_gamma,
            eta=eta,
            y = y,
            A = wrap_A if A is not None else None,
            guiding_strength=guiding_strength,
            ema_factor=ema_factor,
            callback=callback
        )
        return self.denormalize_latent(latent)

    def sample_volume(self,
                    resolution: int = 128,
                    start_noise: typing.Optional[torch.Tensor] = None,
                    start_step: typing.Optional[int] = None,
                    samples: int = 200,
                    scheduler_gamma: float = 1.0,
                    *,
                    y: typing.Optional[torch.Tensor] = None,
                    A: typing.Optional[typing.Callable[[torch.Tensor],torch.Tensor]] = None,
                    guiding_strength: float = 1.0,
                    ema_factor: float = 1.0,
                    decoding_resolution: int = 128,
                    decoding_noise: float = 0.0,
                    out_latent: typing.Optional[torch.Tensor] = None,
                    callback: typing.Optional[typing.Callable[[CallbackInfo], None]] = None
        ):
        def wrap_A(latent):
            g = self.decode_latent(latent, decoding_resolution, decoding_noise)
            g = g * (g > 0.003).float()
            return A(g)
        latent = self.sample_latent(
            start_noise=start_noise,
            start_step=start_step,
            samples=samples,
            scheduler_gamma=scheduler_gamma,
            y=y, A=None if A is None else wrap_A,
            guiding_strength=guiding_strength,
            ema_factor=ema_factor,
            callback=callback)
        if out_latent is not None:
            out_latent.copy_(latent)
        if resolution <= 0:
            return None

        g = self.decode_latent(latent, resolution)
        g = self.clean_volume(g)
        return g

    def clean_volume(self, g: torch.Tensor):
        return g * (g > 0.003).float() * (g < 0.9).float()

    def reconstruct_volume(self,
            y: torch.Tensor,
            A_factory: typing.Callable[[int], typing.Callable[[torch.Tensor], torch.Tensor]],
            L_factory: typing.Optional[typing.Callable[[int, torch.Tensor, torch.Tensor], typing.Callable[[], torch.Tensor]]] = None,
            optimizer: typing.Optional[torch.optim.Optimizer] = None,
            resolution: int = 128,
            samples: int = 100,
            scheduler_gamma: float = 0.8,
            weights: typing.Optional[typing.Union[float, typing.List[float]]] = None,
            ema_factor: float = 1.0,
            decoding_resolution: typing.Union[int, typing.List[int]] = 128,
            decoding_noise: typing.Union[float, typing.List[float]] = 0.0,
            optimization_steps: typing.Union[int, typing.List[int]] = 100,
            optimization_passes: int = 10,
            out_latent: typing.Optional[torch.Tensor] = None,
            callback: typing.Optional[typing.Callable[[CallbackInfo], None]] = None
    ):
        assert (L_factory is None) == (optimizer is None)
        if isinstance(weights, float):
            weights = [weights] * optimization_passes
        if isinstance(decoding_resolution, int):
            decoding_resolution = [decoding_resolution]*optimization_passes
        if isinstance(decoding_noise, float):
            decoding_noise = [decoding_noise] * optimization_passes

        current_pass = -1
        current_subpass = 'dps'

        def wrap_callback(ci: CallbackInfo):
            return callback(CallbackInfo(
                self,
                ci.step,
                ci.total_steps,
                ci.sampled_timestep,
                ci.latent,
                normalized=False,
                pass_index=current_pass,
                subpass=current_subpass
            ))

        def optimize_parameters(p: int, l: torch.Tensor):
            with torch.no_grad():
                g = self.decode_latent(l, resolution=resolution)
                g = self.clean_volume(g)
            L = L_factory(p, g, y)
            iterations = tqdm(range(optimization_steps), desc="Opt")
            for s in iterations:
                optimizer.zero_grad()
                loss = L()
                loss.backward()
                optimizer.step()
                iterations.set_postfix_str(f"Loss: {loss.item()}")
                if callback is not None:
                    callback(CallbackInfo(self, s, optimization_steps, 0, l, False, pass_index=current_pass, subpass='opt'))

        def refine(A, l: torch.Tensor, decoding_resolution: int, decoding_noise: float):
            l.requires_grad_(True)
            opt = torch.optim.NAdam([l], lr=0.0005)
            y_ema = None
            iterations = tqdm(range(optimization_steps // 10), "Refinement")
            for s in iterations:
                opt.zero_grad()
                g = self.decode_latent(l, decoding_resolution, decoding_noise)
                g = self.clean_volume(g)
                y_hat = A(g)
                if y_ema is None or ema_factor == 1.0:
                    y_ema = y_hat
                else:
                    y_ema = modeling.ema_diff(y_hat, y_ema, ema_factor)
                loss = ((y - y_ema)**2).sum()
                # loss += regularizer_latent_l1(l) * 0.001
                loss += regularizer_monoplanar_latent_tv(l) * 0.1
                loss.backward()
                opt.step()
                y_ema = y_ema.detach()
                iterations.set_postfix_str(f"Loss: {loss.item()}")
                if callback is not None:
                    callback(CallbackInfo(self, s, optimization_steps // 4,0, l.detach(), False, pass_index=current_pass, subpass='opt'))
            l.requires_grad_(False)

        diffuser,_ = self.get_diffuser()
        start_timesteps = [diffuser.timesteps - int(0.5 * diffuser.timesteps * (i / (optimization_passes - 1)) ** 2.0) for i
                           in range(optimization_passes)]
        # Start from an unconditional generated volume
        latent = self.sample_latent(
            samples=samples,
            scheduler_gamma=scheduler_gamma,
            callback=wrap_callback if callback is not None else None
        )

        for p, start_t in enumerate(start_timesteps):
            if optimizer is not None:
                optimize_parameters(p, latent)  # optimize phi
            # back to noise
            noise = self.diffuse(self.normalize_latent(latent), 0, start_t)
            # sample p(x|y)
            A = A_factory(p)
            self.sample_volume(
                start_noise=noise,
                start_step=start_t,
                samples=samples,
                resolution=0,  # no need for volume
                y=y,
                A=A,
                decoding_resolution=decoding_resolution[p],
                decoding_noise=decoding_noise[p],
                guiding_strength=weights[p],
                ema_factor=ema_factor,
                out_latent=latent,
                callback=wrap_callback if callback is not None else None
            )
            # refine x wrt L
            if p >= optimization_passes//3 and p < optimization_passes - 2:
                refine(A, latent,
                       decoding_resolution=decoding_resolution[p],
                       decoding_noise=decoding_noise[p])
        if out_latent is not None:
            out_latent.copy_(latent)
        return self.decode_latent(latent, resolution=resolution)


