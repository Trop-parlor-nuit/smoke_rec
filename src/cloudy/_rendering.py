import rendervous as rdv
import torch
import typing
from . import _modeling as modeling
import numpy as np


def camera_poses(
        origins : typing.Union[typing.Tuple[float, float, float], rdv.vec3, torch.Tensor],
        targets : typing.Union[None, typing.Tuple[float, float, float], rdv.vec3, torch.Tensor] = None,
        up_directions: typing.Union[None, typing.Tuple[float, float, float], rdv.vec3, torch.Tensor] = None
):
    if targets is None:
        targets = (0.0, 0.0, 0.0)
    if up_directions is None:
        up_directions = (0.0, 1.0, 0.0)
    if isinstance(origins, tuple):
        origins = torch.tensor([*origins], dtype=torch.float)
    if isinstance(targets, tuple):
        targets = torch.tensor([*targets], dtype=torch.float)
    if isinstance(up_directions, tuple):
        up_directions = torch.tensor([*up_directions], dtype=torch.float)
    origins, targets, up_directions = rdv.broadcast_args_to_max_batch((origins, (3,)), (targets, (3,)), (up_directions, (3,)))
    origins = origins.to(rdv.device())
    targets = targets.to(rdv.device())
    up_directions = up_directions.to(rdv.device())
    camera_poses = torch.cat([origins, rdv.vec3.normalize(targets - origins), up_directions], dim=-1)
    if len(camera_poses.shape) == 1:
        return camera_poses.unsqueeze(0)
    return camera_poses


def environment_objects(environment: torch.Tensor):
    ds = rdv.DependencySet()
    ds.add_parameters(environment_tensor = environment)
    ds.requires(rdv.medium_environment)
    ds.requires(rdv.medium_environment_sampler_quadtree)
    return ds.environment, ds.environment_sampler


def oct_camera_poses(N: int, *, seed: int = 15, distance: float = 2.0):
    return rdv.oct_camera_poses(N, seed=seed, radius=distance)


class DifferentiableRendering(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        rendering_process, *parameters = args
        pars = [torch.nn.Parameter(p, requires_grad=p.requires_grad) for p in parameters]
        ctx.save_for_backward(*pars)
        with torch.enable_grad():
            ds, outputs = rendering_process(*pars)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
        ctx.ds = ds
        ctx.outputs = outputs
        return [o.detach().clone() for o in outputs] if len(outputs) > 1 else outputs[0].detach().clone()

    @staticmethod
    def backward(ctx: typing.Any, *grad_outputs: typing.Any) -> typing.Any:
        ds = ctx.ds
        parameters = ctx.saved_tensors
        outputs = ctx.outputs
        torch.autograd.backward(outputs, grad_outputs, inputs=parameters)
        ds.backward_dependency()
        grads = [None if p.grad is None else p.grad.clone() for p in parameters]
        return None, *grads


def background_radiance(
        environment: typing.Union[torch.Tensor, typing.Callable[[torch.Tensor], torch.Tensor]],
        *,
        camera_poses: torch.Tensor,
        **settings
):
    """
    Renders the background radiance from an environment as an xr environment map,
    or any callable: directions -> R
    """
    ds = rdv.DependencySet()
    ds.add_parameters(
        camera_poses=camera_poses
    )
    ds.requires(rdv.camera_sensors,
                width=settings.get('width', 512),
                height=settings.get('height', 512),
                jittered=settings.get('jittered', False))
    if isinstance(environment, torch.Tensor):  # support xr projection from now
        sampled_positions = ds.camera.capture(rdv.xr_projection(ray_input=True))
        return modeling.sample_grid2d(environment, sampled_positions, align_corners=False)
    else:
        sampled_directions = ds.camera.capture(rdv.ray_direction())
        return environment(sampled_directions)


def reconstruct_environment(environment_model, *, width: int, height: int):
    """
    Maps an environment map given by a model to an xr image.
    """
    stepw = 2.0 / width
    steph = 2.0 / height
    xs = torch.arange(-1.0 + stepw*0.5, 1.0, stepw, device=rdv.device())
    ys = torch.arange(-1.0 + steph*0.5, 1.0, steph, device=rdv.device())
    p = torch.cartesian_prod(ys, xs)[:, [1,0]]
    angles_x = p[:, 0:1] * np.pi
    angles_y = p[:, 1:2] * np.pi/2
    y = -torch.sin(angles_y)
    r = torch.cos(angles_y)
    x = torch.sin(angles_x)*r
    z = torch.cos(angles_x)*r
    w = torch.cat([x,y,z], dim=-1)
    E = environment_model(w)
    return E.reshape(height, width, -1).contiguous()


def transmittance(
        grid: torch.Tensor,
        *,
        camera_poses: torch.Tensor,
        **settings
):
    """
    Computes the transmittance maps through a volume g
    """
    def rendering_process(pgrid):
        ds = rdv.DependencySet()
        ds.add_parameters(
            camera_poses=camera_poses,
            sigma_tensor=pgrid
        )
        ds.requires(rdv.medium_transmittance_DDA)
        ds.requires(rdv.camera_sensors,
                    width=settings.get('width', 512),
                    height=settings.get('height', 512),
                    jittered=settings.get('jittered', False))
        return ds, ds.camera.capture(
            ds.transmittance,
            fw_samples=settings.get('samples', 1),
            batch_size=512*512
        )

    return DifferentiableRendering.apply(rendering_process, grid)
    # return rdv.parameterized_call(rendering_process, grid)


def scattered(
        grid: torch.Tensor,
        *,
        camera_poses: torch.Tensor,
        scattering_albedo: typing.Union[float, typing.Tuple[float, float, float], torch.Tensor],
        environment: typing.Union[torch.Tensor, rdv.MapBase],
        phase_g: float,
        majorant: float,
        environment_sampler: typing.Optional[rdv.MapBase] = None,
        mode: typing.Union[str, typing.Literal['drt', 'sps']] = 'drt',
        **settings
):
    if isinstance(scattering_albedo, float):
        scattering_albedo = (scattering_albedo, scattering_albedo, scattering_albedo)
    if isinstance(scattering_albedo, tuple):
        scattering_albedo = torch.tensor([*scattering_albedo], device=rdv.device())
    def rendering_process(pgrid):
        ds = rdv.DependencySet()
        if isinstance(environment, torch.Tensor):
            ds.add_parameters(
                environment_tensor = environment
            )
        else:
            ds.add_parameters(
                environment=environment
            )
        if environment_sampler is not None:
            ds.add_parameters(
                environment_sampler=environment_sampler
            )
        ds.add_parameters(
            camera_poses=camera_poses,
            sigma_tensor=pgrid,
            majorant_tensor=torch.tensor([majorant, 100000], device=rdv.device()),
            scattering_albedo_tensor=scattering_albedo,
            emission=rdv.ZERO,
            phase_g_tensor=torch.ones([2,2,2,1], device=rdv.device())*phase_g,
        )
        ds.requires(rdv.medium_transmittance_RT)
        ds.requires(rdv.medium_phase_HG)
        ds.requires(rdv.medium_phase_sampler_HG)
        ds.requires(rdv.medium_environment)
        ds.requires(rdv.medium_environment_sampler)
        if mode == 'drt':
            ds.requires(rdv.medium_radiance_path_integrator_NEE_DRT)
        elif mode == 'sps':
            ds.requires(rdv.medium_radiance_path_integrator_NEE_SPS)
        else:
            raise NotImplemented(f'Not supported mode {mode}')

        ds.requires(rdv.camera_sensors,
                    width=settings.get('width', 512),
                    height=settings.get('height', 512),
                    jittered=settings.get('jittered', True))
        return ds, ds.camera.capture(ds.radiance,
                                     fw_samples=settings.get('samples', 1),
                                     bw_samples=settings.get('samples_bw', 1)
                                     )
    return DifferentiableRendering.apply(rendering_process, grid)


def scattered_environment(
        environment: torch.Tensor,
        *,
        grid: torch.Tensor,
        camera_poses: torch.Tensor,
        scattering_albedo: typing.Union[float, typing.Tuple[float, float, float], torch.Tensor],
        phase_g: float,
        majorant: float,
        **settings
):
    """
    Scatters through the medium to reach the environment
    output: E(w) * W
    """
    if isinstance(scattering_albedo, float):
        scattering_albedo = (scattering_albedo, scattering_albedo, scattering_albedo)
    if isinstance(scattering_albedo, tuple):
        scattering_albedo = torch.tensor([*scattering_albedo], device=rdv.device())

    def rendering_process(penvironment):
        ds = rdv.DependencySet()
        ds.add_parameters(
            camera_poses=camera_poses,
            sigma_tensor=grid,
            majorant_tensor=torch.tensor([majorant, 100000], device=rdv.device()),
            scattering_albedo_tensor=scattering_albedo,
            environment_tensor=penvironment,
            phase_g_tensor=torch.ones([2, 2, 2, 1], device=rdv.device()) * phase_g,
        )
        ds.requires(rdv.medium_phase_HG)
        ds.requires(rdv.medium_phase_sampler_HG)
        ds.requires(rdv.medium_scattering_albedo)
        ds.requires(rdv.medium_sigma)
        ds.requires(rdv.medium_boundary)
        ds.requires(rdv.medium_majorant)
        ds.requires(rdv.medium_environment)
        ds.requires(rdv.camera_sensors,
                    width=settings.get('width', 512),
                    height=settings.get('height', 512),
                    jittered=settings.get('jittered', True))
        map = rdv.DeltatrackingScatteringSampler(
            ds.sigma,
            ds.scattering_albedo,
            ds.phase_sampler,
            ds.environment,
            ds.boundary, ds.majorant)

        return ds, ds.camera.capture(map, fw_samples=settings.get('samples', 1), bw_samples=settings.get('bw_samples', 1)) # R = Wout * E(wout)
    return DifferentiableRendering.apply(rendering_process, environment)


def save_video(frames: torch.Tensor, filename: str, fps: int = 20, apply_gamma = True):
    if apply_gamma:
        frames = torch.clamp_min(frames, 0.0) ** (1.0 / 2.2)
    frames = torch.clamp(frames, 0.0, 1.0)
    # rgba = torch.zeros(*frames.shape[:-1], 4)
    # rgba[..., 0:3] = frames
    # rgba[..., 3] = 1.0
    kwargs = {}
    if filename.endswith('.webp'):
        kwargs.update(quality=100, loop=0)
    rdv.save_video(frames, filename, fps, **kwargs)


def accumulate(p, times):
    assert times > 0
    with torch.no_grad():
        img = p()
        for i in range(1, times):
            torch.add(img, p(), alpha=1, out=img)
        return img/times