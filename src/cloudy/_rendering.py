import rendervous as rdv
import torch
import typing


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
        return ds.camera.capture(
            ds.transmittance,
            fw_samples=settings.get('samples', 1),
            batch_size=512*512
        )

    # return DifferentiableRendering.apply(rendering_process, grid)
    return rdv.parameterized_call(rendering_process, grid)


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


def save_video(frames: torch.Tensor, filename: str, fps: int = 20, apply_gamma = True):
    if apply_gamma:
        frames = torch.clamp_min(frames, 0.0) ** (1.0 / 2.2)
    frames = torch.clamp(frames, 0.0, 1.0)
    # rgba = torch.zeros(*frames.shape[:-1], 4)
    # rgba[..., 0:3] = frames
    # rgba[..., 3] = 1.0
    kwargs = {}
    if filename.endswith('.webp'):
        kwargs.update(quality=100)
    rdv.save_video(frames, filename, fps, **kwargs)


def accumulate(p, times):
    assert times > 0
    with torch.no_grad():
        img = p()
        for i in range(1, times):
            torch.add(img, p(), alpha=1, out=img)
        return img/times