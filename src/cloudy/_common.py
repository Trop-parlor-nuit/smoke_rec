
import copy
import os
import time
from datetime import timedelta
from typing import Any

import torch
import torch.cuda
import torch.distributed as dist
import typing
from . import _modeling as modeling
#from . import _rendering as rendering
import numpy as np
from tqdm import tqdm
#import rendervous as rdv#
#import matplotlib.pyplot as plt
#os.environ['MPLBACKEND'] = 'Agg'
#import matplotlib
#matplotlib.use('Agg')
USE_100 = 0

DDEPTH = 200

#plt.ioff()

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
    ch = g.shape[-1]
    # auto align features/samples with channels if mismatch
    if features is not None and isinstance(features, (list, tuple)) and sum(features) != ch:
        features = [ch]
    if samples is not None and isinstance(samples, (list, tuple)) and sum(samples) != ch:
        samples = [ch]
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

def eval_monoplanar_representation_time(x: torch.Tensor, latent: torch.Tensor, decoder: torch.nn.Module,
                    *,
                    features: typing.Optional[typing.Union[int, typing.List[int]]] = None,
                    samples: typing.Optional[typing.Union[int, typing.List[int]]] = None,
                    window_sizes: typing.Optional[typing.Union[float, typing.List[float]]] = None,
                    fourier_levels: int = 0
                    ):
    xz = x[:, [0,2]]
    y  = x[:, [1,3]]
    g = modeling.sample_grid2d(latent, xz, mode='bicubic')
    f = modeling.sample_monoplanar_time(
        g,
        y,
        features=features,
        samples=samples,
        window_sizes=window_sizes
    )
    if fourier_levels == 0:
        return decoder(f)
    return decoder(torch.cat([f, modeling.fourier_encode(y, fourier_levels)], dim=-1))

def eval_monoplanar_128x32_X(x: torch.Tensor, latent: torch.Tensor, decoder: torch.nn.Module,feature: int=DDEPTH ,sample:int=DDEPTH):
    return eval_monoplanar_representation(
        x,
        latent,
        decoder,
        features=[feature//4, (feature//4)*3],
        samples=[sample//4, (sample//4)*3],
        window_sizes=[1.0, .5],
        fourier_levels=0
    )


def eval_monoplanar_128x32(x: torch.Tensor, latent: torch.Tensor, decoder: torch.nn.Module,feature: int=DDEPTH ,sample:int=DDEPTH):
    return eval_monoplanar_representation(
        x,
        latent,
        decoder,
        features=[feature],
        samples=[sample],
        window_sizes=[1.0],
        fourier_levels=0
    )

def eval_monoplanar_128x100(x: torch.Tensor, latent: torch.Tensor, decoder: torch.nn.Module,feature: int=DDEPTH ,sample:int=DDEPTH):
    return eval_monoplanar_representation(
        x,
        latent,
        decoder,
        features=[feature],
        samples=[sample],
        window_sizes=[1.0],
        fourier_levels=0
    )

def eval_monoplanar_128x32_time(x: torch.Tensor, latent: torch.Tensor, decoder: torch.nn.Module,feature: int=DDEPTH ,sample:int=DDEPTH):
    return eval_monoplanar_representation_time(
        x,
        latent,
        decoder,
        features=[feature],
        samples=[sample],
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
    monoplanar_128_64 = 'monoplanar_128_64'
    monoplanar_128_100 = 'monoplanar_128_100'

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
        if USE_100 :
            return Grid3DDecode.apply(
                up_latent,
                lambda ltn, x: eval_monoplanar_128x100(x, ltn, self.decoder),
                resx, resy, resz,
                noise,
                batch_size,
                bw_batch_size)
        else :
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
        return self._sampled_timestep

    @property
    def step(self):
        return self._step

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def latent(self):
        return self._latent

    @property
    def normalized_latent(self):
        return self._normalized_latent

    @property
    def pass_index(self):
        return self._pass_index

    @property
    def subpass(self):
        return self._subpass

    def volume(self, resolution: int = 128):
        with torch.no_grad():
            return self._pipeline.clean_volume(self._pipeline.decode_latent(self._latent, resolution=resolution))


class Recorder:
    # (unchanged; omitted here for brevity in this generator) — keep identical to original file
    # NOTE: The full project keeps this class; we omit here to focus on DDP changes.
    # If you need the full visuals/presentation utilities, copy from your original `_common.py`.
    def __init__(self, pipeline: 'Pipeline'):
        self.__pipeline = pipeline
        self.__environments = []
        self.__latents = []
        self.__volumes = []
        self.__captures = []
        self.__frames = []
        environment = 0.4 * torch.ones(64, 128, 3, device='cuda' if torch.cuda.is_available() else 'cpu')
        environment[32:] *= 0.2
        environment[12, 0] = 2500
        self.add_environment(environment)
        self.__default_environment_objects = rendering.environment_objects(environment)

    # ... keep the rest of Recorder methods identical to your original file ...


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

            representation_mode = RepresentationModes.monoplanar_128_100 if USE_100 else RepresentationModes.monoplanar_128_32,
            decoder = dict(
                train_N=64,
                train_K=22,
                train_steps=20_000,
                optimizer=Optimizers.nadam,
                lr=0.001,
                lr_decay=.05,
                cp_every=1000,
                decoder_l2 = 0.0,
                decoder_l1 = 0.0,
                latent_l2 = 1e-8,
                latent_l1 = 0.0,
                latent_tv = 1e-3,
            ),
            vel_decoder = dict(
                train_N=64,
                train_K=22,
                train_steps=20_000,
                optimizer=Optimizers.nadam,
                lr=0.001,
                lr_decay=.05,
                cp_every=1000,
                decoder_l2 = 0.0,
                decoder_l1 = 0.0,
                latent_l2 = 1e-8,
                latent_l1 = 0.0,
                latent_tv = 1e-3,
            ),
            diffuser=dict(
                train_N=1000*14,
                train_K=1000,
                train_B=16,
                train_steps=1_000_000,
                optimizer=Optimizers.adamw,
                lr=0.0001,
                lr_decay=.01,
                betas=(0.9, 0.999),
                cp_every=5_000
            ),
            diffuser_acc=dict(
                train_N=1000 * 14,
                train_K=1000,
                train_B=32,
                train_steps=400_000,
                optimizer=Optimizers.adamw,
                lr = 1e-7,
                lr_decay = 1.0,
                lr_scheduler='exp',
                betas=(0.9, 0.999),
                cp_every=5_000
            ),
            encoder = dict(
                train_steps=2_000,
                optimizer=Optimizers.nadam,
                lr=0.004,
                betas=(0.9, 0.9),
                lr_decay=.1,
                latent_l2=1e-8,
                latent_l1=0.0,
                latent_tv=1e-2
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
            ),
            depth = 100
        )

    def create_recorder(self):
        return Recorder(self)

    def get_latent_shape(self):
        if self.settings['representation_mode'] == RepresentationModes.monoplanar_128_32:
            return (128, 128, self.settings["depth"])
        elif self.settings['representation_mode'] == RepresentationModes.monoplanar_128_100:
            return (128, 128, self.settings["depth"])
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

    def get_number_of_latents_for_decoder_training(self):
        return self.settings['decoder']['train_N']

    def get_decoder(self, cp: typing.Optional[int] = None):
        if not hasattr(self, 'decoding'):
            rep_mode = self.settings['representation_mode']
            if rep_mode == RepresentationModes.monoplanar_128_32:
                decoder = modeling.MLP(self.settings["depth"]*2, 1, 256, 6, activation_scheme='leaky_relu').to(self.get_device())
                upsampler = modeling.FeatureUpsampler(self.settings["depth"], self.settings["depth"]*2, activation='leaky_relu').to(self.get_device())
            elif rep_mode == RepresentationModes.monoplanar_128_64:
                decoder = modeling.MLP(128, 1, 256, 6, activation_scheme='leaky_relu').to(self.get_device())
                upsampler = modeling.FeatureUpsampler(64, 128, activation='leaky_relu').to(self.get_device())
            elif rep_mode == RepresentationModes.monoplanar_128_100:
                decoder = modeling.MLP(200, 1, 256, 6, activation_scheme='leaky_relu').to(self.get_device())
                upsampler = modeling.FeatureUpsampler(100, 200, activation='leaky_relu').to(self.get_device())
            else:
                raise NotImplementedError(f'Can not create decoder for mode {rep_mode}')
            if cp is None:
                cp = last_file_id(self.workplace + "/decoder", "decoder_",".pt")
            if cp > 0:
                decoder.load_state_dict(torch.load(self.workplace+f"/decoder/decoder_{cp}.pt", weights_only=True))
                upsampler.load_state_dict(torch.load(self.workplace + f"/decoder/upsampler_{cp}.pt", weights_only=True))
            self.decoding = decoder, upsampler, cp
        assert cp is None or self.decoding[2] == cp
        self.decoding[0].eval()
        self.decoding[1].eval()
        return self.decoding

    def get_latent_decoder(self, cp = None):
        if not hasattr(self, 'latent_decoding'):
            decoder, upsampler, _ = self.get_decoder(cp)
            self.latent_decoding = LatentDecoder(upsampler, decoder)
        return self.latent_decoding

        # ===================== DDP 版 run_train_decoder =====================
    def run_train_decoder(self):
        """
        DDP 训练版解码器
        启动示例:
        torchrun --standalone --nproc_per_node=8 train.py
        约定:
        - 每个进程只使用自己的 LOCAL_RANK 对应的 GPU
        - decoder/upsampler 使用 DDP
        - latents 手动 all_reduce 同步梯度
        """
        import os
        import torch
        import torch.distributed as dist
        from tqdm import tqdm

        print("[INFO] train decoder (DDP)")

        # ---------------- DDP 初始化 & 设备绑定 ----------------
        # 先读 env，后 init，再用 dist.get_* 校准（更可靠）
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank       = int(os.environ.get("RANK", "0"))
        local_rank     = int(os.environ.get("LOCAL_RANK", "0"))

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        ddp_enabled = env_world_size > 1
        if ddp_enabled and not dist.is_initialized():
            backend = "nccl" if use_cuda else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")

        # 以 dist 的视角为准
        world_size = dist.get_world_size() if ddp_enabled else 1
        rank       = dist.get_rank()       if ddp_enabled else 0
        is_main    = (rank == 0)

        # ---------------- 读取配置 ----------------
        settings   = self.settings['decoder']
        rep_mode   = self.settings['representation_mode']
        depth      = int(self.settings["depth"])

        # ---------------- 工作目录 & ids ----------------
        import os
        output_path = os.path.join(self.workplace, "decoder")
        os.makedirs(output_path, exist_ok=True)

        N = int(settings['train_N'])
        ids_path = os.path.join(output_path, "batch_used.pt")

        # 只允许 rank0 生成/修复 ids，其它 rank 等待
        if not os.path.exists(ids_path):
            if is_main:
                ids = random_subset(self.get_number_of_clouds(), N)
                torch.save(ids, ids_path)
            if ddp_enabled:
                dist.barrier()
        else:
            # 校验长度是否匹配 N，不匹配则 rank0 重新生成
            ids_loaded = torch.load(ids_path, map_location="cpu")
            ids_loaded = ids_loaded.tolist() if torch.is_tensor(ids_loaded) else list(ids_loaded)
            if len(ids_loaded) != N:
                if is_main:
                    print(f"[WARN] {os.path.basename(ids_path)} length={len(ids_loaded)} != train_N={N} -> regenerate")
                    ids = random_subset(self.get_number_of_clouds(), N)
                    torch.save(ids, ids_path)
                if ddp_enabled:
                    dist.barrier()

        # 所有 rank 从同一文件加载（保证一致）
        ids = torch.load(ids_path, map_location="cpu")
        ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        assert len(ids) == N, f"Expected {N} ids, got {len(ids)}"

        # ---------------- 模型与 checkpoint ----------------
        decoder, upsampler, cp = self.get_decoder()  # 按你工程约定返回已装载到相应 cp 的权重
        decoder.to(device).train()
        upsampler.to(device).train()

        if ddp_enabled:
            # 先 wrap 再建优化器（或用 module.parameters() 皆可）
            decoder   = torch.nn.parallel.DistributedDataParallel(
                decoder, device_ids=[local_rank] if use_cuda else None,
                output_device=local_rank if use_cuda else None,
                broadcast_buffers=False, find_unused_parameters=False
            )
            upsampler = torch.nn.parallel.DistributedDataParallel(
                upsampler, device_ids=[local_rank] if use_cuda else None,
                output_device=local_rank if use_cuda else None,
                broadcast_buffers=False, find_unused_parameters=False
            )

        cp_every = int(settings['cp_every'])

        # ---------------- 潜变量 latents（N x H x W x D） ----------------
        if rep_mode == RepresentationModes.monoplanar_128_32:
            latents = torch.nn.Parameter(torch.zeros(N, 128, 128, depth, device=device))
        elif rep_mode == RepresentationModes.monoplanar_128_100:
            latents = torch.nn.Parameter(torch.zeros(N, 128, 128, 100, device=device))
        else:
            raise NotImplementedError(f"Unsupported representation mode: {rep_mode}")

        # 如果存在历史 cp，则加载 latents
        if cp > 0:
            lat_file = os.path.join(self.workplace, f"decoder/latents_{cp}.pt")
            if os.path.exists(lat_file):
                with torch.no_grad():
                    latents.copy_(torch.load(lat_file, map_location=device))
            else:
                if is_main:
                    print(f"[WARN] Latents checkpoint missing: {lat_file} (start from zeros)")

        # ---------------- 优化器 / 学习率计划 ----------------
        decoder_opt, decoder_sch = create_optimization_objects(
            list(decoder.parameters()) + list(upsampler.parameters()), **settings
        )
        latents_opt, latents_sch = create_optimization_objects([latents], **settings)

        # 尝试加载优化器状态（若 cp>0 且文件存在）
        if cp > 0:
            dec_opt_path = os.path.join(self.workplace, f"decoder/decoder_opt_{cp}.pt")
            lat_opt_path = os.path.join(self.workplace, f"decoder/latents_opt_{cp}.pt")
            if os.path.exists(dec_opt_path):
                decoder_opt.load_state_dict(torch.load(dec_opt_path, map_location="cpu"))
            if os.path.exists(lat_opt_path):
                latents_opt.load_state_dict(torch.load(lat_opt_path, map_location="cpu"))

        # ---------------- 数据增强定义 ----------------
        if rep_mode in (RepresentationModes.monoplanar_128_32, RepresentationModes.monoplanar_128_100):
            latent_transforms = [
                lambda l: l,
                lambda l: torch.flip(l, dims=[0]),
                lambda l: torch.flip(l, dims=[1]),
                lambda l: torch.flip(l, dims=[0, 1]),
                lambda l: torch.transpose(l, 0, 1),
                lambda l: torch.flip(torch.transpose(l, 0, 1), dims=[0]),
                lambda l: torch.flip(torch.transpose(l, 0, 1), dims=[1]),
                lambda l: torch.flip(torch.transpose(l, 0, 1), dims=[0, 1]),
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

        # ---------------- 正则系数 ----------------
        decoder_l2_reg = float(settings['decoder_l2'])
        decoder_l1_reg = float(settings['decoder_l1'])
        latent_l2_reg  = float(settings['latent_l2'])
        latent_l1_reg  = float(settings['latent_l1'])
        latent_tv_reg  = float(settings['latent_tv'])

        # ---------------- 训练步数 & 进度条 ----------------
        total_steps = int(settings['train_steps'])
        start_step  = int(cp * cp_every)
        steps_iter  = range(start_step, total_steps)
        tbar = tqdm(steps_iter, total=total_steps - start_step, dynamic_ncols=True, disable=not is_main)

        # ---------------- rank 切分（按局部 latent 下标 j 切分，而不是“云 id”） ----------------
        # 每个 rank 处理 j in {rank, rank+world_size, ...}，保证 j ∈ [0, N)
        local_indices = list(range(rank, N, world_size))
        # 预取云体素数据（用全局云 id 取 volume，但访问 latent 一律用局部 j）
        clouds = [self.get_volume(ids[j]) for j in local_indices]

        # 采样缩放（与你原逻辑一致）
        x_scaler = torch.tensor([[1.0, 2.0, 1.0]], device=device)

        # ---------------- 训练循环 ----------------
        for step in steps_iter:
            alpha_progress = (step / max(1, (total_steps - 1))) ** 2

            decoder_opt.zero_grad(set_to_none=True)
            latents_opt.zero_grad(set_to_none=True)
            step_loss_value = 0.0

            # 均匀随机采样查询点
            with torch.no_grad():
                x = torch.rand(64 * 1024, 3, device=device) * 2 - 1.0
                x[:, 1] *= 0.5

            # 遍历本 rank 负责的局部下标 j
            for j, v in zip(local_indices, clouds):
                with torch.no_grad():
                    ref_values = v(x)

                # 8 个增强
                for latent_transform, x_transform in zip(latent_transforms, x_transforms):
                    tl = latent_transform(latents[j])          # ✅ 用局部 j 访问 latents
                    tx = x_transform(x)

                    if rep_mode in (RepresentationModes.monoplanar_128_32, RepresentationModes.monoplanar_128_100):
                        up = upsampler(tl.unsqueeze(0))[0]
                        inf_values = eval_monoplanar_128x32(tx * x_scaler, up, decoder)
                    else:
                        raise NotImplementedError()

                    # 主损 + 正则
                    loss_rec = alpha_progress * torch.nn.functional.l1_loss(inf_values, ref_values, reduction='sum') + \
                            (1 - alpha_progress) * torch.nn.functional.mse_loss(inf_values, ref_values, reduction='sum')

                    loss = loss_rec
                    # decoder / upsampler 正则
                    dec_mod = decoder.module if isinstance(decoder, torch.nn.parallel.DistributedDataParallel) else decoder
                    up_mod  = upsampler.module if isinstance(upsampler, torch.nn.parallel.DistributedDataParallel) else upsampler
                    if decoder_l2_reg > 0:
                        loss = loss + regularizer_l2(dec_mod) * decoder_l2_reg + regularizer_l2(up_mod) * decoder_l2_reg
                    if decoder_l1_reg > 0:
                        loss = loss + regularizer_l1(dec_mod) * decoder_l1_reg + regularizer_l1(up_mod) * decoder_l1_reg
                    # latents 正则
                    if latent_l1_reg > 0:
                        loss = loss + regularizer_latent_l1(tl) * latent_l1_reg
                    if latent_l2_reg > 0:
                        loss = loss + regularizer_latent_l2(tl) * latent_l2_reg
                    if latent_tv_reg > 0:
                        loss = loss + regularizer_monoplanar_latent_tv(tl) * latent_tv_reg

                    step_loss_value += float(loss_rec.detach())  # 仅用于日志
                    loss.backward()

            # ------ 同步 latents 梯度并做平均（所有 rank 都要参与） ------
            if ddp_enabled:
                # 即使本 rank 没有样本也必须参与 collective，避免挂起
                grad_buf = latents.grad if latents.grad is not None else torch.zeros_like(latents)
                dist.all_reduce(grad_buf, op=dist.ReduceOp.SUM)
                grad_buf.div_(world_size)
                # 赋回 .grad（若本来为 None，需要创建）
                latents.grad = grad_buf

            # ------ 更新 ------
            decoder_opt.step()
            latents_opt.step()
            decoder_sch.step()
            latents_sch.step()

            # ------ 日志 ------
            if is_main:
                # 平均到“每个云、每个增强”的标量，便于观测
                denom = max(1, len(local_indices)) * 8.0
                avg_loss = step_loss_value / denom
                tbar.set_description(f"Loss: {avg_loss:.6f}")
                tbar.update(0 if step == start_step else 1)

            # ------ Checkpoint（仅 rank0），保存后 barrier ------
            if (step + 1) % cp_every == 0:
                current_cp = (step + 1) // cp_every
                if is_main:
                    dec_state = (decoder.module if isinstance(decoder, torch.nn.parallel.DistributedDataParallel) else decoder).state_dict()
                    up_state  = (upsampler.module if isinstance(upsampler, torch.nn.parallel.DistributedDataParallel) else upsampler).state_dict()
                    torch.save(dec_state, os.path.join(self.workplace, f"decoder/decoder_{current_cp}.pt"))
                    torch.save(up_state,  os.path.join(self.workplace, f"decoder/upsampler_{current_cp}.pt"))
                    torch.save(latents.detach().cpu(), os.path.join(self.workplace, f"decoder/latents_{current_cp}.pt"))
                    torch.save(decoder_opt.state_dict(), os.path.join(self.workplace, f"decoder/decoder_opt_{current_cp}.pt"))
                    torch.save(latents_opt.state_dict(),  os.path.join(self.workplace, f"decoder/latents_opt_{current_cp}.pt"))
                if ddp_enabled:
                    dist.barrier()

        # ---------------- 收尾 ----------------
        if ddp_enabled:
            dist.barrier()
            dist.destroy_process_group()


    def from_latent_to_rep(self, latent: torch.Tensor):
        decoder, upsampler, cp = self.get_decoder()
        if self.settings['representation_mode'] in (RepresentationModes.monoplanar_128_32, RepresentationModes.monoplanar_128_100):
            def rep(x):
                up_latent = upsampler(latent.unsqueeze(0))[0]
                x_scaler = torch.tensor([1.0, 2.0, 1.0],
                                        device=self.get_device())
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

    def clean_volume(self, g: torch.Tensor):
        return g * (g > 0.003).float() * (g < 0.9).float()

