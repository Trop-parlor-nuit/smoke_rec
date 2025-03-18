import typing
from typing import Any
import torch
import numpy as np
import denoising_diffusion_pytorch
from tqdm import tqdm


def sample_grid2d(grid: torch.Tensor, x: torch.Tensor, *, mode: str = 'bilinear', align_corners: bool = True, padding_mode: str = 'border'):
    """
    Samples values at positions of a 2D regular grid
    input: grid (HxWxC), x (B..., 2)
    output: (B..., C)
    """
    linearize_x = x.view(1, -1, 1, x.shape[-1])
    return torch.nn.functional.grid_sample(
        grid.unsqueeze(0).permute(0, 3, 1, 2),
        linearize_x,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    ).permute(0, 2, 3, 1).reshape(*x.shape[:-1], grid.shape[-1])


def sample_grid2d_batch(grids: torch.Tensor, x: torch.Tensor, *, mode: str = 'bilinear', align_corners: bool = True, padding_mode: str = 'border'):
    """
    Samples values at positions of a 2D regular grids
    input: grids (G, H, W, C), x (B..., 2)
    output: (G, B..., C)
    """
    linearize_x = x.view(1, -1, 1, x.shape[-1])
    return torch.nn.functional.grid_sample(
        grids.permute(0, 3, 1, 2),
        linearize_x,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    ).permute(0, 2, 3, 1).reshape(grids.shape[0], *x.shape[:-1], grids.shape[-1])


def sample_window_features(grid: torch.Tensor, x: torch.Tensor, width: float, stride: int):
    """
    grid: (B, F)
    x: (B, 1)
    output (B, stride)
    """
    features = grid.shape[-1]
    cells = features - 1
    e = width * torch.arange(-1., 1.0001, 2./(stride-1), device=x.device).view(*([1]*(len(x.shape)-1)), stride)  # (1..., features)
    s = (x+e)*0.5 + 0.5
    cell = torch.clamp(s, 0.0, 0.9999)*cells
    icell = cell.long()
    # mask_a = (icell >= 1)
    # mask_b = (icell < features)
    offsets = torch.arange(0, x.numel(), 1, dtype=torch.long, device=x.device).view(*x.shape)
    alpha = cell - icell.float()
    va = torch.take(grid, icell + offsets * features) # * torch.exp(-0.5 * (e / 0.25) ** 2) / 0.25 / 2.5
    vb = torch.take(grid, icell + 1 + offsets * features) # * torch.exp(-0.5 * (e / 0.25) ** 2) / 0.25 / 2.5
    return torch.lerp(va, vb, alpha) # * mask


def sample_localized_features(grid: torch.Tensor, x: torch.Tensor):
    """
    grid: (B, F)
    x: (B, 1)
    """
    features = grid.shape[-1]
    bins = (features - 1)
    e = torch.arange(-1., 1.0001, 2./bins, device=x.device).view(*([1]*(len(x.shape)-1)), features)  # (1..., features)
    s = (x+e)*0.5 + 0.5
    mask = (s >= 0.0) * (s <= 1.000001)
    cell = torch.clamp(s, 0.0, 0.9999)*bins
    icell = cell.long()
    offsets = torch.arange(0, x.numel(), 1, dtype=torch.long, device=x.device).view(*x.shape)
    alpha = cell - icell.float()
    va = torch.take(grid, icell + offsets * features) # * torch.exp(-0.5 * (e / 0.25) ** 2) / 0.25 / 2.5
    vb = torch.take(grid, icell + 1 + offsets * features) # * torch.exp(-0.5 * (e / 0.25) ** 2) / 0.25 / 2.5
    return torch.lerp(va, vb, alpha) # * mask


def sample_grid3d(grid: torch.Tensor, x: torch.Tensor, *, mode: str = 'bilinear', align_corners: bool = True, padding_mode: str = 'border'):
    """
    Samples values at positions of a 3D regular grid
    input: grid (DxHxWxC), x (B..., 3)
    output: (B..., C)
    """
    linearize_x = x.view(1, -1, 1, 1, x.shape[-1])
    return torch.nn.functional.grid_sample(
        grid.unsqueeze(0).permute(0, 4, 1, 2, 3),
        linearize_x,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    ).permute(0, 2, 3, 4, 1).reshape(*x.shape[:-1], grid.shape[-1])


def sample_grid2d_corners(grid: torch.Tensor, x: torch.Tensor):
    """
    Samples values at corners and relative distances of the cell containing a specific position.
    input: grid (HxWxC), x (B..., 2)
    output: features (B..., 4, C), dx (B..., 4, 2), alphas (B..., 4, 1)
    """
    H, W = grid.shape[:2]
    nx = torch.clamp(x * 0.5 + 0.5, 0.0, 0.999999)
    p = torch.cat([nx[..., [0]] * (W - 1), nx[..., [1]]*(H - 1)], dim=-1)
    p_batch_shape = p.shape[:-1]
    offsets = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long, device=p.device)
    offsets = offsets.view(*([1] * len(p_batch_shape)), 4, 2)
    p = p.view(*p_batch_shape, 1, 2)
    p_corner = p.long()
    c = p_corner + offsets  # B..., 4, 2
    f = grid[c[..., 1], c[..., 0], :]
    dx = p - c
    alphas = (1 - dx.abs()).prod(dim=-1, keepdim=True)
    return f, dx, alphas


def sample_grid3d_corners(grid: torch.Tensor, x: torch.Tensor):
    """
    Samples values at corners and relative distances of the cell containing a specific position.
    input: grid (DxHxWxC), x (B..., 3)
    output: features (B..., 8, C), dx (B..., 8, 3), alphas (B..., 8, 1)
    """
    D, H, W = grid.shape[:3]
    nx = torch.clamp(x * 0.5 + 0.5, 0.0, 0.999999)
    p = torch.cat([nx[..., [0]] * (W - 1), nx[..., [1]]*(H - 1), nx[..., [2]]*(D - 1)], dim=-1)
    p_batch_shape = p.shape[:-1]
    offsets = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.long, device=p.device)
    offsets = offsets.view(*([1] * len(p_batch_shape)), 8, 3)
    p = p.view(*p_batch_shape, 1, 3)
    p_corner = p.long()
    c = p_corner + offsets  # B..., 8, 3
    f = grid[c[..., 2], c[..., 1], c[..., 0], :]
    dx = p - c
    alphas = (1 - dx.abs()).prod(dim=-1, keepdim=True)
    return f, dx, alphas


def compute_kplanar_dimensions(dimension: int, reduce: str = 'cat'):
    """
    Given the space dimension and the reduction, returns the required number of planes and
    the multiplier for the features
    """
    planes = dimension * (dimension - 1) // 2
    features_multiplier = 1 if reduce != 'cat' and reduce != 'concat' else planes
    return planes, features_multiplier


def sample_kplanar(grid: torch.Tensor, x: torch.Tensor, *, reduce: str = 'cat'):
    """
    Samples the planes with every projection and reduce
    input: grid (K, R, R, C), x (B..., D)
    output: (B..., C | C * K)
    """
    planes = grid.shape[0]
    dimension = 2
    while True:
        num_planes = dimension * (dimension - 1) // 2
        if planes == num_planes:
            break
        if planes < num_planes:
            raise Exception('Wrong number of planes')
        dimension += 1
    feats = []
    for g, pair in zip(grid, [[i, j] for i in range(dimension - 1) for j in range(i + 1, dimension)]):
        feats.append(sample_grid2d(g, x[..., pair]))
    feats = torch.stack(feats, dim=-2)
    if reduce == 'add' or reduce == 'sum':
        feats = feats.sum(dim=-2)
    elif reduce == 'mul' or reduce == 'prod':
        feats = (feats + 1).prod(dim=-2)
    elif reduce == 'cat' or reduce == 'concat':
        feats = feats.flatten(start_dim=-2)
    else:
        raise NotImplemented()
    return feats


def sample_kplanar_disentangled(grid: torch.Tensor, x: torch.Tensor, *, reduce: str = 'cat'):
    pass


def sample_monoplanar(
        grid: torch.Tensor,
        x: torch.Tensor,
        features: typing.Optional[typing.Union[int, typing.List[int]]] = None,
        samples: typing.Optional[typing.Union[int, typing.List[int]]] = None,
        window_sizes: typing.Optional[typing.Union[float, typing.List[float]]] = None
):
    if features is None:
        features = grid.shape[-1]
    if samples is None:
        samples = grid.shape[-1]
    if window_sizes is None:
        window_sizes = 1.0
    if isinstance(features, int):
        features = [features]
    if isinstance(samples, int):
        samples = [samples]
    if isinstance(window_sizes, float):
        window_sizes = [window_sizes]
    levels = len(features)
    assert len(samples) == levels or len(samples) == 1
    assert len(window_sizes) == levels or len(window_sizes) == 1
    assert sum(features) == grid.shape[-1]
    if len(samples) != levels:
        samples = samples * levels
    if len(window_sizes) != levels:
        window_sizes = window_sizes * levels
    output = []
    offset = 0
    for f, s, w in zip(features, samples, window_sizes):
        g = grid[..., offset: offset + f]
        output.append(sample_window_features(g, x, w, s))
        offset += f
    return torch.cat(output, dim=-1)


def fourier_encode(x: torch.Tensor, levels: int):
    x_times_pi = x * np.pi
    return torch.cat(
        [torch.sin(x_times_pi * (1 << l)) / ((l + 1)**2) for l in range(levels)] +
        [torch.cos(x_times_pi * (1 << l)) / ((l + 1)**2) for l in range(levels)], dim=-1
    )


class Volume(torch.nn.Module):
    """
    Represents a volume through a regular grid between bmin and bmax.
    The boundary is obtained when scaled the grid to fit within -fit_box...fit_box
    """
    def __init__(self,
                 grid: torch.Tensor,
                 trainable: bool = False,
                 aligned: bool = True,
                 padding: str = 'zeros',
                 fit_box = (1.0, 1.0, 1.0),
                 ):
        super().__init__()
        self.grid = torch.nn.Parameter(grid, requires_grad=trainable)
        scale = min(b / d for d, b in zip(grid.shape[:-1], fit_box))
        # max_dim = max(grid.shape[:-1])
        dim = [d * scale for d in grid.shape[:-1]]
        dim.reverse()
        self.bmin = torch.nn.Parameter(-torch.tensor(dim, device=grid.device), requires_grad=False)
        self.bmax = torch.nn.Parameter(torch.tensor(dim, device=grid.device), requires_grad=False)
        self.is_aligned = aligned
        self.padding_mode = padding

    def forward(self, *args):
        x, = args
        x = 2.0*(x - self.bmin)/(self.bmax - self.bmin) - 1.0
        return sample_grid3d(self.grid, x, padding_mode=self.padding_mode, align_corners=self.is_aligned)


def sample_slice (vol, step_size: float, umin=-1.0, umax=1.0, vmin=-1.0, vmax=1.0, w=0.0, axis='z', device: typing.Optional[torch.device] = None):
    if device is None:
        assert isinstance(vol, torch.nn.Module)
        device = next(iter(vol.parameters())).device
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


def sample_slices (vol: torch.nn.Module, step_size: float, umin=-1.0, umax=1.0, vmin=-1.0, vmax=1.0, w=0.0, slices: int = 1, slice_delta = 0.1, axis='z'):
    device = next(iter(vol.parameters())).device
    u = torch.arange(umin, umax + 0.0000001, step_size, device=device)
    v = torch.arange(vmin, vmax + 0.0000001, step_size, device=device)
    w = torch.arange(w - (slices - 1) * slice_delta / 2, w + slices * slice_delta / 2, slice_delta, device=device)
    p = torch.cartesian_prod(w, v, u)
    if axis == 'z':
        p = p[:, [2, 1, 0]]
    elif axis == 'y':
        p = p[:, [2, 0, 1]]
    else:
        p = p[:, [0, 1, 2]]
    values = vol(p)
    return values.view(len(w), len(v), len(u), -1).mean(dim=0)


def reconstruct_grid3d(
        rep_function: typing.Callable[[torch.Tensor], torch.Tensor],
        xmin: float = -1.0, xmax: float = 1.0,
        ymin: float = -1.0, ymax: float = 1.0,
        zmin: float = -1.0, zmax: float = 1.0,
        resx : int = 256,
        resy : int = 128,
        resz : int = 256,
        batch_if_greater : int = 129*65*129,
        noise: float = 0.0,
        device: typing.Optional[torch.device] = None
):
    if device is None:
        assert isinstance(rep_function, torch.nn.Module), "Can only infer device if rep_function is a torch.nn.Module object"
        device = next(iter(rep_function.parameters())).device
    stepx = (xmax - xmin)/(resx - 1)
    stepy = (ymax - ymin)/(resy - 1)
    stepz = (zmax - zmin)/(resz - 1)
    x = torch.arange(xmin, xmax + 0.0000001, stepx, device=device)
    y = torch.arange(ymin, ymax + 0.0000001, stepy, device=device)
    z = torch.arange(zmin, zmax + 0.0000001, stepz, device=device)
    count_values = len(x) * len(y) * len(z)
    if count_values <= batch_if_greater:  # no batched solution
        p = torch.cartesian_prod(z, y, x)[:, [2, 1, 0]]
        if noise > 0:
            cell_size = torch.tensor([stepx, stepy, stepz], device=p.device)
            p += (torch.rand_like(p)-.5) * cell_size.view(1, 3) * noise
            # p += torch.randn_like(p) * cell_size.view(1, 3) * noise
        g = rep_function(p).view(len(z), len(y), len(x), -1)
        torch.cuda.empty_cache()
        return g
    else:
        x_count = len(x)
        y_count = len(y)
        z_count = len(z)
        num_batch_z = max(1, (x_count * y_count * z_count) // (1 * 1024 * 1024))
        grid_values = []
        for i, bz in enumerate(torch.chunk(z, num_batch_z)):
            chunk_size = len(bz)
            p = torch.cartesian_prod(x, y, bz)
            values = rep_function(p)
            grid_values.append(values.view(len(x), len(y), chunk_size, values.shape[-1]).permute(2, 1, 0, 3))
        return torch.cat(grid_values, dim=0)


class DiffEMA(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        current_img, prev_img, alpha = args
        ctx.save_for_backward(current_img)
        return current_img * alpha + prev_img.detach() * (1 - alpha)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0], None, None


def ema_diff(current, prev, alpha: float):
    return DiffEMA.apply(current, prev, alpha)


# Sine activation used for SIREN model
class Sine(torch.nn.Module):
    def __init__(self, sin_factor: float = 30):
        super(Sine, self).__init__()
        self.sin_factor = sin_factor

    def forward(self, *args):
        x, = args
        return torch.sin(x * self.sin_factor)


class DClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, min, max = args
        return torch.clamp(x, min, max)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0], None, None


def dclamp(t: torch.Tensor, min, max) -> torch.Tensor:
    # return torch.clamp(t, min, max)
    return DClamp.apply(t, min, max)


# Snake activation from https://github.com/EdwardDixon/snake/blob/master/snake
class SnakeFunction(torch.autograd.Function):
    """
    Autograd function implementing the serpentine-like sine-based periodic activation function.

    .. math::
         \text{Snake}_a := x + \frac{1}{a} \sin^2(ax)

    This function computes the forward and backward pass for the Snake activation, which helps in better
    extrapolating to unseen data, particularly when dealing with periodic functions.

    Attributes:
        ctx (torch.autograd.function._ContextMethodMixin): Context object used for saving and retrieving tensors.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Snake activation function.

        Args:
            x (torch.Tensor): Input tensor.
            a (torch.Tensor): Trainable parameter controlling the frequency of the sine function.

        Returns:
            torch.Tensor: Result of applying the Snake activation function to the input tensor.
        """
        ctx.save_for_backward(x, a)

        # Handle case where `a` is zero to avoid division by zero errors.
        return torch.where(a == 0, x, torch.addcdiv(x, torch.square(torch.sin(a * x)), a))

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Backward pass for the Snake activation function.

        Args:
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The gradients of the loss with respect to `x` and `a`.
        """
        x, a = ctx.saved_tensors

        # Calculate the gradient of the input `x`
        sin2ax = torch.sin(2 * a * x) if any(ctx.needs_input_grad) else None
        grad_x = grad_output * (1 + sin2ax) if ctx.needs_input_grad[0] else None

        # Calculate the gradient of the parameter `a`
        grad_a = (
            grad_output
            * torch.where(a == 0, torch.square(x), sin2ax * x / a - torch.square(torch.sin(a * x) / a))
            if ctx.needs_input_grad[1]
            else None
        )

        return grad_x, grad_a


class Snake(torch.nn.Module):
    """
    Implementation of the Snake activation function as a torch module.

    .. math::
         \text{Snake}_a := x + \frac{1}{a} \sin^2(ax) = x - \frac{1}{2a}\cos(2ax) + \frac{1}{2a}

    This activation function is designed to better extrapolate unseen data, particularly periodic functions.

    Parameters:
        in_features (int or list): The shape or number of input features.
        a (float, optional): Initial value of the trainable parameter `a`, controlling the sine frequency. Defaults to None.
        trainable (bool, optional): If `True`, the parameter `a` will be trainable. Defaults to True.

    Examples:
        >>> snake_layer = Snake(256)
        >>> x = torch.randn(256)
        >>> x = snake_layer(x)

    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(
        self,
        in_features: int | list[int],
        a: float | None = None,
        trainable: bool = True,
    ):
        """
        Initialize the Snake activation layer.

        Args:
            in_features (int or list): Shape of the input, either a single integer or a list of integers indicating feature dimensions.
            a (float, optional): Initial value for the parameter `a`, which controls the sine frequency. If not provided, `a` will be initialized to a random value from an exponential distribution.
            trainable (bool, optional): If `True`, the parameter `a` will be trained during backpropagation.
        """
        super(Snake, self).__init__()
        self.in_features = (
            in_features if isinstance(in_features, list) else [in_features]
        )

        # Ensure initial_a is a floating point tensor
        if isinstance(in_features, int):
            initial_a = torch.full((in_features,), a, dtype=torch.float32)  # Explicitly set dtype to float32
        else:
            initial_a = torch.full(in_features, a, dtype=torch.float32)  # Assuming in_features is a list/tuple of dimensions

        if trainable:
            self.a = torch.nn.Parameter(initial_a)
        else:
            self.register_buffer('a', initial_a)

        if trainable:
            self.a = torch.nn.Parameter(initial_a)
        else:
            self.register_buffer("a", initial_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Snake activation layer.

        Args:
            x (torch.Tensor): Input tensor to the layer.

        Returns:
            torch.Tensor: Result of applying the Snake activation function.
        """
        return SnakeFunction.apply(x, self.a)


class MLPInits:

    @staticmethod
    def relu(m: torch.nn.Linear):
        # torch.nn.init.kaiming_uniform_(m.weight, a=.0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


    @staticmethod
    def leaky_relu(m: torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='leaky_relu', mode='fan_in')


    @staticmethod
    def sigmoid(m: torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


    @staticmethod
    def tanh(m):
        torch.nn.init.xavier_normal_(m.weight)


    @staticmethod
    def selu(m):
        torch.nn.init.normal_(m.weight, std=1 / np.sqrt(m.weight.size(-1)))


    @staticmethod
    def elu(m):
        torch.nn.init.normal_(m.weight, std=1.2451983007 / np.sqrt(m.weight.size(-1)))


    @staticmethod
    def gelu(m):
        with torch.no_grad():
            m.weight /= torch.norm(m.weight, p=2, dim=0, keepdim=True)


    @staticmethod
    def siren(m, first_layer: bool = False, factor: float = 30):
        num_input = m.weight.size(-1)
        range_sd = 1 / num_input if first_layer else np.sqrt(6 / num_input) / factor
        torch.nn.init.uniform_(m.weight, -range_sd, range_sd)
        # if first_layer:
        #     with torch.no_grad():
        #         m.weight *= factor
        if m.bias is not None:
            # torch.nn.init.zeros_(m.bias)
            torch.nn.init.uniform_(m.bias, -range_sd, range_sd)


    @staticmethod
    def snake(m):
        torch.nn.init.kaiming_uniform_(m.weight)
        with torch.no_grad():
            m.weight /= np.sqrt(2)


# Multilayer perceptron base for different end-to-end models like SIREN, regressions, etc.
class MLP(torch.nn.Module):
    @staticmethod
    def append_layer(layers: list, activation_scheme: str, layer_index: int, input_dim: int, output_dim: int, total_layers: int, add_activation: bool = True):
        linear = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
        layers.append(linear)
        if activation_scheme == 'relu':
            MLPInits.relu(linear)
            if add_activation: layers.append(torch.nn.ReLU())
            return
        if activation_scheme == 'leaky_relu':
            MLPInits.leaky_relu(linear)
            if add_activation: layers.append(torch.nn.LeakyReLU(0.01))
            return
        if activation_scheme == 'gelu':
            MLPInits.gelu(linear)
            if add_activation: layers.append(torch.nn.GELU())
            return
        if activation_scheme == 'softplus':
            MLPInits.relu(linear)
            if add_activation: layers.append(torch.nn.Softplus())
            return
        if activation_scheme == 'elu':
            MLPInits.elu(linear)
            if add_activation: layers.append(torch.nn.ELU())
            return
        if activation_scheme == 'siren':
            MLPInits.siren(linear, first_layer=layer_index==0, factor=30)
            if add_activation: layers.append(Sine(30))
            return
        if activation_scheme == 'sin':
            if layer_index == 0:
                MLPInits.siren(linear, first_layer=layer_index == 0, factor=3.0)
                if add_activation: layers.append(Sine())
            else:
                MLPInits.relu(linear)
                if add_activation: layers.append(torch.nn.ReLU())
            return
        if activation_scheme == 'sigmoid':
            MLPInits.sigmoid(linear)
            if add_activation: layers.append(torch.nn.Sigmoid())
            return
        if activation_scheme == 'snake':
            MLPInits.snake(linear)
            if add_activation: layers.append(Snake(output_dim, 2.0, trainable=True))
            return
        raise NotImplemented()

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, layers: int, activation_scheme: str, activate_last_layer: bool = False):
        super().__init__()
        modules = []
        prev_dim = input_dim
        for l in range(layers):
            MLP.append_layer(modules, activation_scheme, l, prev_dim, hidden_dim if l < layers - 1 else output_dim, layers, activate_last_layer or l < layers-1)
            prev_dim = hidden_dim
        self.net = torch.nn.Sequential(*modules)

    def forward(self, *args):
        x, = args
        return self.net(x)




# METRICS

def psnr(grid1, grid2):
    if isinstance(grid1, torch.Tensor):
        return -10 * torch.log10(((grid1 - grid2)**2).mean()).item()
    return -10 * np.log10(((grid1 - grid2)**2).mean())


def rmse(grid1, grid2):
    return np.sqrt(((grid1 - grid2) ** 2).mean().item())


def mae(grid1, grid2):
    return ((grid1 - grid2).abs()).mean().item()


def nmae(grid1, grid2):
    return mae(grid1, grid2) / (grid2.abs().mean() + 0.00000001).item()


def nrmse(grid1, grid2):
    return rmse(grid1, grid2) / (grid2.abs().mean() + 0.00000001).item()


def get_all_measurements(current, target):
    return dict(
        psnr = psnr(current, target),
        rmse = rmse(current, target),
        mae = mae(current, target),
        nmae = nmae(current, target),
        nrmse = nrmse(current, target),
    )


def total_variation_2D(img: torch.Tensor) -> torch.Tensor:
    '''
    :param img: (B, H, W, C)
    :return: (1,)
    '''
    tv_h = (img[:, :-1, :, :] - img[:, 1:, :, :])**2
    tv_w = (img[:, :, :-1, :] - img[:, :, 1:, :])**2
    return tv_h.sum() + tv_w.sum()


def total_variation_2D_abs(img: torch.Tensor) -> torch.Tensor:
    '''
    :param img: (B, H, W, C)
    :return: (1,)
    '''
    tv_h = (img[:, :-1, :, :] - img[:, 1:, :, :]).abs()
    tv_w = (img[:, :, :-1, :] - img[:, :, 1:, :]).abs()
    return tv_h.sum() + tv_w.sum()


def total_variation_3D(img: torch.Tensor) -> torch.Tensor:
    '''
    :param img: (B, H, W, C)
    :return: (1,)
    '''
    tv_d = (img[:, :-1, :, :, :] - img[:, 1:, :, :, :])**2
    tv_h = (img[:, :, :-1, :, :] - img[:, :, 1:, :, :])**2
    tv_w = (img[:, :, :, :-1, :] - img[:, :, :, 1:, :])**2
    return tv_d.sum() + tv_h.sum() + tv_w.sum()


def upsampling(imgs, mode: str = 'bilinear'):
    return torch.nn.functional.interpolate(imgs.permute(0, 3, 1, 2), scale_factor=2.0, mode=mode, align_corners=True).permute(0, 2, 3, 1)


def downsampling(imgs, mode: str = 'bilinear'):
    return torch.nn.functional.interpolate(imgs.permute(0, 3, 1, 2), scale_factor=.5, mode=mode, align_corners=True).permute(0, 2, 3, 1)


def resampling(imgs, size, mode: str = 'bilinear', align_corners=True):
    return torch.nn.functional.interpolate(imgs.permute(0, 3, 1, 2), size=size, mode=mode, align_corners=align_corners).permute(0, 2, 3, 1)


def resample_grid(grids, size):
    return torch.nn.functional.interpolate(
        grids.permute(0, 4, 1, 2, 3), size=size, mode='trilinear', align_corners=True
    ).permute(0, 2, 3, 4, 1)


from torchvision.transforms.transforms import GaussianBlur
def gaussian_filter(imgs, sigma=1.0, channel_last: bool = True, kernel_size: int = 5):
    gaussian = GaussianBlur(kernel_size, sigma=sigma)
    if channel_last:
        return gaussian(imgs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return gaussian(imgs)


def to_channels_first(t: torch.Tensor):
    d = len(t.shape)
    return t.permute(0, d - 1, *range(1, d-1))


def to_channels_last(t: torch.Tensor):
    d = len(t.shape)
    return t.permute(0, *range(2, d), 1)


class UpsampleConv2D(torch.nn.Sequential):
    def __init__(self, in_features: int, out_features: int, batch_norm=False, activation: typing.Literal['none', 'relu', 'leaky_relu'] = 'none'):
        modules = [
            torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            torch.nn.Conv2d(in_features, out_features, 3, 1, 1, dilation=1, padding_mode='replicate')
        ]
        if batch_norm:
            modules.append(torch.nn.BatchNorm2d(out_features))
        if activation != 'none':
            if activation=='relu':
                a = torch.nn.ReLU()
            elif activation == 'leaky_relu':
                a = torch.nn.LeakyReLU(0.1)
            else:
                raise Exception()
            modules.append(a)
        super().__init__(*modules)


class FeatureUpsampler(torch.nn.Sequential):
    def __init__(self, in_features: int, out_features: int, activation: typing.Literal['none', 'relu', 'leaky_relu'] = 'none'):
        modules = [
            torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            torch.nn.Conv2d(in_features, out_features, 3, 1, 1, dilation=1, padding_mode='replicate')
        ]
        if activation != 'none':
            if activation=='relu':
                a = torch.nn.ReLU()
            elif activation == 'leaky_relu':
                a = torch.nn.LeakyReLU(0.1)
            else:
                raise Exception()
            modules.append(a)
        super().__init__(*modules)
        # self.net = torch.nn.Sequential(*modules)

    def forward(self, *args):
        x, = args
        x = to_channels_first(x)
        x = super().forward(x)
        x = to_channels_last(x)
        return x


class GaussianDiffuse2D(torch.nn.Module):
    def __init__(self, channels: int, resolution: int, timesteps: int = 100):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.timesteps = timesteps
        levels = int(np.log2(resolution)) - 1
        dim_mults = [1, 2, 4, 8]
        # dim_mults = [1 << l for l in range(min(4, levels - 1))]
        self.diffusion = denoising_diffusion_pytorch.GaussianDiffusion(
            denoising_diffusion_pytorch.Unet(
                dim=channels*2,
                channels=channels,
                dim_mults=dim_mults,
                flash_attn=True),
            image_size=resolution,
            timesteps=timesteps,
            auto_normalize=False,
            min_snr_loss_weight=True,
        )

    def forward(self, *args):
        x_0, = args
        return self.diffusion(x_0)

    def forward_loss(self, *args):
        return self(*args)

    def eval_generator(self, x_t, t: typing.Optional[int] = None):
        if t is None:
            t = self.timesteps
        for t in tqdm(reversed(range(t)), total=t, desc='Generating'):
            x_t, _ = self.diffusion.p_sample(x_t, t)
        return x_t

    def sample_noise(self, x_0, t: typing.Optional[int] = None):
        if t is None:
            t = self.timesteps - 1
        t = torch.full((x_0.shape[0],), t, device = x_0.device)
        return self.diffusion.q_sample(x_0, t)


class GaussianDiffuser2Dprev(torch.nn.Module):
    def __init__(self, channels: int, resolution: int, timesteps: int = 1000):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.timesteps = timesteps
        # dim_mults = [1, 1, 2, 2, 4]  # ver 7
        dim_mults = [1, 2, 2, 4, 4]
        # dim_mults = [1, 2, 4, 8, 8]  # ver 9
        # dim_mults = [1, 2, 4, 8, 8]

        u_net = denoising_diffusion_pytorch.Unet(
                dim=128,
                channels=channels,
                dim_mults=dim_mults,
                flash_attn=True)

        def init_weights(m):
            # if isinstance(m, torch.nn.Conv2d): #  or isinstance(m, torch.nn.Linear):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        u_net.apply(init_weights)

        self.diffusion = denoising_diffusion_pytorch.GaussianDiffusion(
            u_net,
            image_size=resolution,
            timesteps=timesteps,
            auto_normalize=False,
            min_snr_loss_weight=True,
            objective='pred_noise',
            beta_schedule='cosine',
            # ddim
            sampling_timesteps=50,
            ddim_sampling_eta=.1
        )
        self.denoiser = self.diffusion.model
        self.betas = self.diffusion.betas.cpu().numpy()
        self.alphas = 1 - self.diffusion.betas.cpu().numpy()
        self.alpha_hat = self.diffusion.alphas_cumprod.cpu().numpy()
        self.alpha_prev_hat = self.diffusion.alphas_cumprod_prev.cpu().numpy()
        self.sqrt_alpha_hat = self.diffusion.sqrt_alphas_cumprod.cpu().numpy()
        self.sqrt_one_minus_alpha_hat = self.diffusion.sqrt_one_minus_alphas_cumprod.cpu().numpy()

    def forward(self, *args):
        x_0, = args
        return self.diffusion(x_0)

    def forward_loss(self, *args):
        return self(*args)

    def predict_noise(self, x_t, t: int):
        t = torch.full((x_t.shape[0],), t, device = x_t.device)
        return self.denoiser(x_t, t)

    def eval_score(self, x_t, t: int):
        e = self.predict_noise(x_t, t)
        return -e/np.sqrt(1 - self.alpha_hat[t])

    def reverse_diffusion_DDPM(self, x_t: torch.Tensor, t: typing.Optional[int] = None, steps: typing.Optional[int] = None):
        if t is None:
            t = self.timesteps  # Assumed to be the noise
        if steps is None:
            steps = t  # Assumed to be all steps to 0.
        assert t - steps >= 0
        for i in range(t - 1, t - steps - 1, -1):
            # model predictions
            e = self.predict_noise(x_t, i)
            # computing x_{t-1}
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_prev_hat[i]
            sigma = np.sqrt(beta * (1 - alpha_prev_hat)/(1 - alpha_hat))
            z = torch.randn_like(x_t) if i > 0 else 0.0
            c1 = 1/np.sqrt(alpha)
            c2 = ((1 - alpha)/(np.sqrt(1 - alpha_hat)))/np.sqrt(alpha)
            x_i = c1 * x_t - c2 * e
            x_i = x_i + sigma * z
            x_t = x_i
        # x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t

    def reverse_diffusion_DDIM(self, x_t: torch.Tensor, t: typing.Optional[int] = None, steps: typing.Optional[int] = None, eta: float = 0.5, steps_rate: int = 1):
        if t is None:
            t = self.timesteps-1  # Assumed to be the noise
        if steps is None:
            steps = t  # Assumed to be all steps to 0.
        assert steps <= t
        TOTAL = max(1, steps // steps_rate)
        FINAL = self.timesteps * 1 // 100
        if TOTAL <= FINAL:
            tt = list(range(-1, t + 1))
            tt.reverse()
        else:
            # tt = list(int(i) for i in torch.arange(0, t, t/TOTAL).long())
            # tt.reverse()
            # tt = [t] + tt + [-1]
            tt = list(int(i) for i in torch.arange(FINAL, t, (t - FINAL) / (TOTAL - FINAL)).long())
            tt.reverse()
            tt = [t] + tt + list(range(FINAL - 1, -2, -1))
        iterations = tqdm(range(len(tt) - 1), 'Unconditional sampling DDIM')
        for i in iterations:
            current_i = tt[i]
            next_i = tt[i + 1]
            alpha_hat = self.alpha_hat[current_i]
            alpha_prev_hat = self.alpha_hat[next_i] if next_i >= 0 else 1.0
            ## Computing x_{t-1} ##
            e = self.predict_noise(x_t, current_i)
            # x0_hat = np.sqrt(1 / max(alpha_hat, 0.0001)) * x_t - np.sqrt(1 / max(alpha_hat, 0.0001) - 1) * e
            x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
            x0_hat = torch.clamp(x0_hat, -4., 4.)
            # x0_hat *= 1.0 / max(1.0, x0_hat.abs().max().item()*0.4)
            sigma = eta * np.sqrt ((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
            c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
            z = torch.randn_like(x_t) if current_i > 0 else 0.0
            x_t = np.sqrt(alpha_prev_hat) * x0_hat + c * e
            x_t = x_t + sigma * z
            # x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t

    def forward_diffusion(self, x_k: torch.Tensor, k: typing.Optional[int] = None, steps: typing.Optional[int] = None, noise: torch.Tensor = None):
        if k is None:
            k = 0
        if steps is None:
            steps = self.timesteps
        if steps == 0:
            return x_k
        if noise is None:
            noise = torch.randn_like(x_k)
        assert k + steps <= self.timesteps
        t = k + steps - 1
        alpha_hat_t = self.alpha_hat[t]
        alpha_hat_km1 = 1 if k <= 1 else self.alpha_hat[k - 2]
        alpha_cum = alpha_hat_t / alpha_hat_km1
        return x_k * np.sqrt(alpha_cum) + np.sqrt(1 - alpha_cum)*noise

    def estimate_x0(self, x_t: torch.Tensor, t: int):
        alpha_hat = self.alpha_hat[t]
        e = self.predict_noise(x_t, t)
        x0_hat = np.sqrt(1 / (alpha_hat + 0.001)) * x_t - np.sqrt(1 / (alpha_hat+0.001) - 1) * e
        return torch.clamp(x0_hat, -1.0, 1.0)

    def posterior_sampling_DPS_DDIM(self,
                                x_t: torch.Tensor,
                                t: int,
                                eta: float = 1.0,
                                weight: float = 1.0,
                                steps: int = 100,
                                criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        # t = self.timesteps  # Assumed to be the noise
        TOTAL = steps
        FINAL = 40
        if t <= TOTAL:
            tt = list(range(-1, t))
            tt.reverse()
        else:
            # tt = list(int(i) for i in torch.arange(0, t, t/TOTAL).long())
            # tt.reverse()
            # tt = [t] + tt + [-1]
            tt = list(int(i) for i in torch.arange(FINAL, t - 1, (t - FINAL)/(TOTAL - FINAL)).long())
            tt.reverse()
            tt = [t - 1] + tt + list(range(FINAL - 1, -2, -1))
        iterations = tqdm(range(len(tt)-1), 'Posterior sampling DPS_DDIM')
        ema_grad_xt = None
        for i in iterations:
            current_i = tt[i]
            next_i = tt[i+1]
            alpha_hat = self.alpha_hat[current_i]
            alpha_prev_hat = self.alpha_hat[next_i] if next_i >= 0 else 1.0
            steps = current_i - next_i
            # model predictions
            with torch.enable_grad():
                x_t.requires_grad_()
                e = self.predict_noise(x_t, current_i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / max(alpha_hat, 0.001)) * x_t - np.sqrt(1 / max(alpha_hat, 0.001) - 1) * e
                # x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
                x0_hat /= max(1.0, x0_hat.abs().max().item())
                # x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                x0_hat_detach = x0_hat.detach()
                sigma = eta * np.sqrt((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
                z = torch.randn_like(x_t) if current_i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat_detach + c * e.detach()
                x_next += sigma * z
                if current_i > 0 and criteria is not None:
                    loss = criteria(x0_hat[0])
                    # loss.backward()
                    # grad_xt = x_t.grad.clone()
                    grad_xt = torch.autograd.grad(loss, x_t)[0]
                    grad_norm = max(.001, np.sqrt(loss.item()))
                    if True:  #ema_grad_xt is None:
                        ema_grad_xt = grad_xt
                    else:
                        a = 0.2 # (0.95 * (current_i / (self.timesteps - 1)) + 0.05)
                        ema_grad_xt = ema_grad_xt * (1 - a) + grad_xt * a
                    x_next -= weight * 0.5 * ema_grad_xt * steps / grad_norm
                    # x_next -= weight * 0.5 * grad_xt * steps_rate / grad_norm
                x_t.detach_()
            x_t = x_next
            # x_t = torch.clamp(x_next, -1.0, 1.0)
        return x_t

    def parameterized_posterior_sampling_DPS(self,
                                x_t: torch.Tensor,
                                eta: float = 1.0,
                                weight: float = 1.0,
                                external_parameters: typing.Optional[typing.List[torch.Tensor]] = None,
                                criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Samples p(x0|y) and estimates dA(x0)/dtheta
        """
        if external_parameters is None:
            external_parameters = []
        estimated_grads = [torch.zeros_like(p) for p in external_parameters]
        power = 0.9
        weights = [power**i for i in range(self.timesteps)]
        weight_sum = sum(weights)
        weights = [w/weight_sum for w in weights]
        weights.reverse()
        t = self.timesteps  # Assumed to be the noise
        steps_rate = 1
        timesteps = tqdm(range(t - 1, -steps_rate, -steps_rate), 'Posterior sampling DPS_DDIM')
        for s in timesteps:
            # eta *= 0.994
            i = max(0, s)
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_hat[max(0, i - steps_rate)] if i > 0 else 1.0
            # model predictions
            with torch.enable_grad():
                x_t.requires_grad_()
                e = self.predict_noise(x_t, i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                x0_hat_detach = x0_hat.detach()
                sigma = eta * np.sqrt((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
                z = torch.randn_like(x_t) if i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat_detach + c * e.detach()
                x_next += sigma * z
                if criteria is not None:
                    loss = criteria(x0_hat[0])
                    all_grads = torch.autograd.grad(loss, [x_t]+external_parameters)
                    grad_xt = all_grads[0]
                    grad_norm = max(.001, np.sqrt(loss.item()))
                # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
                x_next -= weight * 0.5 * grad_xt * alpha_hat / grad_norm
                x_t.detach_()
            x_t = x_next
            # x_t = torch.clamp(x_next, -1.0, 1.0)
        return x0_hat_detach

    def posterior_sampling_DPSX(self,
                                x_t: torch.Tensor,
                                t: int,
                                y: torch.Tensor,
                                eta: float = 1.0,
                                weight: float = 1.0,
                                criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        if criteria is None:  # Assume identity
            criteria = lambda x: x
        # t = self.timesteps  # Assumed to be the noise
        steps_rate = max(1, t // 100)
        timesteps = tqdm(range(t - 1, -steps_rate, -steps_rate), 'Posterior sampling DPS_DDIM')
        for s in timesteps:
            # eta *= 0.994
            i = max(0, s)
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_hat[max(0, i - steps_rate)] if i > 0 else 1.0
            # model predictions
            with torch.enable_grad():
                x_t.requires_grad_()
                e = self.predict_noise(x_t, i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                x0_hat_detach = x0_hat.detach()
                sigma = eta * np.sqrt((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
                z = torch.randn_like(x_t) if i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat_detach + c * e.detach()
                x_next += sigma * z
                if i > 0:
                    y_hat = criteria(x0_hat[0])
                    loss = ((y_hat - y) ** 2).sum()
                    # loss.backward()
                    # grad_xt = x_t.grad
                    grad_xt = torch.autograd.grad(loss, x_t)[0]
                    grad_norm = max(.001, np.sqrt(loss.item()))
                    # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
                    x_next -= weight * 0.5 * grad_xt * alpha_hat * steps_rate / grad_norm
                x_t.detach_()
            x_t = x_next
            # x_t = torch.clamp(x_next, -1.0, 1.0)
        return x_t

    def posterior_sampling_DPS(self, x_t: torch.Tensor, criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        # with torch.no_grad():
        x0_hat = x_t  # just for consistency of the while
        for i in range(self.timesteps-1, -1, -1):
            alpha = self.alphas[i]
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_prev_hat[i]
            beta = self.betas[i]
            sigma = np.sqrt(beta * (1 - alpha_prev_hat)/(1 - alpha_hat))
            x_t.requires_grad_()
            s = self.eval_score(x_t, i)
            x0_hat = np.sqrt(1 / alpha_hat) * (x_t + (1 - alpha_hat) * s)
            z = torch.randn_like(x_t)
            x_next = np.sqrt(alpha) * (1 - alpha_prev_hat) / (1 - alpha_hat) * x_t.detach()
            x_next += np.sqrt(alpha_prev_hat)*beta/(1 - alpha_hat) * x0_hat.detach()
            x_next += sigma * z
            # Add conditioning by criteria
            if criteria is not None:
                with torch.enable_grad():
                    # x0_hat = 1 / np.sqrt(alpha_hat) * (x_prime - np.sqrt(1 - alpha_hat) * e)
                    loss = criteria(x0_hat)
                    # loss = loss / (0.01 + np.sqrt(loss.item()))
                    grad_xt = torch.autograd.grad(loss, x_t)[0]
                # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
                grad_norm = max(0.5, np.sqrt(loss.item()))
                x_next -= 0.05 * grad_xt / grad_norm
            x_t = x_next
        return x_t

    def optimize_for_HDC(self, x: torch.Tensor, criteria: typing.Callable[[torch.Tensor], torch.Tensor]):
        # print('Optimizing...')
        x = x.detach().clone()
        x.requires_grad_()
        optimizer = torch.optim.NAdam([x], lr=0.01)
        loss = 0.0
        N = 20
        start_loss = 0
        for s in range(N):
            optimizer.zero_grad()
            loss = criteria(x[0])
            # if s == 0:
            #     print(f"Starting loss: {loss}")
            # if s == N//2:
            #     print(f"Middle loss: {loss}")
            loss.backward()
            optimizer.step()
            if s == 0:
                start_loss = loss.item()
            elif loss.item() / (0.001 + start_loss) < 0.01:
                break
        # print(f"Ending loss: {loss}")
        # print('Done...')
        return x.detach().clone()

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        if a_t == 1 or a_t == 0.0 or sigma == 0.0:
            return x_t
        noise = torch.randn_like(pseudo_x0)
        mean = (sigma * np.sqrt(a_t) * pseudo_x0 + (1 - a_t)*x_t)/(sigma + 1 - a_t)
        sd = np.sqrt(sigma*(1 - a_t) / (sigma + 1 - a_t))
        return mean + sd * noise
        # return (sigma * np.sqrt(a_t) * pseudo_x0 + (1 - a_t) * x_t) / (sigma + 1 - a_t) + noise * np.sqrt(
        #     1 / (1 / sigma + 1 / (1 - a_t)))

    def posterior_sampling_HDC(self,
                               x_t: torch.Tensor,
                               t: int,
                               eta: float = 1.0,
                               weight: float = 1.0,
                               criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Implementation for the paper
        Hard Data Consistency
        """
        steps_rate = max(1, t // 100)
        timesteps = tqdm(range(t - 1, -steps_rate, -steps_rate), 'Posterior sampling DPS_DDIM')
        for s in timesteps:
            i = max(0, s)
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_hat[max(0, i - steps_rate)] if i > 0 else 1.0
            # model predictions
            with torch.enable_grad():
                x_t.requires_grad_()
                e = self.predict_noise(x_t, i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                x0_hat_detach = x0_hat.detach()
                sigma = eta * np.sqrt((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
                z = torch.randn_like(x_t) if i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat_detach + c * e.detach()
                x_next += sigma * z
                if criteria is not None:
                    # loss = criteria(x0_hat)
                    # grad_xt = torch.autograd.grad(loss, x_t)[0]
                    # grad_norm = max(.0000001, np.sqrt(loss.item()))
                    # Working for transmittance x_next -= 0.5 * grad_xt * alpha_hat / grad_norm
                    # x_next -= 0.05 * grad_xt * alpha_hat # / grad_norm # * alpha_hat
                    # x_next = self.stochastic_resample(x0_hat.detach(), x_next, alpha_prev_hat, 1*sigma**2)
                    # if i % 10 == 0: # and i <= self.timesteps * 2 // 3:  # Perform HDC every 10 steps and DPS the rest
                        # x_next -= .5 * grad_xt * alpha_hat / grad_norm
                    if i > 0:
                        x0_hat = self.optimize_for_HDC(x0_hat.detach(), criteria)
                        if i % 100 == 0:
                            print(f"[INFO] Distance {torch.sqrt(criteria(x0_hat[0])).item()}")
                        x_next = self.stochastic_resample(x0_hat, x_next, alpha_prev_hat, 1*sigma**2)
                x_t.detach_()
                # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
            x_t = torch.clamp(x_next, -1.0, 1.0)
            # x_t = x_next
        return x_t

    def closest_sampling_HDC(self, x0: torch.Tensor, eta: float = 1.0):
        """
        Implementation for getting the closest
        """
        x_t = self.sample_noise(x0).clone().detach()
        t = self.timesteps  # Assumed to be the noise
        steps_rate = 10
        for s in range(t - 1, -steps_rate, -steps_rate):
            # eta *= 0.995
            i = max(0, s)
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_hat[max(0, i - steps_rate)] if i > 0 else 1.0
            # model predictions
            with torch.enable_grad():
                x_t = x_t.clone().requires_grad_()
                e = self.predict_noise(x_t, i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / max(alpha_hat, 0.001)) * x_t - np.sqrt(1 / max(alpha_hat, 0.001) - 1) * e
                # x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                sigma = np.sqrt((1 - alpha_hat / alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - (eta * sigma) ** 2)
                z = torch.randn_like(x_t) if i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat.detach() + c * e.detach()
                x_next += eta * sigma * z
                if i > 0:
                    loss = ((x0_hat - x0)**2).sum()
                    grad_xt = torch.autograd.grad(loss, x_t)[0]
                    grad_norm = max(.0001, np.sqrt(loss.item()))
                    # Working for transmittance x_next -= 0.5 * grad_xt * alpha_hat / grad_norm
                    x_next -= steps_rate * .5 * grad_xt / grad_norm # / grad_norm  # * alpha_hat # / grad_norm # * alpha_hat
                # if i % 100 == 0:
                #     print(f"[INFO] Distance {torch.sqrt(criteria(x0_hat)).item()}")
                # x0_guide = x0_hat.detach() * 0.9 + x0 * 0.1
                # x_next = self.stochastic_resample(x0_guide, x_next, alpha_prev_hat, 1000*(s / 1000)*(sigma**2))
                    # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
                x_t.detach_()
                # x_t = torch.clamp(x_next, -1.0, 1.0)
                x_t = x_next
        return x_t

    def eval_generator(self, x_t, t: typing.Optional[int] = None):
        if t is None:
            t = self.timesteps
        for t in tqdm(reversed(range(t)), total=t, desc='Generating'):
            x_t, _ = self.diffusion.p_sample(x_t, t)
        return x_t

    def eval_generator_ddim(self, x_t, t: typing.Optional[int] = None):
        if t is None:
            t = self.timesteps
        return self.diffusion.ddim_sample(x_t.shape)
        # for t in tqdm(reversed(range(t)), total=t, desc='Generating'):
        #     x_t, _ = self.diffusion.p_sample(x_t, t)
        # return x_t

    def sample_noise(self, x_0, t: typing.Optional[typing.Union[int, torch.Tensor]] = None, noise: typing.Optional[torch.Tensor] = None):
        if t is None:
            t = self.timesteps - 1
        if isinstance(t, int):
            t = torch.full((x_0.shape[0],), t, device = x_0.device)
        return self.diffusion.q_sample(x_0, t, noise=noise)

    def sample_epsilon(self, x_t, t: typing.Union[int, torch.Tensor]):
        if isinstance(t, int):
            t = torch.full((x_t.shape[0],), t, device = x_t.device)
        return self.diffusion.model_predictions(x_t, t).pred_noise

    def p_sample(self, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.diffusion.p_mean_variance(x = x, t = batched_times, clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def predict_xstart(self, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.diffusion.p_mean_variance(x = x, t = batched_times, clip_denoised=True)
        return x_start


    def p_losses(self, x_0, t: typing.Union[int, torch.Tensor]):
        if isinstance(t, int):
            t = torch.full((x_0.shape[0],), t, device = x_0.device)
        return self.diffusion.p_losses(x_0, t)

    # Diffusion Posterior Sampling
    # Operator represents the measurement function
    def retrieve_grad(self, operator, x_prev, x_0_hat, measurement):
        x_0_hat_measurement = operator(x_0_hat)
        difference = measurement - x_0_hat_measurement
        # norm = (difference ** 2).sum()
        norm = torch.linalg.norm(difference.flatten(), ord=2)
        # norm = torch.nn.functional.l1_loss(measurement, x_0_hat_measurement, reduction='mean')
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, x_0_hat_measurement
        # return norm_grad / (0.001 + torch.sqrt(norm).item()), x_0_hat_measurement
        # return norm_grad, x_0_hat_measurement

    def conditioning(self, operator, x_prev, x_t, x_0_hat, measurement, scale):
        """
        Posterior sampling conditioning method
        """
        norm_grad, x_0_hat_measurement = self.retrieve_grad(operator=operator, x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement)
        return x_t - norm_grad * scale, x_0_hat_measurement

    def p_sample_loop_with_conditioning_DPS(self,
                                        x_t: torch.Tensor,
                                        measurement_function: typing.Callable[[torch.Tensor], torch.Tensor],
                                        measurement: torch.Tensor,
                                        scale: float = 1.0,
                                        callback: typing.Callable[[int, torch.Tensor, torch.Tensor], None] = None
                      ):
        pbar = tqdm(list(range(self.timesteps))[::-1])

        for t in pbar:
            x_t.requires_grad_()
            x_tm1, x_0_hat = self.p_sample(x_t, t)
            x_tm1, x_0_hat_measurement = self.conditioning(
                operator=measurement_function,
                x_prev=x_t,
                x_t=x_tm1,
                x_0_hat=x_0_hat,
                measurement=measurement,
                scale=scale * self.alphas[t]# * sigmoid(((t / self.timesteps) - 0.1)*10)
                # scale=scale / sub_steps  # * np.exp(-(1 - t / self.timesteps)*4) # * sigmoid(((1 - t / self.timesteps) - 0.5)*4)
            )
            x_t = x_tm1.detach_()
            if callback is not None:
                callback(t, x_0_hat, x_0_hat_measurement)

        # num_prewarms = 2
        #
        # for s in range(num_prewarms + 1):
        #     pbar = tqdm(list(range(self.timesteps))[::-1])
        #     for t in pbar:
        #         x_t.requires_grad_()
        #         x_tm1, x_0_hat = self.p_sample(x_t, t)
        #         x_tm1, x_0_hat_measurement = self.conditioning(
        #             operator=measurement_function,
        #             x_prev=x_t,
        #             x_t=x_tm1,
        #             x_0_hat=x_0_hat,
        #             measurement=measurement,
        #             # scale=scale # * (5 if s < 2 else 1) # * ((1 - t / self.timesteps)*0.8 + 0.2)
        #             scale = scale # * ((1 - t / self.timesteps)*4 + 1)
        #         )
        #         x_t = x_tm1.detach_()
        #         if callback is not None:
        #             callback(t, x_0_hat, x_0_hat_measurement)
        #         if t < self.timesteps//2 and s < num_prewarms:  # interrumpt here
        #             x_t = self.sample_noise(x_0_hat)
        #             break
        return x_t

    def optimize_for_x0Ixt_y (self, x_t: torch.Tensor, measurement_function, measurement):
        # Optimize for x0|y
        # x_t is initialized with x_0_hat
        if not x_t.requires_grad:
            x_t = x_t.requires_grad_()
        lr_val = 5e-3
        loss = torch.nn.MSELoss()  # MSE loss
        optimizer = torch.optim.AdamW([x_t], lr=lr_val)  # Initializing optimizer ###change the learning rate
        # Training loop
        for itr in range(500):
            optimizer.zero_grad()
            output = loss(measurement, measurement_function(x_t))
            output.backward()  # Take GD step
            optimizer.step()
            cur_loss = output.item()
            # Convergence criteria
            if cur_loss < 1e-3 ** 2:  # needs tuning according to noise level for early stopping
                break
        return x_t

    def p_sample_loop_with_conditioning_HDC(self,
                                        x_t: torch.Tensor,
                                        measurement_function: typing.Callable[[torch.Tensor], torch.Tensor],
                                        measurement: torch.Tensor,
                                        scale: float = 1.0,
                                        callback: typing.Callable[[int, torch.Tensor, torch.Tensor], None] = None
                      ):
        pbar = tqdm(list(range(self.timesteps))[::-1])
        for t in pbar:
            x_t.requires_grad_()
            x_tm1, x_0_hat = self.p_sample(x_t, t)
            x_tm1, x_0_hat_measurement = self.conditioning(
                operator=measurement_function,
                x_prev=x_t,
                x_t=x_tm1,
                x_0_hat=x_0_hat,
                measurement=measurement,
                scale=scale # * sigmoid(((t / self.timesteps) - 0.1)*10)
                # scale=scale / sub_steps  # * np.exp(-(1 - t / self.timesteps)*4) # * sigmoid(((1 - t / self.timesteps) - 0.5)*4)
            )
            x_t = x_tm1.detach_()
            if callback is not None:
                callback(t, x_0_hat, x_0_hat_measurement)

        # num_prewarms = 2
        #
        # for s in range(num_prewarms + 1):
        #     pbar = tqdm(list(range(self.timesteps))[::-1])
        #     for t in pbar:
        #         x_t.requires_grad_()
        #         x_tm1, x_0_hat = self.p_sample(x_t, t)
        #         x_tm1, x_0_hat_measurement = self.conditioning(
        #             operator=measurement_function,
        #             x_prev=x_t,
        #             x_t=x_tm1,
        #             x_0_hat=x_0_hat,
        #             measurement=measurement,
        #             # scale=scale # * (5 if s < 2 else 1) # * ((1 - t / self.timesteps)*0.8 + 0.2)
        #             scale = scale # * ((1 - t / self.timesteps)*4 + 1)
        #         )
        #         x_t = x_tm1.detach_()
        #         if callback is not None:
        #             callback(t, x_0_hat, x_0_hat_measurement)
        #         if t < self.timesteps//2 and s < num_prewarms:  # interrumpt here
        #             x_t = self.sample_noise(x_0_hat)
        #             break
        return x_t


class GaussianDiffuser2D(torch.nn.Module):
    def __init__(self, channels: int, resolution: int, timesteps: int = 1000):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.timesteps = timesteps
        # dim_mults = [1, 1, 2, 2, 4]  # ver 7
        dim_mults = [
            1, # 64x64, 128
            2, # 32x32  256
            4, # 16x16  512
            4, # 8x8    512
            # 8, # 4x4    512
        ]
        # dim_mults = [1, 2, 4, 8, 8]  # ver 9
        # dim_mults = [1, 2, 4, 8, 8]

        u_net = denoising_diffusion_pytorch.Unet(
                dim=128,
                channels=channels,
                dim_mults=dim_mults,
                full_attn=(False, True, True, False),
                flash_attn=True)

        def init_weights(m):
            # if isinstance(m, torch.nn.Conv2d): #  or isinstance(m, torch.nn.Linear):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        u_net.apply(init_weights)

        self.diffusion = denoising_diffusion_pytorch.GaussianDiffusion(
            u_net,
            image_size=resolution,
            timesteps=timesteps,
            auto_normalize=False,
            min_snr_loss_weight=True,
            objective='pred_noise',
            beta_schedule='cosine',
            # ddim
            sampling_timesteps=50,
            ddim_sampling_eta=1.0
        )
        self.denoiser = self.diffusion.model
        self.betas = self.diffusion.betas.cpu().numpy()
        self.alphas = 1 - self.diffusion.betas.cpu().numpy()
        self.alpha_hat = self.diffusion.alphas_cumprod.cpu().numpy()
        self.alpha_prev_hat = self.diffusion.alphas_cumprod_prev.cpu().numpy()
        self.sqrt_alpha_hat = self.diffusion.sqrt_alphas_cumprod.cpu().numpy()
        self.sqrt_one_minus_alpha_hat = self.diffusion.sqrt_one_minus_alphas_cumprod.cpu().numpy()

    def forward(self, *args):
        x_0, = args
        return self.diffusion(x_0)

    def forward_loss(self, *args):
        return self(*args)

    def predict_noise(self, x_t, t: int):
        t = torch.full((x_t.shape[0],), t, device = x_t.device)
        return self.denoiser(x_t, t)

    def eval_score(self, x_t, t: int):
        e = self.predict_noise(x_t, t)
        return -e/np.sqrt(1 - self.alpha_hat[t])

    def reverse_diffusion_DDPM(self, x_t: torch.Tensor, t: typing.Optional[int] = None, steps: typing.Optional[int] = None):
        if t is None:
            t = self.timesteps  # Assumed to be the noise
        if steps is None:
            steps = t  # Assumed to be all steps to 0.
        assert t - steps >= 0
        for i in range(t - 1, t - steps - 1, -1):
            # model predictions
            e = self.predict_noise(x_t, i)
            # computing x_{t-1}
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_prev_hat[i]
            sigma = np.sqrt(beta * (1 - alpha_prev_hat)/(1 - alpha_hat))
            z = torch.randn_like(x_t) if i > 0 else 0.0
            c1 = 1/np.sqrt(alpha)
            c2 = ((1 - alpha)/(np.sqrt(1 - alpha_hat)))/np.sqrt(alpha)
            x_i = c1 * x_t - c2 * e
            x_i = x_i + sigma * z
            x_t = x_i
        # x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t

    def reverse_diffusion_DDIM(self,
                               x_t: torch.Tensor,
                               steps: typing.Optional[typing.List[int]] = None,
                               eta: float = 1.0,
                               callback: typing.Optional[typing.Callable[[int, int, int, torch.Tensor], None]] = None
                               ):
        if steps is None:
            steps = list(range(0, self.timesteps+1))
            steps.reverse()
        assert all(i == 0 or steps[i] <= steps[i-1] for i in range(len(steps))), "Steps has to be non-increasing sequence"
        assert all(s <= self.timesteps for s in steps), f"All steps should be in range 0..{self.timesteps}"
        steps = list(set(steps))  # remove duplicates and add special index
        steps.sort()
        steps.reverse()
        iterations = tqdm(range(len(steps)-1), 'Unconditional sampling DDIM')
        x0_ema = None
        for i in iterations:
            current_i = steps[i] - 1
            next_i = steps[i + 1] - 1
            alpha_hat = max(0.0001, self.alpha_hat[current_i])
            alpha_prev_hat = max(0.0001, self.alpha_hat[next_i]) if next_i >= 0 else 1.0
            ## Computing x_{t-1} ##
            e = self.predict_noise(x_t, current_i)
            x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
            # x0_hat = np.sqrt(1 / max(alpha_hat, 0.001)) * x_t - np.sqrt(max(0.001, 1 / alpha_hat - 1)) * e
            x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
            if x0_ema is None: # or current_i > 200:
                x0_ema = x0_hat
            else:
                alpha = 0.8 * ((self.timesteps - current_i)/self.timesteps) ** 2 #  * (((current_i + 1)/self.timesteps)**2)
                torch.add(x0_ema * alpha, other=x0_hat, alpha=1 - alpha, out=x0_ema)
            # x0_hat *= 1.0 / max(1.0, x0_hat.abs().max().item())
            if callback is not None:
                callback(i, len(steps) - 1, current_i + 1, x0_ema)
            sigma = eta * np.sqrt ((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
            c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
            z = torch.randn_like(x_t) if next_i >= 0 else 0.0
            x_t = np.sqrt(alpha_prev_hat) * x0_ema + c * e
            x_t = x_t + sigma * z
        # x_t = torch.clamp(x_t, -1.0, 1.0)
        return x0_ema

    def forward_diffusion(self, x_k: torch.Tensor, k: typing.Optional[int] = None, steps: typing.Optional[int] = None, noise: torch.Tensor = None):
        if k is None:
            k = 0
        if steps is None:
            steps = self.timesteps
        if steps == 0:
            return x_k
        if noise is None:
            noise = torch.randn_like(x_k)
        assert k + steps <= self.timesteps
        t = k + steps - 1
        alpha_hat_t = self.alpha_hat[t]
        alpha_hat_km1 = 1 if k <= 1 else self.alpha_hat[k - 2]
        alpha_cum = alpha_hat_t / alpha_hat_km1
        return x_k * np.sqrt(alpha_cum) + np.sqrt(1 - alpha_cum)*noise

    def estimate_x0(self, x_t: torch.Tensor, t: int):
        if t == 0:
            return x_t  # nothing to denoise
        alpha_hat = self.alpha_hat[t-1]
        e = self.predict_noise(x_t, t-1)
        x0_hat = np.sqrt(1 / max(0.000001, alpha_hat)) * x_t - np.sqrt(1 / max(0.000001, alpha_hat) - 1) * e
        return torch.clamp(x0_hat, -1.0, 1.0)

    def posterior_sampling_DPS_DDIM(self,
                                x_t: torch.Tensor,
                                steps: typing.List[int],
                                y: torch.Tensor,
                                A: typing.Callable[[torch.Tensor], torch.Tensor],
                                eta: float = 1.0,
                                weight: float = 1.0,
                                ema_factor: float = 1.0,
                                callback: typing.Optional[typing.Callable[[int, int, int, torch.Tensor], None]] = None):
        if steps is None:
            steps = list(range(0, self.timesteps+1))
            steps.reverse()
        assert all(i == 0 or steps[i] <= steps[i-1] for i in range(len(steps))), "Steps has to be non-increasing sequence"
        assert all(s <= self.timesteps for s in steps), f"All steps should be in range 0..{self.timesteps}"
        x_t = x_t.unsqueeze(0).detach()  # add batch dimension required by diffuser
        steps = list(set(steps))  # remove duplicates and add special index
        steps.sort()
        steps.reverse()
        iterations = tqdm(range(len(steps)-1), 'Posterior sampling DPS_DDIM')
        def step_size_scheduler_cosine(i):
            alpha = i/self.timesteps
            return (1 - np.cos(alpha ** 2 * 3.141593 * 2))*0.5
        def step_size_scheduler_exp(i):
            alpha = i/self.timesteps
            return 0.995 ** (800*(1 - alpha))
        steps_size_scheduler = step_size_scheduler_cosine

        values = []
        losses = []

        # rescale max gradient norm to account for reduced samples
        max_norm_scaler = self.timesteps / (len(steps) - 1)

        x0_ema = None
        y_ema = None

        for i in iterations:
            current_i = steps[i] - 1
            next_i = steps[i+1] - 1
            alpha_hat = self.alpha_hat[current_i]
            alpha_prev_hat = self.alpha_hat[next_i] if next_i >= 0 else 1.0
            delta_steps = current_i - next_i
            # model predictions
            with (torch.enable_grad()):
                x_t.requires_grad_()
                e = self.predict_noise(x_t, current_i)
                # computing x_{t-1}
                # x0_hat = np.sqrt(1 / max(alpha_hat, 0.00001)) * x_t - np.sqrt(1 / max(alpha_hat, 0.00001) - 1) * e
                x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
                # x0_hat /= max(1.0, x0_hat.abs().max().item())
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                # x0_hat_detach = torch.clamp(x0_hat.detach(), -1.0, 1.0)
                x0_hat_detach = x0_hat.detach()
                if x0_ema is None:
                    x0_ema = x0_hat_detach
                else:
                    alpha = (0.8 ** (delta_steps*0.25)) * (((self.timesteps - current_i) / self.timesteps) ** 2)
                    torch.add(x0_ema * alpha, other=x0_hat_detach, alpha=1 - alpha, out=x0_ema)
                if callback is not None:
                    callback(i, len(steps) - 1, current_i + 1, x0_ema)
                sigma = eta * np.sqrt((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
                z = torch.randn_like(x_t) if next_i >= 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_ema + c * e.detach()
                x_next += sigma * z
                if current_i < self.timesteps * 0.95:
                    yhat = A(x0_hat[0])
                    if y_ema is None or ema_factor == 1.0:
                        y_ema = yhat
                    else:
                        y_ema = ema_diff(yhat, y_ema, ema_factor)
                    loss = ((y - y_ema)**2).sum()
                    # loss += (x0_hat ** 2).sum() * 1e-9
                    # grad_x0 = torch.autograd.grad(loss, x0_hat)[0]
                    grad_xt = torch.autograd.grad(loss, x_t)[0]
                    y_ema = y_ema.detach()
                    # loss_norm = max(0.1, np.sqrt(loss.item())) #max(0.0001, torch.norm(grad_xt).item())# + .00001 # + np.sqrt(loss.item())

                    # sqr L2 norm of grad_xt
                    grad_norm_sqr = (grad_xt ** 2).sum().item()
                    grad_norm = np.sqrt(grad_norm_sqr) + 0.0001
                    # compute one-step size
                    # grad_xt *= (loss.item() / (grad_norm_sqr + 0.0001)) / grad_norm
                    # clamp to a max norm
                    # max_norm = steps_size_scheduler(current_i) * 20.0 * max_norm_scaler
                    # grad_xt /= max(1, np.sqrt((grad_xt ** 2).sum().item())/(max_norm+0.0001))

                    # grad_step = delta_steps * steps_size_scheduler(current_i) * weight * grad_xt / grad_norm
                    # grad_step = steps_size_scheduler(current_i) * grad_xt / grad_norm
                    # grad_step = weight * steps_size_scheduler(current_i) * grad_xt
                    grad_step = alpha_hat * max_norm_scaler * weight * 8 * (step_size_scheduler_cosine(next_i+1)**0.5) * grad_xt / grad_norm
                    # grad_step_norm = np.sqrt((grad_step ** 2).sum().item())
                    # max_norm = 5.0 * max_norm_scaler
                    # grad_step /= max(1, grad_step_norm/(max_norm+0.0001))

                    x_next -= grad_step # / loss_norm # / loss_norm#  * loss_norm / grad_norm
                    values.append(np.sqrt((grad_step**2).sum().item()))
                    losses.append(loss.item())
                    # x_next -= alpha_hat * np.sqrt(1 - alpha_prev_hat) * 0.5 * delta_steps * weight * grad_xt # / loss_norm#  * loss_norm / grad_norm
                    # x_next -= np.sqrt(1 - alpha_hat) * delta_steps * weight * grad_xt / grad_norm
                    # x_next -= (sigma) * weight * grad_xt * delta_steps / grad_norm
                    # x_next -= np.sqrt(alpha_hat / alpha_prev_hat) * weight * 0.5 * grad_xt * delta_steps  # / grad_norm
                x_t.detach_()
            x_t = x_next.detach()

        # mean_losses = sum(losses[len(losses)//2:]) * 2 / len(losses)
        # p = 10 ** (int(np.log10(mean_losses))+1)
        #
        # import matplotlib.pyplot as plt
        # plt.plot(values)
        # plt.show()
        # plt.plot(losses)
        # plt.gca().set_ylim(0, p)
        # plt.show()

        return x0_ema

    def parameterized_posterior_sampling_DPS(self,
                                x_t: torch.Tensor,
                                eta: float = 1.0,
                                weight: float = 1.0,
                                external_parameters: typing.Optional[typing.List[torch.Tensor]] = None,
                                criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Samples p(x0|y) and estimates dA(x0)/dtheta
        """
        if external_parameters is None:
            external_parameters = []
        estimated_grads = [torch.zeros_like(p) for p in external_parameters]
        power = 0.9
        weights = [power**i for i in range(self.timesteps)]
        weight_sum = sum(weights)
        weights = [w/weight_sum for w in weights]
        weights.reverse()
        t = self.timesteps  # Assumed to be the noise
        steps_rate = 1
        timesteps = tqdm(range(t - 1, -steps_rate, -steps_rate), 'Posterior sampling DPS_DDIM')
        for s in timesteps:
            # eta *= 0.994
            i = max(0, s)
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_hat[max(0, i - steps_rate)] if i > 0 else 1.0
            # model predictions
            with torch.enable_grad():
                x_t.requires_grad_()
                e = self.predict_noise(x_t, i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                x0_hat_detach = x0_hat.detach()
                sigma = eta * np.sqrt((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
                z = torch.randn_like(x_t) if i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat_detach + c * e.detach()
                x_next += sigma * z
                if criteria is not None:
                    loss = criteria(x0_hat[0])
                    all_grads = torch.autograd.grad(loss, [x_t]+external_parameters)
                    grad_xt = all_grads[0]
                    grad_norm = max(.001, np.sqrt(loss.item()))
                # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
                x_next -= weight * 0.5 * grad_xt * alpha_hat / grad_norm
                x_t.detach_()
            x_t = x_next
            # x_t = torch.clamp(x_next, -1.0, 1.0)
        return x0_hat_detach

    def posterior_sampling_DPSX(self,
                                x_t: torch.Tensor,
                                t: int,
                                y: torch.Tensor,
                                eta: float = 1.0,
                                weight: float = 1.0,
                                criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        if criteria is None:  # Assume identity
            criteria = lambda x: x
        # t = self.timesteps  # Assumed to be the noise
        steps_rate = max(1, t // 100)
        timesteps = tqdm(range(t - 1, -steps_rate, -steps_rate), 'Posterior sampling DPS_DDIM')
        for s in timesteps:
            # eta *= 0.994
            i = max(0, s)
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_hat[max(0, i - steps_rate)] if i > 0 else 1.0
            # model predictions
            with torch.enable_grad():
                x_t.requires_grad_()
                e = self.predict_noise(x_t, i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                x0_hat_detach = x0_hat.detach()
                sigma = eta * np.sqrt((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
                z = torch.randn_like(x_t) if i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat_detach + c * e.detach()
                x_next += sigma * z
                if i > 0:
                    y_hat = criteria(x0_hat[0])
                    loss = ((y_hat - y) ** 2).sum()
                    # loss.backward()
                    # grad_xt = x_t.grad
                    grad_xt = torch.autograd.grad(loss, x_t)[0]
                    grad_norm = max(.001, np.sqrt(loss.item()))
                    # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
                    x_next -= weight * 0.5 * grad_xt * alpha_hat * steps_rate / grad_norm
                x_t.detach_()
            x_t = x_next
            # x_t = torch.clamp(x_next, -1.0, 1.0)
        return x_t

    def posterior_sampling_DPS(self, x_t: torch.Tensor, criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        # with torch.no_grad():
        x0_hat = x_t  # just for consistency of the while
        for i in range(self.timesteps-1, -1, -1):
            alpha = self.alphas[i]
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_prev_hat[i]
            beta = self.betas[i]
            sigma = np.sqrt(beta * (1 - alpha_prev_hat)/(1 - alpha_hat))
            x_t.requires_grad_()
            s = self.eval_score(x_t, i)
            x0_hat = np.sqrt(1 / alpha_hat) * (x_t + (1 - alpha_hat) * s)
            z = torch.randn_like(x_t)
            x_next = np.sqrt(alpha) * (1 - alpha_prev_hat) / (1 - alpha_hat) * x_t.detach()
            x_next += np.sqrt(alpha_prev_hat)*beta/(1 - alpha_hat) * x0_hat.detach()
            x_next += sigma * z
            # Add conditioning by criteria
            if criteria is not None:
                with torch.enable_grad():
                    # x0_hat = 1 / np.sqrt(alpha_hat) * (x_prime - np.sqrt(1 - alpha_hat) * e)
                    loss = criteria(x0_hat)
                    # loss = loss / (0.01 + np.sqrt(loss.item()))
                    grad_xt = torch.autograd.grad(loss, x_t)[0]
                # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
                grad_norm = max(0.5, np.sqrt(loss.item()))
                x_next -= 0.05 * grad_xt / grad_norm
            x_t = x_next
        return x_t

    def optimize_for_HDC(self, x: torch.Tensor, criteria: typing.Callable[[torch.Tensor], torch.Tensor]):
        # print('Optimizing...')
        x = x.detach().clone()
        x.requires_grad_()
        optimizer = torch.optim.NAdam([x], lr=0.01)
        loss = 0.0
        N = 20
        start_loss = 0
        for s in range(N):
            optimizer.zero_grad()
            loss = criteria(x[0])
            # if s == 0:
            #     print(f"Starting loss: {loss}")
            # if s == N//2:
            #     print(f"Middle loss: {loss}")
            loss.backward()
            optimizer.step()
            if s == 0:
                start_loss = loss.item()
            elif loss.item() / (0.001 + start_loss) < 0.01:
                break
        # print(f"Ending loss: {loss}")
        # print('Done...')
        return x.detach().clone()

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        if a_t == 1 or a_t == 0.0 or sigma == 0.0:
            return x_t
        noise = torch.randn_like(pseudo_x0)
        mean = (sigma * np.sqrt(a_t) * pseudo_x0 + (1 - a_t)*x_t)/(sigma + 1 - a_t)
        sd = np.sqrt(sigma*(1 - a_t) / (sigma + 1 - a_t))
        return mean + sd * noise
        # return (sigma * np.sqrt(a_t) * pseudo_x0 + (1 - a_t) * x_t) / (sigma + 1 - a_t) + noise * np.sqrt(
        #     1 / (1 / sigma + 1 / (1 - a_t)))

    def posterior_sampling_HDC(self,
                               x_t: torch.Tensor,
                               t: int,
                               eta: float = 1.0,
                               weight: float = 1.0,
                               criteria: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Implementation for the paper
        Hard Data Consistency
        """
        steps_rate = max(1, t // 100)
        timesteps = tqdm(range(t - 1, -steps_rate, -steps_rate), 'Posterior sampling DPS_DDIM')
        for s in timesteps:
            i = max(0, s)
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_hat[max(0, i - steps_rate)] if i > 0 else 1.0
            # model predictions
            with torch.enable_grad():
                x_t.requires_grad_()
                e = self.predict_noise(x_t, i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / alpha_hat) * x_t - np.sqrt(1 / alpha_hat - 1) * e
                x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                x0_hat_detach = x0_hat.detach()
                sigma = eta * np.sqrt((1 - alpha_hat/alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - sigma ** 2)
                z = torch.randn_like(x_t) if i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat_detach + c * e.detach()
                x_next += sigma * z
                if criteria is not None:
                    # loss = criteria(x0_hat)
                    # grad_xt = torch.autograd.grad(loss, x_t)[0]
                    # grad_norm = max(.0000001, np.sqrt(loss.item()))
                    # Working for transmittance x_next -= 0.5 * grad_xt * alpha_hat / grad_norm
                    # x_next -= 0.05 * grad_xt * alpha_hat # / grad_norm # * alpha_hat
                    # x_next = self.stochastic_resample(x0_hat.detach(), x_next, alpha_prev_hat, 1*sigma**2)
                    # if i % 10 == 0: # and i <= self.timesteps * 2 // 3:  # Perform HDC every 10 steps and DPS the rest
                        # x_next -= .5 * grad_xt * alpha_hat / grad_norm
                    if i > 0:
                        x0_hat = self.optimize_for_HDC(x0_hat.detach(), criteria)
                        if i % 100 == 0:
                            print(f"[INFO] Distance {torch.sqrt(criteria(x0_hat[0])).item()}")
                        x_next = self.stochastic_resample(x0_hat, x_next, alpha_prev_hat, 1*sigma**2)
                x_t.detach_()
                # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
            x_t = torch.clamp(x_next, -1.0, 1.0)
            # x_t = x_next
        return x_t

    def closest_sampling_HDC(self, x0: torch.Tensor, eta: float = 1.0):
        """
        Implementation for getting the closest
        """
        x_t = self.sample_noise(x0).clone().detach()
        t = self.timesteps  # Assumed to be the noise
        steps_rate = 10
        for s in range(t - 1, -steps_rate, -steps_rate):
            # eta *= 0.995
            i = max(0, s)
            alpha_hat = self.alpha_hat[i]
            alpha_prev_hat = self.alpha_hat[max(0, i - steps_rate)] if i > 0 else 1.0
            # model predictions
            with torch.enable_grad():
                x_t = x_t.clone().requires_grad_()
                e = self.predict_noise(x_t, i)
                # computing x_{t-1}
                x0_hat = np.sqrt(1 / max(alpha_hat, 0.001)) * x_t - np.sqrt(1 / max(alpha_hat, 0.001) - 1) * e
                # x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
                sigma = np.sqrt((1 - alpha_hat / alpha_prev_hat) * (1 - alpha_prev_hat) / (1 - alpha_hat))
                c = np.sqrt(1 - alpha_prev_hat - (eta * sigma) ** 2)
                z = torch.randn_like(x_t) if i > 0 else 0.0
                x_next = np.sqrt(alpha_prev_hat) * x0_hat.detach() + c * e.detach()
                x_next += eta * sigma * z
                if i > 0:
                    loss = ((x0_hat - x0)**2).sum()
                    grad_xt = torch.autograd.grad(loss, x_t)[0]
                    grad_norm = max(.0001, np.sqrt(loss.item()))
                    # Working for transmittance x_next -= 0.5 * grad_xt * alpha_hat / grad_norm
                    x_next -= steps_rate * .5 * grad_xt / grad_norm # / grad_norm  # * alpha_hat # / grad_norm # * alpha_hat
                # if i % 100 == 0:
                #     print(f"[INFO] Distance {torch.sqrt(criteria(x0_hat)).item()}")
                # x0_guide = x0_hat.detach() * 0.9 + x0 * 0.1
                # x_next = self.stochastic_resample(x0_guide, x_next, alpha_prev_hat, 1000*(s / 1000)*(sigma**2))
                    # x_next -= torch.clamp(grad_xt * sigma / (0.001 + np.sqrt(loss.item())), -0.2, 0.2)
                x_t.detach_()
                # x_t = torch.clamp(x_next, -1.0, 1.0)
                x_t = x_next
        return x_t

    def eval_generator(self, x_t, t: typing.Optional[int] = None):
        if t is None:
            t = self.timesteps
        for t in tqdm(reversed(range(t)), total=t, desc='Generating'):
            x_t, _ = self.diffusion.p_sample(x_t, t)
        return x_t

    def eval_generator_ddim(self, x_t, t: typing.Optional[int] = None):
        if t is None:
            t = self.timesteps
        return self.diffusion.ddim_sample(x_t.shape)
        # for t in tqdm(reversed(range(t)), total=t, desc='Generating'):
        #     x_t, _ = self.diffusion.p_sample(x_t, t)
        # return x_t

    def sample_noise(self, x_0, t: typing.Optional[typing.Union[int, torch.Tensor]] = None, noise: typing.Optional[torch.Tensor] = None):
        if t is None:
            t = self.timesteps - 1
        if isinstance(t, int):
            t = torch.full((x_0.shape[0],), t, device = x_0.device)
        return self.diffusion.q_sample(x_0, t, noise=noise)

    def sample_epsilon(self, x_t, t: typing.Union[int, torch.Tensor]):
        if isinstance(t, int):
            t = torch.full((x_t.shape[0],), t, device = x_t.device)
        return self.diffusion.model_predictions(x_t, t).pred_noise

    def p_sample(self, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.diffusion.p_mean_variance(x = x, t = batched_times, clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def predict_xstart(self, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.diffusion.p_mean_variance(x = x, t = batched_times, clip_denoised=True)
        return x_start


    def p_losses(self, x_0, t: typing.Union[int, torch.Tensor]):
        if isinstance(t, int):
            t = torch.full((x_0.shape[0],), t, device = x_0.device)
        return self.diffusion.p_losses(x_0, t)

    # Diffusion Posterior Sampling
    # Operator represents the measurement function
    def retrieve_grad(self, operator, x_prev, x_0_hat, measurement):
        x_0_hat_measurement = operator(x_0_hat)
        difference = measurement - x_0_hat_measurement
        # norm = (difference ** 2).sum()
        norm = torch.linalg.norm(difference.flatten(), ord=2)
        # norm = torch.nn.functional.l1_loss(measurement, x_0_hat_measurement, reduction='mean')
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, x_0_hat_measurement
        # return norm_grad / (0.001 + torch.sqrt(norm).item()), x_0_hat_measurement
        # return norm_grad, x_0_hat_measurement

    def conditioning(self, operator, x_prev, x_t, x_0_hat, measurement, scale):
        """
        Posterior sampling conditioning method
        """
        norm_grad, x_0_hat_measurement = self.retrieve_grad(operator=operator, x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement)
        return x_t - norm_grad * scale, x_0_hat_measurement

    def p_sample_loop_with_conditioning_DPS(self,
                                        x_t: torch.Tensor,
                                        measurement_function: typing.Callable[[torch.Tensor], torch.Tensor],
                                        measurement: torch.Tensor,
                                        scale: float = 1.0,
                                        callback: typing.Callable[[int, torch.Tensor, torch.Tensor], None] = None
                      ):
        pbar = tqdm(list(range(self.timesteps))[::-1])

        for t in pbar:
            x_t.requires_grad_()
            x_tm1, x_0_hat = self.p_sample(x_t, t)
            x_tm1, x_0_hat_measurement = self.conditioning(
                operator=measurement_function,
                x_prev=x_t,
                x_t=x_tm1,
                x_0_hat=x_0_hat,
                measurement=measurement,
                scale=scale * self.alphas[t]# * sigmoid(((t / self.timesteps) - 0.1)*10)
                # scale=scale / sub_steps  # * np.exp(-(1 - t / self.timesteps)*4) # * sigmoid(((1 - t / self.timesteps) - 0.5)*4)
            )
            x_t = x_tm1.detach_()
            if callback is not None:
                callback(t, x_0_hat, x_0_hat_measurement)

        # num_prewarms = 2
        #
        # for s in range(num_prewarms + 1):
        #     pbar = tqdm(list(range(self.timesteps))[::-1])
        #     for t in pbar:
        #         x_t.requires_grad_()
        #         x_tm1, x_0_hat = self.p_sample(x_t, t)
        #         x_tm1, x_0_hat_measurement = self.conditioning(
        #             operator=measurement_function,
        #             x_prev=x_t,
        #             x_t=x_tm1,
        #             x_0_hat=x_0_hat,
        #             measurement=measurement,
        #             # scale=scale # * (5 if s < 2 else 1) # * ((1 - t / self.timesteps)*0.8 + 0.2)
        #             scale = scale # * ((1 - t / self.timesteps)*4 + 1)
        #         )
        #         x_t = x_tm1.detach_()
        #         if callback is not None:
        #             callback(t, x_0_hat, x_0_hat_measurement)
        #         if t < self.timesteps//2 and s < num_prewarms:  # interrumpt here
        #             x_t = self.sample_noise(x_0_hat)
        #             break
        return x_t

    def optimize_for_x0Ixt_y (self, x_t: torch.Tensor, measurement_function, measurement):
        # Optimize for x0|y
        # x_t is initialized with x_0_hat
        if not x_t.requires_grad:
            x_t = x_t.requires_grad_()
        lr_val = 5e-3
        loss = torch.nn.MSELoss()  # MSE loss
        optimizer = torch.optim.AdamW([x_t], lr=lr_val)  # Initializing optimizer ###change the learning rate
        # Training loop
        for itr in range(500):
            optimizer.zero_grad()
            output = loss(measurement, measurement_function(x_t))
            output.backward()  # Take GD step
            optimizer.step()
            cur_loss = output.item()
            # Convergence criteria
            if cur_loss < 1e-3 ** 2:  # needs tuning according to noise level for early stopping
                break
        return x_t

    def p_sample_loop_with_conditioning_HDC(self,
                                        x_t: torch.Tensor,
                                        measurement_function: typing.Callable[[torch.Tensor], torch.Tensor],
                                        measurement: torch.Tensor,
                                        scale: float = 1.0,
                                        callback: typing.Callable[[int, torch.Tensor, torch.Tensor], None] = None
                      ):
        pbar = tqdm(list(range(self.timesteps))[::-1])
        for t in pbar:
            x_t.requires_grad_()
            x_tm1, x_0_hat = self.p_sample(x_t, t)
            x_tm1, x_0_hat_measurement = self.conditioning(
                operator=measurement_function,
                x_prev=x_t,
                x_t=x_tm1,
                x_0_hat=x_0_hat,
                measurement=measurement,
                scale=scale # * sigmoid(((t / self.timesteps) - 0.1)*10)
                # scale=scale / sub_steps  # * np.exp(-(1 - t / self.timesteps)*4) # * sigmoid(((1 - t / self.timesteps) - 0.5)*4)
            )
            x_t = x_tm1.detach_()
            if callback is not None:
                callback(t, x_0_hat, x_0_hat_measurement)

        # num_prewarms = 2
        #
        # for s in range(num_prewarms + 1):
        #     pbar = tqdm(list(range(self.timesteps))[::-1])
        #     for t in pbar:
        #         x_t.requires_grad_()
        #         x_tm1, x_0_hat = self.p_sample(x_t, t)
        #         x_tm1, x_0_hat_measurement = self.conditioning(
        #             operator=measurement_function,
        #             x_prev=x_t,
        #             x_t=x_tm1,
        #             x_0_hat=x_0_hat,
        #             measurement=measurement,
        #             # scale=scale # * (5 if s < 2 else 1) # * ((1 - t / self.timesteps)*0.8 + 0.2)
        #             scale = scale # * ((1 - t / self.timesteps)*4 + 1)
        #         )
        #         x_t = x_tm1.detach_()
        #         if callback is not None:
        #             callback(t, x_0_hat, x_0_hat_measurement)
        #         if t < self.timesteps//2 and s < num_prewarms:  # interrumpt here
        #             x_t = self.sample_noise(x_0_hat)
        #             break
        return x_t


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_model = self._copy_model(model)

    def _copy_model(self, model):
        """Create a copy of the model for EMA weights."""
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(model.state_dict())  # Copy weights
        for param in ema_model.parameters():
            param.requires_grad = False  # EMA model is not trained
        return ema_model

    def update(self):
        """Apply EMA update to the model."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = self.decay * ema_param.data + (1.0 - self.decay) * model_param.data