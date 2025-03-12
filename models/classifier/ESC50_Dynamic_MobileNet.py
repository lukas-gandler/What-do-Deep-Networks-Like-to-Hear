import torch
import math
import numpy as np
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Dict
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.misc import Conv2dNormActivation as ConvNormActivation
from torch.hub import load_state_dict_from_url
import urllib.parse

def NAME_TO_WIDTH(name):

    dymn_map = {
        'dymn04': 0.4,
        'dymn10': 1.0,
        'dymn20': 2.0
    }

    try:
        w = dymn_map[name[:6]]
    except:
        w = 1.0
    return w

#region Dynamic MobileNet - Utils
def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)

#endregion

#region Dynamic MobileNet - Dy-Block
class DynamicInvertedResidualConfig:
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_dy_block: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_dy_block = use_dy_block
        self.use_hs = activation == "HS"
        self.use_se = False
        self.stride = stride
        self.dilation = dilation
        self.width_mult = width_mult

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class DynamicConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 context_dim,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding=0,
                 groups=1,
                 att_groups=1,
                 bias=False,
                 k=4,
                 temp_schedule=(30, 1, 1, 0.05)
                 ):
        super(DynamicConv, self).__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.k = k
        self.T_max, self.T_min, self.T0_slope, self.T1_slope = temp_schedule
        self.temperature = self.T_max
        # att_groups splits the channels into 'att_groups' groups and predicts separate attention weights
        # for each of the groups; did only give slight improvements in our experiments and not mentioned in paper
        self.att_groups = att_groups

        # Equation 6 in paper: obtain coefficients for K attention weights over conv. kernels
        self.residuals = nn.Sequential(
                nn.Linear(context_dim, k * self.att_groups)
        )

        # k sets of weights for convolution
        weight = torch.randn(k, out_channels, in_channels // groups, kernel_size, kernel_size)

        if bias:
            self.bias = nn.Parameter(torch.zeros(k, out_channels), requires_grad=True)
        else:
            self.bias = None

        self._initialize_weights(weight, self.bias)

        weight = weight.view(1, k, att_groups, out_channels,
                             in_channels // groups, kernel_size, kernel_size)

        weight = weight.transpose(1, 2).view(1, self.att_groups, self.k, -1)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def _initialize_weights(self, weight, bias):
        init_func = partial(nn.init.kaiming_normal_, mode="fan_out")
        for i in range(self.k):
            init_func(weight[i])
            if bias is not None:
                nn.init.zeros_(bias[i])

    def forward(self, x, g=None):
        b, c, f, t = x.size()
        g_c = g[0].view(b, -1)
        residuals = self.residuals(g_c).view(b, self.att_groups, 1, -1)
        attention = F.softmax(residuals / self.temperature, dim=-1)

        # attention shape: batch_size x 1 x 1 x k
        # self.weight shape: 1 x 1 x k x out_channels * (in_channels // groups) * kernel_size ** 2
        aggregate_weight = (attention @ self.weight).transpose(1, 2).reshape(b, self.out_channels,
                                                                             self.in_channels // self.groups,
                                                                             self.kernel_size, self.kernel_size)

        # aggregate_weight shape: batch_size x out_channels x in_channels // groups x kernel_size x kernel_size
        aggregate_weight = aggregate_weight.view(b * self.out_channels, self.in_channels // self.groups,
                                                 self.kernel_size, self.kernel_size)
        # each sample in the batch has different weights for the convolution - therefore batch and channel dims need to
        # be merged together in channel dimension
        x = x.view(1, -1, f, t)
        if self.bias is not None:
            aggregate_bias = torch.mm(attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * b)

        # output shape: 1 x batch_size * channels x f_bands x time_frames
        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        return output

    def update_params(self, epoch):
        # temperature schedule for attention weights
        # see Equation 5: tau = temperature
        t0 = self.T_max - self.T0_slope * epoch
        t1 = 1 + self.T1_slope * (self.T_max - 1) / self.T0_slope - self.T1_slope * epoch
        self.temperature = max(t0, t1, self.T_min)
        print(f"Setting temperature for attention over kernels to {self.temperature}")


class DyReLU(nn.Module):
    def __init__(self, channels, context_dim, M=2):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.M = M

        self.coef_net = nn.Sequential(
                nn.Linear(context_dim, 2 * M)
        )

        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.] * M + [0.5] * M).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * M - 1)).float())

    def get_relu_coefs(self, x):
        theta = self.coef_net(x)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x, g):
        raise NotImplementedError


class DyReLUB(DyReLU):
    def __init__(self, channels, context_dim, M=2):
        super(DyReLUB, self).__init__(channels, context_dim, M)
        # Equation 4 in paper: obtain coefficients for M linear mappings for each of the C channels
        self.coef_net[-1] = nn.Linear(context_dim, 2 * M * self.channels)

    def forward(self, x, g):
        assert x.shape[1] == self.channels
        assert g is not None
        b, c, f, t = x.size()
        h_c = g[0].view(b, -1)
        theta = self.get_relu_coefs(h_c)

        relu_coefs = theta.view(-1, self.channels, 1, 1, 2 * self.M) * self.lambdas + self.init_v
        # relu_coefs shape: batch_size x channels x 1 x 1 x 2*M
        # x shape: batch_size x channels x f_bands x time_frames
        x_mapped = x.unsqueeze(-1) * relu_coefs[:, :, :, :, :self.M] + relu_coefs[:, :, :, :, self.M:]
        if self.M == 2:
            # torch.maximum turned out to be faster than torch.max for M=2
            result = torch.maximum(x_mapped[:, :, :, :, 0], x_mapped[:, :, :, :, 1])
        else:
            result = torch.max(x_mapped, dim=-1)[0]
        return result


class CoordAtt(nn.Module):
    def __init__(self):
        super(CoordAtt, self).__init__()

    def forward(self, x, g):
        g_cf, g_ct = g[1], g[2]
        a_f = g_cf.sigmoid()
        a_t = g_ct.sigmoid()
        # recalibration with channel-frequency and channel-time weights
        out = x * a_f * a_t
        return out


class DynamicWrapper(torch.nn.Module):
    # wrap a pytorch module in a dynamic module
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, g):
        return self.module(x)


class ContextGen(nn.Module):
    def __init__(self, context_dim, in_ch, exp_ch, norm_layer, stride: int = 1):
        super(ContextGen, self).__init__()

        # shared linear layer implemented as a 2D convolution with 1x1 kernel
        self.joint_conv = nn.Conv2d(in_ch, context_dim, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.joint_norm = norm_layer(context_dim)
        self.joint_act = nn.Hardswish(inplace=True)

        # separate linear layers for Coordinate Attention
        self.conv_f = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv_t = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)

        if stride > 1:
            # sequence pooling for Coordinate Attention
            self.pool_f = nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0))
            self.pool_t = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, stride), padding=(0, 1))
        else:
            self.pool_f = nn.Sequential()
            self.pool_t = nn.Sequential()

    def forward(self, x, g):
        cf = F.adaptive_avg_pool2d(x, (None, 1))
        ct = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        f, t = cf.size(2), ct.size(2)

        g_cat = torch.cat([cf, ct], dim=2)
        # joint frequency and time sequence transformation (S_F and S_T in the paper)
        g_cat = self.joint_norm(self.joint_conv(g_cat))
        g_cat = self.joint_act(g_cat)

        h_cf, h_ct = torch.split(g_cat, [f, t], dim=2)
        h_ct = h_ct.permute(0, 1, 3, 2)
        # pooling over sequence dimension to get context vector of size H to parameterize Dy-ReLU and Dy-Conv
        h_c = torch.mean(g_cat, dim=2, keepdim=True)
        g_cf, g_ct = self.conv_f(self.pool_f(h_cf)), self.conv_t(self.pool_t(h_ct))

        # g[0]: context vector of size H to parameterize Dy-ReLU and Dy-Conv
        # g[1], g[2]: frequency and time sequences for Coordinate Attention
        g = (h_c, g_cf, g_ct)
        return g


class DY_Block(nn.Module):
    def __init__(
            self,
            cnf: DynamicInvertedResidualConfig,
            context_ratio: int = 4,
            max_context_size: int = 128,
            min_context_size: int = 32,
            temp_schedule: tuple = (30, 1, 1, 0.05),
            dyrelu_k: int = 2,
            dyconv_k: int = 4,
            no_dyrelu: bool = False,
            no_dyconv: bool = False,
            no_ca: bool = False,
            **kwargs: Any
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        # context_dim is denoted as 'H' in the paper
        self.context_dim = np.clip(make_divisible(cnf.expanded_channels // context_ratio, 8),
                                   make_divisible(min_context_size * cnf.width_mult, 8),
                                   make_divisible(max_context_size * cnf.width_mult, 8)
                                   )

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            if no_dyconv:
                self.exp_conv = DynamicWrapper(
                    nn.Conv2d(
                        cnf.input_channels,
                        cnf.expanded_channels,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        dilation=(1, 1),
                        padding=0,
                        bias=False
                    )
                )
            else:
                self.exp_conv = DynamicConv(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    self.context_dim,
                    kernel_size=1,
                    k=dyconv_k,
                    temp_schedule=temp_schedule,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=False
                )

            self.exp_norm = norm_layer(cnf.expanded_channels)
            self.exp_act = DynamicWrapper(activation_layer(inplace=True))
        else:
            self.exp_conv = DynamicWrapper(nn.Identity())
            self.exp_norm = nn.Identity()
            self.exp_act = DynamicWrapper(nn.Identity())

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        padding = (cnf.kernel - 1) // 2 * cnf.dilation
        if no_dyconv:
            self.depth_conv = DynamicWrapper(
                nn.Conv2d(
                    cnf.expanded_channels,
                    cnf.expanded_channels,
                    kernel_size=(cnf.kernel, cnf.kernel),
                    groups=cnf.expanded_channels,
                    stride=(stride, stride),
                    dilation=(cnf.dilation, cnf.dilation),
                    padding=padding,
                    bias=False
                )
            )
        else:
            self.depth_conv = DynamicConv(
                cnf.expanded_channels,
                cnf.expanded_channels,
                self.context_dim,
                kernel_size=cnf.kernel,
                k=dyconv_k,
                temp_schedule=temp_schedule,
                groups=cnf.expanded_channels,
                stride=stride,
                dilation=cnf.dilation,
                padding=padding,
                bias=False
            )
        self.depth_norm = norm_layer(cnf.expanded_channels)
        self.depth_act = DynamicWrapper(activation_layer(inplace=True)) if no_dyrelu \
            else DyReLUB(cnf.expanded_channels, self.context_dim, M=dyrelu_k)

        self.ca = DynamicWrapper(nn.Identity()) if no_ca else CoordAtt()

        # project
        if no_dyconv:
            self.proj_conv = DynamicWrapper(
                nn.Conv2d(
                    cnf.expanded_channels,
                    cnf.out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    dilation=(1, 1),
                    padding=0,
                    bias=False
                )
            )
        else:
            self.proj_conv = DynamicConv(
                cnf.expanded_channels,
                cnf.out_channels,
                self.context_dim,
                kernel_size=1,
                k=dyconv_k,
                temp_schedule=temp_schedule,
                stride=1,
                dilation=1,
                padding=0,
                bias=False,
            )

        self.proj_norm = norm_layer(cnf.out_channels)

        context_norm_layer = norm_layer
        self.context_gen = ContextGen(self.context_dim, cnf.input_channels, cnf.expanded_channels,
                                      norm_layer=context_norm_layer, stride=stride)

    def forward(self, x, g=None):
        # x: CNN feature map (C x F x T)
        inp = x

        g = self.context_gen(x, g)
        x = self.exp_conv(x, g)
        x = self.exp_norm(x)
        x = self.exp_act(x, g)

        x = self.depth_conv(x, g)
        x = self.depth_norm(x)
        x = self.depth_act(x, g)
        x = self.ca(x, g)

        x = self.proj_conv(x, g)
        x = self.proj_norm(x)

        if self.use_res_connect:
            x += inp
        return x

#endregion

#region MobileNet - Utils

def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def collapse_dim(x: Tensor, dim: int, mode: str = "pool", pool_fn:  Callable[[Tensor, int], Tensor] = torch.mean,
                 combine_dim: int = None):
    """
    Collapses dimension of multi-dimensional tensor by pooling or combining dimensions
    :param x: input Tensor
    :param dim: dimension to collapse
    :param mode: 'pool' or 'combine'
    :param pool_fn: function to be applied in case of pooling
    :param combine_dim: dimension to join 'dim' to
    :return: collapsed tensor
    """
    if mode == "pool":
        return pool_fn(x, dim)
    elif mode == "combine":
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)


class CollapseDim(nn.Module):
    def __init__(self, dim: int, mode: str = "pool", pool_fn:  Callable[[Tensor, int], Tensor] = torch.mean,
                 combine_dim: int = None):
        super(CollapseDim, self).__init__()
        self.dim = dim
        self.mode = mode
        self.pool_fn = pool_fn
        self.combine_dim = combine_dim

    def forward(self, x):
        return collapse_dim(x, dim=self.dim, mode=self.mode, pool_fn=self.pool_fn, combine_dim=self.combine_dim)

#endregion

#region MobileNet - Block-Types

class ConcurrentSEBlock(torch.nn.Module):
    def __init__(
        self,
        c_dim: int,
        f_dim: int,
        t_dim: int,
        se_cnf: Dict
    ) -> None:
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()
        for d in se_cnf['se_dims']:
            input_dim = dims[d-1]
            squeeze_dim = make_divisible(input_dim // se_cnf['se_r'], 8)
            self.conc_se_layers.append(SqueezeExcitation(input_dim, squeeze_dim, d))
        if se_cnf['se_agg'] == "max":
            self.agg_op = lambda x: torch.max(x, dim=0)[0]
        elif se_cnf['se_agg'] == "avg":
            self.agg_op = lambda x: torch.mean(x, dim=0)
        elif se_cnf['se_agg'] == "add":
            self.agg_op = lambda x: torch.sum(x, dim=0)
        elif se_cnf['se_agg'] == "min":
            self.agg_op = lambda x: torch.min(x, dim=0)[0]
        else:
            raise NotImplementedError(f"SE aggregation operation '{self.agg_op}' not implemented")

    def forward(self, input: Tensor) -> Tensor:
        # apply all concurrent se layers
        se_outs = []
        for se_layer in self.conc_se_layers:
            se_outs.append(se_layer(input))
        out = self.agg_op(torch.stack(se_outs, dim=0))
        return out


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507.
    Args:
        input_dim (int): Input dimension
        squeeze_dim (int): Size of Bottleneck
        activation (Callable): activation applied to bottleneck
        scale_activation (Callable): activation applied to the output
    """

    def __init__(
        self,
        input_dim: int,
        squeeze_dim: int,
        se_dim: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, squeeze_dim)
        self.fc2 = torch.nn.Linear(squeeze_dim, input_dim)
        assert se_dim in [1, 2, 3]
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = torch.mean(input, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = scale
        return self.scale_activation(scale).view(shape)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.f_dim = None
        self.t_dim = None

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_cnf: Dict,
        norm_layer: Callable[..., nn.Module],
        depthwise_norm_layer: Callable[..., nn.Module]
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=depthwise_norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se and se_cnf['se_dims'] is not None:
            layers.append(ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_cnf))

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, inp: Tensor) -> Tensor:
        result = self.block(inp)
        if self.use_res_connect:
            result += inp
        return result

#endregion

#region Dynamic MobileNet - Model

# points to github releases
model_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
# folder to store downloaded models to
model_dir = "resources"


pretrained_models = {
    # ImageNet pre-trained models
    "dymn04_im": urllib.parse.urljoin(model_url, "dymn04_im.pt"),
    "dymn10_im": urllib.parse.urljoin(model_url, "dymn10_im.pt"),
    "dymn20_im": urllib.parse.urljoin(model_url, "dymn20_im.pt"),

    # Models trained on AudioSet
    "dymn04_as": urllib.parse.urljoin(model_url, "dymn04_as.pt"),
    "dymn10_as": urllib.parse.urljoin(model_url, "dymn10_as.pt"),
    "dymn20_as": urllib.parse.urljoin(model_url, "dymn20_as_mAP_493.pt"),
    "dymn20_as(1)": urllib.parse.urljoin(model_url, "dymn20_as.pt"),
    "dymn20_as(2)": urllib.parse.urljoin(model_url, "dymn20_as_mAP_489.pt"),
    "dymn20_as(3)": urllib.parse.urljoin(model_url, "dymn20_as_mAP_490.pt"),
    "dymn04_replace_se_as": urllib.parse.urljoin(model_url, "dymn04_replace_se_as.pt"),
    "dymn10_replace_se_as": urllib.parse.urljoin(model_url, " dymn10_replace_se_as.pt"),
}


class DyMN(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[DynamicInvertedResidualConfig],
            last_channel: int,
            num_classes: int = 527,
            head_type: str = "mlp",
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.2,
            in_conv_kernel: int = 3,
            in_conv_stride: int = 2,
            in_channels: int = 1,
            context_ratio: int = 4,
            max_context_size: int = 128,
            min_context_size: int = 32,
            dyrelu_k=2,
            dyconv_k=4,
            no_dyrelu: bool = False,
            no_dyconv: bool = False,
            no_ca: bool = False,
            temp_schedule: tuple = (30, 1, 1, 0.05),
            **kwargs: Any,
    ) -> None:
        super(DyMN, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, DynamicInvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[DynamicInvertedResidualConfig]")

        if block is None:
            block = DY_Block

        norm_layer = \
            norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        self.layers = nn.ModuleList()

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.in_c = ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
        )

        for cnf in inverted_residual_setting:
            if cnf.use_dy_block:
                b = block(cnf,
                          context_ratio=context_ratio,
                          max_context_size=max_context_size,
                          min_context_size=min_context_size,
                          dyrelu_k=dyrelu_k,
                          dyconv_k=dyconv_k,
                          no_dyrelu=no_dyrelu,
                          no_dyconv=no_dyconv,
                          no_ca=no_ca,
                          temp_schedule=temp_schedule
                          )
            else:
                b = InvertedResidual(cnf, None, norm_layer, partial(nn.BatchNorm2d, eps=0.001, momentum=0.01))

            self.layers.append(b)

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.out_c = ConvNormActivation(
            lastconv_input_channels,
            lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish,
        )

        self.head_type = head_type
        if self.head_type == "fully_convolutional":
            self.classifier = nn.Sequential(
                nn.Conv2d(
                    lastconv_output_channels,
                    num_classes,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False),
                nn.BatchNorm2d(num_classes),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif self.head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
        else:
            raise NotImplementedError(f"Head '{self.head_type}' unknown. Must be one of: 'mlp', "
                                      f"'fully_convolutional', 'multihead_attention_pooling'")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _feature_forward(self, x: Tensor) -> (Tensor, Tensor):
        x = self.in_c(x)
        g = None
        for layer in self.layers:
            x = layer(x)
        x = self.out_c(x)
        return x

    def _clf_forward(self, x: Tensor):
        embed = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x).squeeze()
        if x.dim() == 1:
            # squeezed batch dimension
            x = x.unsqueeze(0)
        return x, embed

    def _forward_impl(self, x: Tensor) -> (Tensor, Tensor):
        x = self._feature_forward(x)
        x, embed = self._clf_forward(x)
        return x, embed

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)[0]

    def update_params(self, epoch):
        for module in self.modules():
            if isinstance(module, DynamicConv):
                module.update_params(epoch)


def _dymn_conf(
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        strides: Tuple[int] = (2, 2, 2, 2),
        use_dy_blocks: str = "all",
        **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(DynamicInvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(DynamicInvertedResidualConfig.adjust_channels, width_mult=width_mult)

    activations = ["RE", "RE", "RE", "RE", "RE", "RE", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS"]

    if use_dy_blocks == "all":
        # per default the dynamic blocks replace all conventional IR blocks
        use_dy_block = [True] * 15
    elif use_dy_blocks == "replace_se":
        use_dy_block = [False, False, False, True, True, True, False, False, False, False, True, True, True, True, True]
    else:
        raise NotImplementedError(f"Config use_dy_blocks={use_dy_blocks} not implemented.")

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, use_dy_block[0], activations[0], 1, 1),
        bneck_conf(16, 3, 64, 24, use_dy_block[1], activations[1], strides[0], 1),  # C1
        bneck_conf(24, 3, 72, 24, use_dy_block[2], activations[2], 1, 1),
        bneck_conf(24, 5, 72, 40, use_dy_block[3], activations[3], strides[1], 1),  # C2
        bneck_conf(40, 5, 120, 40, use_dy_block[4], activations[4], 1, 1),
        bneck_conf(40, 5, 120, 40, use_dy_block[5], activations[5], 1, 1),
        bneck_conf(40, 3, 240, 80, use_dy_block[6], activations[6], strides[2], 1),  # C3
        bneck_conf(80, 3, 200, 80, use_dy_block[7], activations[7], 1, 1),
        bneck_conf(80, 3, 184, 80, use_dy_block[8], activations[8], 1, 1),
        bneck_conf(80, 3, 184, 80, use_dy_block[9], activations[9], 1, 1),
        bneck_conf(80, 3, 480, 112, use_dy_block[10], activations[10], 1, 1),
        bneck_conf(112, 3, 672, 112, use_dy_block[11], activations[11], 1, 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, use_dy_block[12], activations[12], strides[3], dilation),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, use_dy_block[13],
                   activations[13], 1, dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, use_dy_block[14],
                   activations[14], 1, dilation),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


def _dymn(
        inverted_residual_setting: List[DynamicInvertedResidualConfig],
        last_channel: int,
        pretrained_name: str,
        **kwargs: Any,
):
    model = DyMN(inverted_residual_setting, last_channel, **kwargs)

    # load pre-trained model using specified name
    if pretrained_name:
        # download from GitHub or load cached state_dict from 'resources' folder
        model_url = pretrained_models.get(pretrained_name)
        state_dict = load_state_dict_from_url(model_url, model_dir=model_dir, map_location="cpu")
        cls_in_state_dict = state_dict['classifier.5.weight'].shape[0]
        cls_in_current_model = model.classifier[5].out_features
        if cls_in_state_dict != cls_in_current_model:
            print(f"=> The number of classes in the loaded state dict (={cls_in_state_dict}) and "
                  f"the current model (={cls_in_current_model}) is not the same. Dropping final fully-connected layer "
                  f"and loading weights in non-strict mode!")
            del state_dict['classifier.5.weight']
            del state_dict['classifier.5.bias']
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    return model


def dymn(pretrained_name: str = None, **kwargs: Any):
    inverted_residual_setting, last_channel = _dymn_conf(**kwargs)
    return _dymn(inverted_residual_setting, last_channel, pretrained_name, **kwargs)


def get_model(num_classes: int = 527,
              pretrained_name: str = None,
              width_mult: float = 1.0,
              strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
              # Context
              context_ratio: int = 4,
              max_context_size: int = 128,
              min_context_size: int = 32,
              # Dy-ReLU
              dyrelu_k: int = 2,
              no_dyrelu: bool = False,
              # Dy-Conv
              dyconv_k: int = 4,
              no_dyconv: bool = False,
              T_max: float = 30.0,
              T0_slope: float = 1.0,
              T1_slope: float = 0.02,
              T_min: float = 1,
              pretrain_final_temp: float = 1.0,
              # Coordinate Attention
              no_ca: bool = False,
              use_dy_blocks="all"):
    """
    Arguments to modify the instantiation of a DyMN

    Args:
        num_classes (int): Specifies number of classes to predict
        pretrained_name (str): Specifies name of pre-trained model to load
        width_mult (float): Scales width of network
        strides (Tuple): Strides that are set to '2' in original implementation;
            might be changed to modify the size of receptive field and the downsampling factor in
            time and frequency dimension
        context_ratio (int): fraction of expanded channel representation used as context size
        max_context_size (int): maximum size of context
        min_context_size (int): minimum size of context
        dyrelu_k (int): number of linear mappings
        no_dyrelu (bool): not use Dy-ReLU
        dyconv_k (int): number of kernels for dynamic convolution
        no_dyconv (bool): not use Dy-Conv
        T_max, T0_slope, T1_slope, T_min (float): hyperparameters to steer the temperature schedule for Dy-Conv
        pretrain_final_temp (float): if model is pre-trained, then final Dy-Conv temperature
                                     of pre-training stage should be used
        no_ca (bool): not use Coordinate Attention
        use_dy_blocks (str): use dynamic block at all positions per default, other option: "replace_se"
    """

    block = DY_Block
    if pretrained_name:
        # if model is pre-trained, set Dy-Conv temperature to 'pretrain_final_temp'
        # pretrained on ImageNet -> 30
        # pretrained on AudioSet -> 1
        T_max = pretrain_final_temp

    temp_schedule = (T_max, T_min, T0_slope, T1_slope)

    m = dymn(num_classes=num_classes,
             pretrained_name=pretrained_name,
             block=block,
             width_mult=width_mult,
             strides=strides,
             context_ratio=context_ratio,
             max_context_size=max_context_size,
             min_context_size=min_context_size,
             dyrelu_k=dyrelu_k,
             dyconv_k=dyconv_k,
             no_dyrelu=no_dyrelu,
             no_dyconv=no_dyconv,
             no_ca=no_ca,
             temp_schedule=temp_schedule,
             use_dy_blocks=use_dy_blocks
             )
    # print(m)
    return m
#endregion

def get_dynamic_mobilenet(model_name: str='dymn10_as', pretrain_final_temp: float=1.0, checkpoint: str=None, no_dyrelu=False, no_dyconv=False, no_ca=False) -> DyMN:
    width = NAME_TO_WIDTH(model_name)
    model = get_model(width_mult=width, pretrained_name=model_name, pretrain_final_temp=pretrain_final_temp, num_classes=50, no_dyrelu=no_dyrelu, no_dyconv=no_dyconv, no_ca=no_ca)

    if checkpoint is not None:
        print(f'=> Loading checkpoint {checkpoint}')
        state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

    return model
