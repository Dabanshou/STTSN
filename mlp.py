import torch
from torch import nn
from einops.layers.torch import Rearrange
# from SEAttention import SEAttention
# from SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN4D(nn.Module):
    def __init__(self, dim, epsilon=1e-5, normalize_dim=1):
        super().__init__()
        self.epsilon = epsilon
        self.dim = dim
        self.normalize_dim = normalize_dim

        self.alpha = nn.Parameter(torch.ones([1, self.dim, 1, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, self.dim, 1, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=self.normalize_dim, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=self.normalize_dim, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v24D(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.fc.weight, gain=1e-8)
    #     nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPblock(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial', pn=6):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)

        self.fc1 = Temporal_FC(pn)
        # self.fc2 = Temporal_FC(dim)
        # self.gelu = nn.GELU()

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN4D(dim)
                # self.norm0 = nn.LayerNorm(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()
        # self.norm2 = LN(dim)
        self.norm1 = LN4D(dim)
        self.reset_parameters()

        # self.norm2 = LN4D(66)
        # self.ln = nn.LayerNorm(50)
        # self.att = SimplifiedScaledDotProductAttention(d_model=300,h=6)
        # self.interpatchmix = MLPblock1(6, 18)
        # self.se = SEAttention(channel=seq, reduction=5)
        # self.mixer = MLPblock1(50,100)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc1.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc1.fc.bias, 0)
        # nn.init.xavier_uniform_(self.fc2.fc.weight, gain=1e-8)
        # nn.init.constant_(self.fc2.fc.bias, 0)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = x1.permute(0, 1, 3, 2)
        x1 = self.fc1(x1)
        x1 = x1.permute(0, 1, 3, 2)
        x2 = x + x1

        x_ = self.norm0(x2)
        x_ = self.fc0(x_)
        x3 = x2 + x_
        return x3

class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis, pn):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis, pn)
            for i in range(num_layers)])
        # self.mlps = MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
    def forward(self, x):
        x = self.mlps(x)
        return x

def build_mlps(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    return TransMLP(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
        pn = args.pn,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    if activation == 'silu':
        return nn.SiLU
    #if activation == 'swish':
    #    return nn.Hardswish
    if activation == 'softplus':
        return nn.Softplus
    if activation == 'tanh':
        return nn.Tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_norm_fn(norm):
    if norm == "batchnorm":
        return nn.BatchNorm1d
    if norm == "layernorm":
        return nn.LayerNorm
    if norm == 'instancenorm':
        return nn.InstanceNorm1d
    raise RuntimeError(F"norm should be batchnorm/layernorm, not {norm}.")


# class MLPblock_nopatch(nn.Module):
#
#     def __init__(self, dim, seq):
#         super().__init__()
#
#         self.fc0 = Temporal_FC(seq)
#         self.norm0 = LN(dim)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
#         nn.init.constant_(self.fc0.fc.bias, 0)
#
#     def forward(self, x):
#         x_ = self.norm0(x)
#         x_ = self.fc0(x_)
#         x = x + x_
#         return x
