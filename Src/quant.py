import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FP = 32

def round_pass(x):
    return (x.round() - x).detach() + x


def grad_scale(x, g):
    return (x - x * g).detach() + x * g


def qrange(bits, signed):
    if signed:
        return 2 ** (bits - 1), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


def fake_quant(v, s, qn, qp):
    return round_pass(torch.clamp(v / s, -qn, qp)) * s

@torch.no_grad()
def scale_minmax(v, bits, signed):
    qn, qp = qrange(bits, signed)
    return (v.abs().max() / qp).clamp_min(1e-8)


@torch.no_grad()
def scale_lsq(v, bits, signed):
    qn, qp = qrange(bits, signed)
    return (2 * v.abs().mean() / math.sqrt(qp)).clamp_min(1e-8)


@torch.no_grad()
def scale_mse(v, bits, signed, n_grid=100):
    qn, qp = qrange(bits, signed)
    v = v.float().flatten()
    if v.numel() > 200_000:
        v = v[torch.randint(0, v.numel(), (200_000,), device=v.device)]
    vmax = v.abs().max().clamp_min(1e-8)
    best, best_s = float("inf"), vmax / qp
    for a in torch.linspace(0.05, 1.0, n_grid):
        s = a * vmax / qp
        err = (torch.clamp((v / s).round(), -qn, qp) * s - v).pow(2).mean().item()
        if err < best:
            best, best_s = err, s
    return best_s


SCALE_INIT = {"minmax": scale_minmax, "lsq": scale_lsq, "mse": scale_mse}

class Quantizer(nn.Module):

    def __init__(self, bits, signed, is_weight):
        super().__init__()
        self.bits, self.signed, self.is_weight = bits, signed, is_weight
        self.s = nn.Parameter(torch.tensor(1.0))
        self.mode = "quant"
        self._obs = []

    @property
    def qn_qp(self):
        return qrange(self.bits, self.signed)

    def forward(self, v):
        if self.bits >= FP or self.mode == "off":
            return v
        if self.mode == "observe":
            flat = v.detach().float().flatten()
            k = min(flat.numel(), 20_000)
            self._obs.append(flat[torch.randint(0, flat.numel(), (k,), device=flat.device)].cpu())
            return v
        qn, qp = self.qn_qp
        n = v.numel() if self.is_weight else v[0].numel()
        s = grad_scale(self.s, 1.0 / math.sqrt(n * qp)) if self.training else self.s
        return fake_quant(v, s, qn, qp)

    @torch.no_grad()
    def init_scale(self, v, method):
        self.s.data.fill_(float(SCALE_INIT[method](v, self.bits, self.signed)))

    @torch.no_grad()
    def finish_observe(self, method):
        if self.bits < FP and self._obs:
            self.init_scale(torch.cat(self._obs), method)
        self._obs = []
        self.mode = "quant"

    def extra_repr(self):
        return f"bits={self.bits}, signed={self.signed}, s={self.s.item():.4g}"


class QConv2d(nn.Conv2d):
    @classmethod
    def from_module(cls, m, wbits, abits, a_signed):
        q = cls(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding,
                m.dilation, m.groups, bias=m.bias is not None, padding_mode=m.padding_mode)
        q.weight = m.weight
        if m.bias is not None:
            q.bias = m.bias
        q.wq = Quantizer(wbits, signed=True, is_weight=True)
        q.aq = Quantizer(abits, signed=a_signed, is_weight=False)
        return q

    def forward(self, x):
        return self._conv_forward(self.aq(x), self.wq(self.weight), self.bias)


class QLinear(nn.Linear):
    @classmethod
    def from_module(cls, m, wbits, abits, a_signed):
        q = cls(m.in_features, m.out_features, bias=m.bias is not None)
        q.weight = m.weight
        if m.bias is not None:
            q.bias = m.bias
        q.wq = Quantizer(wbits, signed=True, is_weight=True)
        q.aq = Quantizer(abits, signed=a_signed, is_weight=False)
        return q

    def forward(self, x):
        return F.linear(self.aq(x), self.wq(self.weight), self.bias)

@torch.no_grad()
def fold_bn(model):
    for parent in model.modules():
        names = list(parent._modules.keys())
        for a, b in zip(names[:-1], names[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                conv.weight.mul_(k.view(-1, 1, 1, 1))
                b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
                conv.bias = nn.Parameter(bn.bias + (b0 - bn.running_mean) * k)
                parent._modules[b] = nn.Identity()
    left = [n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    assert not left, f"unfolded BatchNorms remain: {left}"
    return model

def quantizable_layers(model):
    return [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def _set_module(model, name, new):
    parent = model
    *path, last = name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, last, new)

def quantize_model(model, wbits, abits=8, first_last_bits=8):
    names = quantizable_layers(model)
    first, last = names[0], names[-1]
    for n in names:
        m = model.get_submodule(n)
        edge = n in (first, last)
        all_fp = isinstance(wbits, int) and wbits >= FP
        if edge:
            wb = FP if all_fp else first_last_bits
        else:
            wb = wbits[n] if isinstance(wbits, dict) else wbits
        ab = first_last_bits if (edge and abits < FP) else abits
        cls = QConv2d if isinstance(m, nn.Conv2d) else QLinear
        _set_module(model, n, cls.from_module(m, wb, ab, a_signed=(n == first)))
    return model

def qlayers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (QConv2d, QLinear))]


@torch.no_grad()
def init_weight_scales(model, method="mse"):
    for _, m in qlayers(model):
        if m.wq.bits < FP:
            m.wq.init_scale(m.weight, method)


@torch.no_grad()
def calibrate_activations(model, loader, dev, method="mse", n_batches=4):
    model.eval()
    for _, m in qlayers(model):
        m.aq.mode = "observe"
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x.to(dev))
    for _, m in qlayers(model):
        m.aq.finish_observe(method)


def set_wbits(model, name, bits, method="mse"):
    m = model.get_submodule(name)
    m.wq.bits = bits
    if bits < FP:
        m.wq.init_scale(m.weight, method)


def layer_param_counts(model):
    return {n: m.weight.numel() for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))}


def size_mb(model):
    bits, counted = 0.0, set()
    for _, m in model.named_modules():
        if isinstance(m, (QConv2d, QLinear)):
            bits += min(m.wq.bits, FP) * m.weight.numel(); counted.add(id(m.weight))
    for p in model.parameters():
        if id(p) not in counted and p.numel() > 1:
            bits += 32 * p.numel()
    return bits / 8e6
