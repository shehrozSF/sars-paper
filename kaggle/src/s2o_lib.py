"""s2o_lib - shared code for the SAR->optical comparison and improvement notebooks.

Both notebooks embed this file (via %%writefile) so that the data split, models, training loops and
metrics are byte-identical between them. Everything is pixel space (no VAE), 256x256, SAR as one channel.
"""
import os, re, json, math, time, random, hashlib, copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

SEASONS = ("spring", "summer", "fall", "winter")
S2_NAME = re.compile(r"(ROIs\d+)_(spring|summer|fall|winter)_s2_(\d+)_p(\d+)\.png$")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SessionClock:
    """Wall-clock budget for a whole notebook session (Kaggle stops sessions at 12 h)."""

    def __init__(self, hours):
        self.t0, self.total = time.time(), hours * 3600

    def left_min(self):
        return (self.total - (time.time() - self.t0)) / 60

    def __repr__(self):
        return f"{self.left_min():.0f} min left of {self.total / 3600:.1f} h"


# ============================================================================ data index and split
def build_index(root):
    """Read the per-season CSVs of shambac/augmented-sentinel-1-2. Keeps only files with SEN1-2 names
    (scene id recoverable); drops the extra img_* files and duplicate rows."""
    rows = []
    for season in SEASONS:
        for f in sorted((Path(root) / season).glob("*.csv")):
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            for s1, s2 in zip(df["s1_fileName"].astype(str), df["s2_fileName"].astype(str)):
                s1, s2 = s1.replace("\\", "/"), s2.replace("\\", "/")
                s1 = re.sub(r"(\w+?)1s2_", r"\1_s1_", s1)          # filename typo in the CSVs
                m = S2_NAME.search(s2)
                rows.append({"season": season, "s1": s1, "s2": s2,
                             "scene": f"{m[1]}_{m[2]}_{int(m[3])}" if m else None})
    df = pd.DataFrame(rows)
    n_all = len(df)
    df = df[df.scene.notna()].drop_duplicates("s2").reset_index(drop=True)
    info = {"rows_in_csvs": n_all, "kept": len(df), "dropped_non_sen12_or_duplicate": n_all - len(df),
            "scenes": int(df.scene.nunique())}
    return df, info


def scene_split(df, fracs=(0.8, 0.1, 0.1), seed=2026):
    """Scene-disjoint split, per season, ordered by a seeded MD5 of the scene id (platform independent).
    Scenes are assigned to train/val/test by cumulative patch count."""
    assign = {}
    for season in SEASONS:
        sizes = df[df.season == season].groupby("scene").size()
        order = sorted(sizes.index, key=lambda s: hashlib.md5(f"{seed}:{s}".encode()).hexdigest())
        total, cum = sizes.sum(), 0
        for s in order:
            f = cum / total
            assign[s] = "train" if f < fracs[0] else ("val" if f < fracs[0] + fracs[1] else "test")
            cum += sizes[s]
    df = df.assign(split=df.scene.map(assign))
    ids = {k: sorted(df[df.split == k].s2) for k in ("train", "val", "test")}
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    return df, digest


def audit_duplicates(df, root, workers=32):
    """Exact-duplicate optical files across splits (MD5 of file bytes). Returns a report and the val/test
    rows whose optical image also appears in train, which must be excluded from evaluation."""
    from concurrent.futures import ThreadPoolExecutor

    def h(p):
        with open(os.path.join(root, p), "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        hashes = list(ex.map(h, df.s2))
    d = df.assign(md5=hashes)
    train_h = set(d[d.split == "train"].md5)
    leak = d[(d.split != "train") & d.md5.isin(train_h)]
    within = d.duplicated("md5", keep=False).sum()
    rep = {"files_hashed": len(d), "seconds": round(time.time() - t0), "duplicate_files_total": int(within),
           "val_or_test_copies_of_train": int(len(leak)),
           "by_split": leak.split.value_counts().to_dict()}
    return rep, set(leak.s2)


def save_split(path, df, digest, info, audit=None, excluded=()):
    out = {"version": "scene_v1", "digest": digest, "index": info, "duplicate_audit": audit,
           "excluded_from_eval": sorted(excluded),
           "counts": df.split.value_counts().to_dict(),
           "scenes": {k: sorted(df[df.split == k].scene.unique()) for k in ("train", "val", "test")}}
    json.dump(out, open(path, "w"), indent=1)
    return out


# ============================================================================ dataset
class PairDS(Dataset):
    """SAR (1 x H x W) and optical RGB (3 x H x W), both in [-1, 1]; paired flips / 90-degree rotations."""

    def __init__(self, df, root, augment=False, size=256):
        self.s1, self.s2 = df.s1.tolist(), df.s2.tolist()
        self.season = [SEASONS.index(s) for s in df.season]
        self.scene = df.scene.tolist()
        self.root, self.augment, self.size = root, augment, size

    def __len__(self):
        return len(self.s2)

    def _load(self, rel, mode):
        im = Image.open(os.path.join(self.root, rel)).convert(mode)
        if im.size != (self.size, self.size):
            im = im.resize((self.size, self.size), Image.BILINEAR)
        a = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0)
        return a[None] if mode == "L" else a.permute(2, 0, 1)

    def __getitem__(self, i):
        sar, opt = self._load(self.s1[i], "L"), self._load(self.s2[i], "RGB")
        if self.augment:
            k = random.randint(0, 3)
            sar, opt = torch.rot90(sar, k, (1, 2)), torch.rot90(opt, k, (1, 2))
            if random.random() < 0.5:
                sar, opt = sar.flip(2), opt.flip(2)
        return {"sar": sar, "opt": opt, "season": self.season[i], "idx": i}


def make_loader(ds, batch, shuffle, workers, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True,
                      drop_last=drop_last, persistent_workers=workers > 0)


def forever(loader):
    while True:
        for b in loader:
            yield b


# ============================================================================ network building blocks
def gn(c):
    return nn.GroupNorm(32 if c % 32 == 0 else 8, c)


def zero_(m):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb=0, p=0.0):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(emb, 2 * cout) if emb else None
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.p = p

    def forward(self, x, e=None):
        h = self.n2(self.c1(F.silu(self.n1(x))))
        if self.film is not None and e is not None:
            s, b = self.film(e)[:, :, None, None].chunk(2, 1)
            h = h * (1 + s) + b
        h = self.c2(F.dropout(F.silu(h), self.p, self.training))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n, self.qkv, self.o, self.h = gn(c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1), heads

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).unbind(1)
        y = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.o(y.transpose(-1, -2).reshape(B, C, H, W))


def timestep_emb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class UNet(nn.Module):
    """Conditional U-Net used by every method. Optional timestep embedding (diffusion), season embedding
    (FiLM), and one refinement block inserted in the decoder at `refine_level` (0 = full resolution)."""

    def __init__(self, in_ch, out_ch, ch=64, mult=(1, 2, 4, 4), nres=2, time=False, n_cond=0,
                 attn=True, p=0.0, refine=None, refine_level=2):
        super().__init__()
        emb = ch * 4 if (time or n_cond) else 0
        self.ch, self.use_time = ch, time
        if time:
            self.t_mlp = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.c_emb = nn.Embedding(n_cond, emb) if n_cond else None
        self.inc = nn.Conv2d(in_ch, ch, 3, padding=1)
        chans = [ch * m for m in mult]
        self.down, self.ds = nn.ModuleList(), nn.ModuleList()
        c = ch
        for i, co in enumerate(chans):
            self.down.append(nn.ModuleList([ResBlock(c if j == 0 else co, co, emb, p) for j in range(nres)]))
            c = co
            self.ds.append(nn.Conv2d(co, co, 3, 2, 1) if i < len(chans) - 1 else nn.Identity())
        self.mid1, self.mid2 = ResBlock(c, c, emb, p), ResBlock(c, c, emb, p)
        self.mid_attn = Attn(c) if attn else nn.Identity()
        self.up, self.us = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(chans))):
            co = chans[i]
            self.up.append(nn.ModuleList([ResBlock(c + chans[i] if j == 0 else co, co, emb, p) for j in range(nres)]))
            c = co
            self.us.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(co, co, 3, padding=1))
                           if i > 0 else nn.Identity())
        self.out = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        self.refine, self.refine_level = refine, refine_level

    def level_channels(self, level):
        return self.down[level][-1].c2.out_channels

    def forward(self, x, t=None, cond=None, aux=None):
        e = None
        if self.use_time:
            e = self.t_mlp(timestep_emb(t, self.ch))
        if self.c_emb is not None and cond is not None:
            e = self.c_emb(cond) if e is None else e + self.c_emb(cond)
        h, hs = self.inc(x), []
        for blocks, ds in zip(self.down, self.ds):
            for b in blocks:
                h = b(h, e)
            hs.append(h)
            h = ds(h)
        h = self.mid2(self.mid_attn(self.mid1(h, e)), e)
        for k, (blocks, us) in enumerate(zip(self.up, self.us)):
            level = len(hs) - 1 - k
            h = torch.cat([h, hs[level]], 1)
            for b in blocks:
                h = b(h, e)
            if self.refine is not None and level == self.refine_level:
                h = h + self.refine(h, aux)
            h = us(h)
        return self.out(h)


class PatchD(nn.Module):
    """70x70 PatchGAN discriminator (Isola et al. 2017)."""

    def __init__(self, in_ch, nf=64):
        super().__init__()
        L = [nn.Conv2d(in_ch, nf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        c = nf
        for co, s in ((nf * 2, 2), (nf * 4, 2), (nf * 8, 1)):
            L += [nn.Conv2d(c, co, 4, s, 1), nn.InstanceNorm2d(co, affine=True), nn.LeakyReLU(0.2, True)]
            c = co
        L += [nn.Conv2d(c, 1, 4, 1, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


# ============================================================================ refinement blocks (notebook 2)
# Every block maps the decoder feature h (B,C,H,W) to a residual with a zero-initialised output projection,
# so a variant equals the base network exactly at initialisation.
class ConvRefine(nn.Module):
    """Control: plain convolutional residual stack (capacity match for the recurrent blocks)."""

    def __init__(self, c, hid=128, depth=3):
        super().__init__()
        self.inp = nn.Conv2d(c, hid, 1)
        self.blocks = nn.ModuleList([ResBlock(hid, hid) for _ in range(depth)])
        self.out = zero_(nn.Conv2d(hid, c, 1))

    def forward(self, h, aux=None):
        x = self.inp(h)
        for b in self.blocks:
            x = b(x)
        return self.out(x)


class ConvLSTMRefine(nn.Module):
    """PReNet-style recurrent refinement (Ren et al. 2019): one ConvLSTM cell (Shi et al. 2015) applied K
    times to the same input feature, carrying hidden/cell state between iterations."""

    def __init__(self, c, hid=128, K=3):
        super().__init__()
        self.inp, self.K = nn.Conv2d(c, hid, 1), K
        self.gates = nn.Conv2d(2 * hid, 4 * hid, 3, padding=1)
        self.out = zero_(nn.Conv2d(hid, c, 1))

    def forward(self, h, aux=None):
        x = self.inp(h)
        hh, cc = torch.zeros_like(x), torch.zeros_like(x)
        for _ in range(self.K):
            i, f, o, g = self.gates(torch.cat([x, hh], 1)).chunk(4, 1)
            cc = torch.sigmoid(f) * cc + torch.sigmoid(i) * torch.tanh(g)
            hh = torch.sigmoid(o) * torch.tanh(cc)
        return self.out(hh)


def slstm_step(z_i, z_f, z_o, z_z, c, n, m):
    """One stabilised sLSTM step (Beck et al. 2024, xLSTM): exponential input gate, sigmoid forget gate
    in log space, normaliser n and stabiliser m. m=None means first step."""
    logf = F.logsigmoid(z_f)
    if m is None:
        m_new = z_i
        i, f = torch.ones_like(z_i), torch.zeros_like(z_i)
    else:
        m_new = torch.maximum(logf + m, z_i)
        i, f = torch.exp(z_i - m_new), torch.exp(logf + m - m_new)
    c = f * c + i * torch.tanh(z_z)
    n = f * n + i
    h = torch.sigmoid(z_o) * c / n.clamp_min(1e-6)
    return h, c, n, m_new


class ConvSLSTMRefine(nn.Module):
    """Convolutional sLSTM refinement (xLSTM family; cf. Vision-LSTM / xLSTM-UNet), K iterations."""

    def __init__(self, c, hid=128, K=3):
        super().__init__()
        self.inp, self.K = nn.Conv2d(c, hid, 1), K
        self.gates = nn.Conv2d(2 * hid, 4 * hid, 3, padding=1)
        self.out = zero_(nn.Conv2d(hid, c, 1))

    def forward(self, h, aux=None):
        with torch.autocast(h.device.type, enabled=False):
            x = self.inp(h.float())
            hh, c, n, m = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), None
            for _ in range(self.K):
                zi, zf, zo, zz = self.gates(torch.cat([x, hh], 1)).chunk(4, 1)
                hh, c, n, m = slstm_step(zi, zf, zo, zz, c, n, m)
            return self.out(hh).to(h.dtype)


class SelectiveSSM(nn.Module):
    """Mamba-1 style selective state-space layer (Gu & Dao 2023), pure-PyTorch sequential reference scan.
    Input/output (B, L, d). Intended for short sequences (L <= 32)."""

    def __init__(self, d, d_state=16, d_conv=4, expand=2):
        super().__init__()
        di, self.N, self.r = expand * d, d_state, max(1, math.ceil(d / 16))
        self.in_proj = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.r + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.r, di)
        dt = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))     # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1)))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)

    def forward(self, x):
        B, L, _ = x.shape
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))      # causal depthwise conv
        dt, Bm, Cm = self.x_proj(u).split([self.r, self.N, self.N], -1)
        dt = F.softplus(self.dt_proj(dt))                                      # (B,L,di)
        A = -torch.exp(self.A_log)                                             # (di,N)
        dA = torch.exp(dt.unsqueeze(-1) * A)                                   # (B,L,di,N)
        dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)             # (B,L,di,N)
        h, ys = torch.zeros(B, u.shape[-1], self.N, device=x.device, dtype=x.dtype), []
        for t in range(L):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = (torch.stack(ys, 1) + u * self.D) * F.silu(z)
        return self.out_proj(y)


class MambaRefine(nn.Module):
    """Global context via selective scans over a pooled feature map (cf. MambaIR / ColorMamba): four axial
    directions (left/right/up/down) over a pool x pool grid, upsampled and added back."""

    def __init__(self, c, d=128, pool=16):
        super().__init__()
        self.inp, self.pool = nn.Conv2d(c, d, 1), pool
        self.norm, self.ssm = nn.LayerNorm(d), SelectiveSSM(d)
        self.out = zero_(nn.Conv2d(d, c, 1))

    def _bi(self, s):
        s = self.norm(s)
        return self.ssm(s) + self.ssm(s.flip(1)).flip(1)

    def forward(self, h, aux=None):
        with torch.autocast(h.device.type, enabled=False):
            x = self.inp(h.float())
            H, W = x.shape[-2:]
            p = F.adaptive_avg_pool2d(x, self.pool)
            B, d, P, Q = p.shape
            g = p.permute(0, 2, 3, 1)                                          # B,P,Q,d
            rows = self._bi(g.reshape(B * P, Q, d)).reshape(B, P, Q, d)
            cols = self._bi(g.transpose(1, 2).reshape(B * Q, P, d)).reshape(B, Q, P, d).transpose(1, 2)
            y = F.interpolate((rows + cols).permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False)
            return self.out(y).to(h.dtype)


class SARSpatialRefine(nn.Module):
    """SPADE-style spatially adaptive modulation (Park et al. 2019): per-pixel scale/shift predicted from the
    SAR image at this resolution, so normalisation cannot wash out the SAR layout."""

    def __init__(self, c, hid=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, hid, 3, padding=1), nn.SiLU(), nn.Conv2d(hid, hid, 3, padding=1), nn.SiLU())
        self.g, self.b = nn.Conv2d(hid, c, 3, padding=1), nn.Conv2d(hid, c, 3, padding=1)
        self.n = nn.GroupNorm(32 if c % 32 == 0 else 8, c, affine=False)
        self.out = zero_(nn.Conv2d(c, c, 1))

    def forward(self, h, aux):
        a = self.enc(F.interpolate(aux, size=h.shape[-2:], mode="area"))
        return self.out(F.silu(self.n(h) * (1 + self.g(a)) + self.b(a)))


REFINE_BLOCKS = {"base": None, "conv": ConvRefine, "convlstm": ConvLSTMRefine, "convxlstm": ConvSLSTMRefine,
                 "mamba": MambaRefine, "spade": SARSpatialRefine}


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


# ============================================================================ EMA, LR schedule, time budget
class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval().requires_grad_(False)
        self.d = decay

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - self.d)
        for e, b in zip(self.m.buffers(), model.buffers()):
            e.copy_(b)


class TimedRun:
    """Step iterator bounded by wall-clock minutes and max_steps. The cosine LR factor uses a running
    projection of how many steps fit in the budget, re-estimated every 200 steps."""

    def __init__(self, minutes, max_steps, warmup=500, floor=0.05):
        self.sec, self.max_steps, self.warmup, self.floor = minutes * 60, max_steps, warmup, floor
        self.t0, self.total, self.step = None, max_steps, 0

    def __iter__(self):
        self.t0 = time.time()
        while self.step < min(self.total, self.max_steps):
            el = time.time() - self.t0
            if el > self.sec:
                break
            if self.step >= 50 and self.step % 200 == 0:
                self.total = min(self.max_steps, int(self.step * self.sec / max(el, 1e-6)))
            yield self.step
            self.step += 1

    def lr(self):
        s = self.step
        if s < self.warmup:
            return (s + 1) / self.warmup
        prog = min(1.0, (s - self.warmup) / max(1, self.total - self.warmup))
        return self.floor + (1 - self.floor) * 0.5 * (1 + math.cos(math.pi * prog))

    def elapsed(self):
        return time.time() - self.t0


def set_lr(opt, base_lrs, factor):
    for g, lr in zip(opt.param_groups, base_lrs):
        g["lr"] = lr * factor


def ssim_loss(p, g):
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    return 1 - ssim(p.float(), g.float(), data_range=2.0)


# ============================================================================ diffusion
def cosine_abar(T):
    s = 0.008
    t = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    return (f[1:] / f[0]).clamp(1e-5, 0.9999).float()


@torch.no_grad()
def ddim_sample(model, sar, abar, steps=50, gen=None):
    """Deterministic DDIM (eta=0) for an epsilon-prediction model conditioned by concatenation."""
    T = len(abar)
    ts = torch.linspace(T - 1, 0, steps, device=sar.device).round().long()
    x = torch.randn((sar.shape[0], 3, *sar.shape[-2:]), generator=gen, device=sar.device)
    for i, t in enumerate(ts):
        a = abar[t]
        eps = model(torch.cat([x, sar], 1), t.repeat(sar.shape[0])).float()
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
        a_prev = abar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=sar.device)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x0


def resshift_eta(T=15, e1=0.04, eT=0.99):
    i = torch.arange(T, dtype=torch.float32)
    return e1 * (eT / e1) ** (i / (T - 1))


@torch.no_grad()
def resshift_sample(model, sar, eta, kappa, gen=None):
    """ResShift (Yue et al. 2023) reverse chain, x0-prediction; starts from the SAR image plus noise."""
    T, y = len(eta), sar.repeat(1, 3, 1, 1)
    x = y + kappa * eta[-1].sqrt() * torch.randn(y.shape, generator=gen, device=y.device)
    for t in reversed(range(T)):
        tt = torch.full((y.shape[0],), t * 1000.0 / T, device=y.device)
        x0 = model(torch.cat([x, sar], 1), tt).float().clamp(-1, 1)
        if t > 0:
            e = eta[t - 1]
            x = x0 + e * (y - x0) + kappa * e.sqrt() * torch.randn(y.shape, generator=gen, device=y.device)
        else:
            x = x0
    return x


# ============================================================================ metrics
def _srgb_to_lab(img01):
    c = torch.where(img01 <= 0.04045, img01 / 12.92, ((img01 + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],
                     device=img01.device, dtype=img01.dtype)
    xyz = torch.einsum("ij,bjhw->bihw", M, c) / torch.tensor([0.95047, 1.0, 1.08883], device=img01.device,
                                                             dtype=img01.dtype)[None, :, None, None]
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-8) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _gblur(x, sigma):
    ks = int(2 * round(3 * sigma) + 1)
    r = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
    k = torch.exp(-r ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    C, p = x.shape[1], ks // 2
    mode = "reflect" if p < min(x.shape[-2:]) else "replicate"     # reflect needs pad < size (tiny test images)
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode=mode), k.view(1, 1, 1, ks).expand(C, 1, 1, ks), groups=C)
    return F.conv2d(F.pad(x, (0, 0, p, p), mode=mode), k.view(1, 1, ks, 1).expand(C, 1, ks, 1), groups=C)


def _grad_mag(x):
    g = x.mean(1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 8
    gx, gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx), F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx.transpose(2, 3))
    return (gx ** 2 + gy ** 2 + 1e-12).sqrt()


@torch.no_grad()
def image_metrics(p, g, lpips_fn=None):
    """Per-image metrics for p, g in [-1, 1], shape (B,3,H,W). PSNR on the [0,255] scale (peak-to-peak 2 in
    [-1,1]). SAM in degrees, deltaE = CIE76 in Lab, LF-RMSE = RMSE of Gaussian(sigma=4) low-pass in 8-bit units
    (coarse colour), GradErr = mean |Sobel magnitude difference| (edge/structure fidelity)."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    p, g = p.float().clamp(-1, 1), g.float()
    mse = ((p - g) ** 2).flatten(1).mean(1)
    out = {"psnr": 10 * torch.log10(4.0 / mse.clamp_min(1e-12)),
           "ssim": ssim(p, g, data_range=2.0, reduction="none").flatten()}
    p01, g01 = (p + 1) / 2, (g + 1) / 2
    a, b = p01.clamp(1e-4, 1), g01.clamp(1e-4, 1)
    cs = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).clamp(-1, 1)
    out["sam"] = torch.rad2deg(torch.acos(cs)).flatten(1).mean(1)
    x, y = p.flatten(2) - p.flatten(2).mean(2, keepdim=True), g.flatten(2) - g.flatten(2).mean(2, keepdim=True)
    out["cc"] = ((x * y).sum(2) / (x.norm(dim=2) * y.norm(dim=2) + 1e-8)).mean(1)
    out["deltaE"] = (_srgb_to_lab(p01) - _srgb_to_lab(g01)).norm(dim=1).flatten(1).mean(1)
    out["lf_rmse"] = 255 * ((_gblur(p01, 4.0) - _gblur(g01, 4.0)) ** 2).flatten(1).mean(1).sqrt()
    out["grad_err"] = 255 * (_grad_mag(p01) - _grad_mag(g01)).abs().flatten(1).mean(1)
    if lpips_fn is not None:
        out["lpips"] = lpips_fn(p, g).flatten().float()
    return out


def scene_bootstrap(values, scenes, n=2000, seed=0):
    """95% CI of the mean, resampling scenes (the independent unit) with replacement."""
    rng = np.random.default_rng(seed)
    v, s = np.asarray(values, float), np.asarray(scenes)
    uniq = np.unique(s)
    groups = [v[s == u] for u in uniq]
    sums, cnts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    idx = rng.integers(0, len(uniq), (n, len(uniq)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(uniq)


@torch.no_grad()
def evaluate(predict, ds, batch, device, lpips_fn=None, fid=True, desc=""):
    """Runs predict(sar, season) over ds, returns a per-image DataFrame and the FID (or nan)."""
    rows, fidm = [], None
    if fid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fidm = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    t0 = time.time()
    for b in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4):
        sar, opt = b["sar"].to(device), b["opt"].to(device)
        pred = predict(sar, b["season"].to(device)).float().clamp(-1, 1)
        m = image_metrics(pred, opt, lpips_fn)
        for j, i in enumerate(b["idx"].tolist()):
            rows.append({"s2": ds.s2[i], "scene": ds.scene[i], "season": SEASONS[ds.season[i]],
                         **{k: float(v[j]) for k, v in m.items()}})
        if fidm is not None:
            fidm.update((opt + 1) / 2, real=True)
            fidm.update((pred + 1) / 2, real=False)
    df = pd.DataFrame(rows)
    f = float(fidm.compute()) if fidm is not None else float("nan")
    log(f"  eval {desc}: n={len(df)} PSNR {df.psnr.mean():.3f} SSIM {df.ssim.mean():.4f} FID {f:.1f} "
        f"({(time.time() - t0) / 60:.1f} min)")
    return df, f


def summarise(df, fid, name, extra=None):
    lo, hi, n_sc = scene_bootstrap(df.psnr, df.scene)
    slo, shi, _ = scene_bootstrap(df.ssim, df.scene)
    r = {"model": name, "n": len(df), "scenes": n_sc, "PSNR": df.psnr.mean(), "PSNR_lo": lo, "PSNR_hi": hi,
         "SSIM": df.ssim.mean(), "SSIM_lo": slo, "SSIM_hi": shi, "FID": fid}
    for k in ("lpips", "sam", "cc", "deltaE", "lf_rmse", "grad_err"):
        if k in df:
            r[k.upper() if k in ("lpips", "sam", "cc") else k] = df[k].mean()
    r.update(extra or {})
    return r


def paired_delta(df_a, df_b, metric="psnr"):
    """Mean of (b - a) on shared images with a scene-bootstrap CI."""
    m = df_a[["s2", "scene", metric]].merge(df_b[["s2", metric]], on="s2", suffixes=("_a", "_b"))
    d = m[f"{metric}_b"] - m[f"{metric}_a"]
    lo, hi, _ = scene_bootstrap(d, m.scene)
    return float(d.mean()), lo, hi


def save_ckpt(path, model, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.detach().half().cpu() if v.is_floating_point() else v.cpu() for k, v in model.state_dict().items()}, path)
    json.dump(meta, open(path + ".json", "w"), indent=1, default=str)


def load_ckpt(path, model):
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=False)
    return missing, unexpected


# ============================================================================ training procedures
def _amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def quick_psnr(predict, ds, device, batch=32):
    vals = []
    for b in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2):
        p = predict(b["sar"].to(device), b["season"].to(device)).float().clamp(-1, 1)
        mse = ((p - b["opt"].to(device)) ** 2).flatten(1).mean(1)
        vals += (10 * torch.log10(4.0 / mse.clamp_min(1e-12))).tolist()
    return float(np.mean(vals))


def _batch(it, device):
    b = next(it)
    return b["sar"].to(device, non_blocking=True), b["opt"].to(device, non_blocking=True), b["season"].to(device)


def _finish(meta, run, device, t_steps):
    meta.update(steps=run.step, minutes=round(run.elapsed() / 60, 1),
                sec_per_step=round(run.elapsed() / max(1, run.step), 4),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if device.type == "cuda" else None)
    return meta


def train_regressor(name, model, loader, val_ds, device, minutes, max_steps, out_dir, lr=2e-4, w_ssim=0.15,
                    use_season=True, select="best", val_every=1000, warmup=500, param_groups=None, ema_decay=0.999):
    """Deterministic SAR->RGB regressor: L1 + w_ssim * (1 - SSIM). Selection: best validation PSNR of the EMA
    weights (select='best') or the final EMA weights (select='last')."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.to(device).train()
    opt = torch.optim.AdamW(param_groups or model.parameters(), lr=lr, weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]
    ema, run, it = EMA(model, ema_decay), TimedRun(minutes, max_steps, warmup), forever(loader)
    pred = lambda m: (lambda sar, season: m(sar, cond=season if use_season else None, aux=sar))
    hist, best, best_sd = [], -1e9, None

    def validate(step):
        nonlocal best, best_sd
        v = quick_psnr(pred(ema.m), val_ds, device)
        hist.append((step, round(v, 4)))
        if v > best:
            best, best_sd = v, {k: t.detach().clone().cpu() for k, t in ema.m.state_dict().items()}
        log(f"  [{name}] step {step} val PSNR {v:.3f} (best {best:.3f}) lr x{run.lr():.2f}")

    for step in run:
        sar, y, season = _batch(it, device)
        set_lr(opt, base, run.lr())
        with _amp(device):
            out = model(sar, cond=season if use_season else None, aux=sar)
        loss = F.l1_loss(out.float(), y) + (w_ssim * ssim_loss(out, y) if w_ssim else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non-finite loss at step {step}")
        if (step + 1) % val_every == 0:
            validate(step + 1)
    validate(run.step)
    if select == "best" and best_sd is not None:
        ema.m.load_state_dict(best_sd)
    meta = _finish({"name": name, "selection": select, "best_val_psnr": best, "val_hist": hist,
                    "params_M": round(n_params(model), 2)}, run, device, None)
    save_ckpt(os.path.join(out_dir, f"{name}.pt"), ema.m, meta)
    return pred(ema.m), meta


class ImagePool:
    def __init__(self, size=50):
        self.size, self.items = size, []

    def __call__(self, x):
        out = []
        for im in x.detach():
            im = im[None]
            if len(self.items) < self.size:
                self.items.append(im); out.append(im)
            elif random.random() < 0.5:
                j = random.randrange(self.size); out.append(self.items[j].clone()); self.items[j] = im
            else:
                out.append(im)
        return torch.cat(out)


def train_pix2pix(name, G, loader, val_ds, device, minutes, max_steps, out_dir, lam_l1=100.0, lr=2e-4, val_every=1000):
    """Pix2Pix (Isola et al. 2017): U-Net generator + conditional 70x70 PatchGAN, LSGAN loss + lambda*L1.
    Selection: final EMA generator (no validation-PSNR selection for generative models)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    D = PatchD(4).to(device).train()
    G.to(device).train()
    oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    ema, run, it, hist = EMA(G), TimedRun(minutes, max_steps, warmup=200), forever(loader), []
    pred = lambda m: (lambda sar, season: m(sar))
    for step in run:
        sar, y, _ = _batch(it, device)
        f = run.lr(); set_lr(oG, [lr], f); set_lr(oD, [lr], f)
        with _amp(device):
            fake = G(sar)
            d_real, d_fake = D(torch.cat([sar, y], 1)).float(), D(torch.cat([sar, fake.detach()], 1)).float()
        lD = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
        oD.zero_grad(set_to_none=True); lD.backward(); oD.step()
        with _amp(device):
            d_fake = D(torch.cat([sar, fake], 1)).float()
        lG = F.mse_loss(d_fake, torch.ones_like(d_fake)) + lam_l1 * F.l1_loss(fake.float(), y)
        oG.zero_grad(set_to_none=True); lG.backward(); oG.step(); ema.update(G)
        if (step + 1) % val_every == 0:
            v = quick_psnr(pred(ema.m), val_ds, device); hist.append((step + 1, round(v, 4)))
            log(f"  [{name}] step {step + 1} lossD {lD.item():.3f} lossG {lG.item():.3f} val PSNR {v:.3f}")
    meta = _finish({"name": name, "selection": "last", "val_hist": hist, "params_M": round(n_params(G), 2),
                    "disc_params_M": round(n_params(D), 2)}, run, device, None)
    save_ckpt(os.path.join(out_dir, f"{name}.pt"), ema.m, meta)
    return pred(ema.m), meta


def train_cyclegan(name, GA, GB, loader, val_ds, device, minutes, max_steps, out_dir, lam_cyc=10.0, lr=2e-4, val_every=1000):
    """CycleGAN (Zhu et al. 2017), unpaired: the optical target of each step comes from the previous batch,
    so no SAR/optical pair is seen together. GA: SAR->RGB, GB: RGB->SAR, LSGAN + cycle L1, image pools.
    No identity loss (SAR has 1 channel, RGB 3). Selection: final EMA of GA."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    DA, DB = PatchD(3).to(device).train(), PatchD(1).to(device).train()
    GA.to(device).train(); GB.to(device).train()
    oG = torch.optim.Adam(list(GA.parameters()) + list(GB.parameters()), lr=lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(list(DA.parameters()) + list(DB.parameters()), lr=lr, betas=(0.5, 0.999))
    pA, pB = ImagePool(), ImagePool()
    ema, run, it, hist, prev_rgb = EMA(GA), TimedRun(minutes, max_steps, warmup=200), forever(loader), [], None
    pred = lambda m: (lambda sar, season: m(sar))
    mse1 = lambda x: F.mse_loss(x.float(), torch.ones_like(x.float()))
    mse0 = lambda x: F.mse_loss(x.float(), torch.zeros_like(x.float()))
    for step in run:
        sar, y, _ = _batch(it, device)
        if prev_rgb is None or prev_rgb.shape[0] != y.shape[0]:
            prev_rgb = y.roll(1, 0)
        rgb, prev_rgb = prev_rgb, y                                           # unpaired optical sample
        f = run.lr(); set_lr(oG, [lr], f); set_lr(oD, [lr], f)
        with _amp(device):
            fake_rgb, fake_sar = GA(sar), GB(rgb)
            rec_sar, rec_rgb = GB(fake_rgb), GA(fake_sar)
            lG = mse1(DA(fake_rgb)) + mse1(DB(fake_sar))
        lG = lG + lam_cyc * (F.l1_loss(rec_sar.float(), sar) + F.l1_loss(rec_rgb.float(), rgb))
        oG.zero_grad(set_to_none=True); lG.backward(); oG.step(); ema.update(GA)
        with _amp(device):
            lD = 0.5 * (mse1(DA(rgb)) + mse0(DA(pA(fake_rgb)))) + 0.5 * (mse1(DB(sar)) + mse0(DB(pB(fake_sar))))
        oD.zero_grad(set_to_none=True); lD.backward(); oD.step()
        if (step + 1) % val_every == 0:
            v = quick_psnr(pred(ema.m), val_ds, device); hist.append((step + 1, round(v, 4)))
            log(f"  [{name}] step {step + 1} lossD {lD.item():.3f} lossG {lG.item():.3f} val PSNR {v:.3f}")
    meta = _finish({"name": name, "selection": "last", "val_hist": hist, "params_M": round(n_params(GA) + n_params(GB), 2)},
                   run, device, None)
    save_ckpt(os.path.join(out_dir, f"{name}.pt"), ema.m, meta)
    return pred(ema.m), meta


def train_ddpm(name, model, loader, val_ds, device, minutes, max_steps, out_dir, T=1000, ddim_steps=50, lr=2e-4,
               val_every=2000, seed=0):
    """Conditional DDPM (Ho et al. 2020; SAR by channel concatenation), epsilon prediction, cosine schedule,
    deterministic DDIM sampling. Selection: final EMA."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.to(device).train()
    abar = cosine_abar(T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema, run, it, hist = EMA(model, 0.9995), TimedRun(minutes, max_steps, warmup=500), forever(loader), []

    def pred(m, steps):
        gen = torch.Generator(device=device).manual_seed(seed)
        return lambda sar, season: ddim_sample(m, sar, abar, steps, gen)

    for step in run:
        sar, y, _ = _batch(it, device)
        set_lr(opt, [lr], run.lr())
        t = torch.randint(0, T, (y.shape[0],), device=device)
        eps = torch.randn_like(y)
        a = abar[t].view(-1, 1, 1, 1)
        with _amp(device):
            out = model(torch.cat([a.sqrt() * y + (1 - a).sqrt() * eps, sar], 1), t)
        loss = F.mse_loss(out.float(), eps)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update(model)
        if (step + 1) % val_every == 0:
            sub = torch.utils.data.Subset(val_ds, range(min(64, len(val_ds))))
            v = quick_psnr(pred(ema.m, 20), sub, device); hist.append((step + 1, round(v, 4)))
            log(f"  [{name}] step {step + 1} loss {loss.item():.4f} val PSNR (64 img, DDIM20) {v:.3f}")
    meta = _finish({"name": name, "selection": "last", "val_hist": hist, "params_M": round(n_params(model), 2),
                    "T": T, "ddim_steps": ddim_steps}, run, device, None)
    save_ckpt(os.path.join(out_dir, f"{name}.pt"), ema.m, meta)
    return pred(ema.m, ddim_steps), meta


def train_resshift(name, model, loader, val_ds, device, minutes, max_steps, out_dir, T=15, kappa=1.0, lr=2e-4,
                   val_every=2000, seed=0):
    """ResShift-style residual-shifting bridge (Yue et al. 2023) in pixel space: x_t = x0 + eta_t (y - x0)
    + kappa sqrt(eta_t) eps with y = SAR replicated to 3 channels; x0 prediction; T=15 (NFE 15).
    kappa=1.0 (the latent-space paper used 2.0; pixel values span [-1,1]). Selection: final EMA."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.to(device).train()
    eta = resshift_eta(T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema, run, it, hist = EMA(model, 0.9995), TimedRun(minutes, max_steps, warmup=500), forever(loader), []

    def pred(m):
        gen = torch.Generator(device=device).manual_seed(seed)
        return lambda sar, season: resshift_sample(m, sar, eta, kappa, gen)

    for step in run:
        sar, y, _ = _batch(it, device)
        set_lr(opt, [lr], run.lr())
        t = torch.randint(0, T, (y.shape[0],), device=device)
        e = eta[t].view(-1, 1, 1, 1)
        ysar = sar.repeat(1, 3, 1, 1)
        xt = y + e * (ysar - y) + kappa * e.sqrt() * torch.randn_like(y)
        with _amp(device):
            out = model(torch.cat([xt, sar], 1), t.float() * 1000.0 / T)
        loss = F.mse_loss(out.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update(model)
        if (step + 1) % val_every == 0:
            sub = torch.utils.data.Subset(val_ds, range(min(64, len(val_ds))))
            v = quick_psnr(pred(ema.m), sub, device); hist.append((step + 1, round(v, 4)))
            log(f"  [{name}] step {step + 1} loss {loss.item():.4f} val PSNR (64 img) {v:.3f}")
    meta = _finish({"name": name, "selection": "last", "val_hist": hist, "params_M": round(n_params(model), 2),
                    "T": T, "kappa": kappa}, run, device, None)
    save_ckpt(os.path.join(out_dir, f"{name}.pt"), ema.m, meta)
    return pred(ema.m), meta


@torch.no_grad()
def train_mean_colour(ds, n=2000, seed=0):
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    m = torch.stack([ds[i]["opt"].mean((1, 2)) for i in idx]).mean(0)
    return m, (lambda sar, season: m.to(sar.device).view(1, 3, 1, 1).expand(sar.shape[0], 3, *sar.shape[-2:]))


def make_variant(kind, base_kwargs, level=2):
    """Base U-Net plus one refinement block (zero-initialised output) at decoder level `level`."""
    probe = UNet(**base_kwargs)
    c = probe.level_channels(level)
    blk = REFINE_BLOCKS[kind]
    return UNet(**base_kwargs, refine=None if blk is None else blk(c), refine_level=level)


@torch.no_grad()
def max_output_diff(a, b, sar, season):
    a.eval(); b.eval()
    return float((a(sar, cond=season, aux=sar).float() - b(sar, cond=season, aux=sar).float()).abs().max())
