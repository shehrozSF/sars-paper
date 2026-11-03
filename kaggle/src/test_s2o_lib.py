"""CPU smoke test for s2o_lib on a tiny synthetic copy of the augmented SEN1-2 layout.
Run:  python test_s2o_lib.py   (needs torch, torchmetrics, numpy, pandas, pillow)"""
import os, sys, json, shutil, tempfile, math
import numpy as np, pandas as pd, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import s2o_lib as L

torch.set_num_threads(4)
IMG = 64
dev = torch.device("cpu")
root = tempfile.mkdtemp(prefix="s2o_smoke_")
rng = np.random.default_rng(0)


def make_dataset():
    for season, roi in zip(L.SEASONS, ("ROIs1158", "ROIs1868", "ROIs1970", "ROIs2017")):
        rows = []
        for scene in range(10):
            for p in range(1, 7):
                s1_disk = f"{season}/s1_{scene}/{roi}_{season}_s1_{scene}_p{p}.png"
                s2 = f"{season}/s2_{scene}/{roi}_{season}_s2_{scene}_p{p}.png"
                for rel, ch in ((s1_disk, 1), (s2, 3)):
                    os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                    a = rng.integers(0, 255, (IMG, IMG, ch), dtype=np.uint8)
                    Image.fromarray(a[..., 0] if ch == 1 else a).save(os.path.join(root, rel))
                rows.append({"s1_fileName": f"{season}\\s1_{scene}\\{roi}_{season}1s2_{scene}_p{p}.png",   # CSV typo + backslashes
                             "s2_fileName": s2.replace("/", "\\"), "season": season, "region": "Temperate"})
        rows.append({"s1_fileName": f"{season}/x/img_temperate_p1.png", "s2_fileName": f"{season}/x/img_temperate_p1.png",
                     "season": season, "region": "Temperate"})                                    # extra file: dropped
        rows.append(dict(rows[0]))                                                                # duplicate row: dropped
        pd.DataFrame(rows).to_csv(os.path.join(root, season, f"{season}.csv"), index=False)


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


make_dataset()
df, info = L.build_index(root)
check(info["kept"] == 4 * 10 * 6 and info["dropped_non_sen12_or_duplicate"] == 8, f"index {info}")
df, d1 = L.scene_split(df, (0.6, 0.2, 0.2))
_, d2 = L.scene_split(L.build_index(root)[0], (0.6, 0.2, 0.2))
check(d1 == d2, "split digest is deterministic")
sc = {k: set(df[df.split == k].scene) for k in ("train", "val", "test")}
check(not (sc["train"] & sc["val"]) and not (sc["train"] & sc["test"]) and not (sc["val"] & sc["test"]), "splits are scene-disjoint")
check(all(len(df[(df.split == k) & (df.season == s)]) > 0 for k in sc for s in L.SEASONS), "every split has every season")

# plant an exact duplicate: copy a train optical image over a test one
tr, te = df[df.split == "train"].iloc[0], df[df.split == "test"].iloc[0]
shutil.copy(os.path.join(root, tr.s2), os.path.join(root, te.s2))
rep, excl = L.audit_duplicates(df, root, workers=4)
check(rep["val_or_test_copies_of_train"] == 1 and te.s2 in excl, f"duplicate audit {rep}")
out = L.save_split(os.path.join(root, "split.json"), df, d1, info, rep, excl)
check(out["digest"] == d1, "split file written")

tr_ds = L.PairDS(df[df.split == "train"], root, augment=True, size=IMG)
va_ds = L.PairDS(df[df.split == "val"], root, size=IMG)
b = tr_ds[0]
check(b["sar"].shape == (1, IMG, IMG) and b["opt"].shape == (3, IMG, IMG) and b["sar"].abs().max() <= 1, "dataset shapes/range")
loader = L.make_loader(tr_ds, 4, True, 0, drop_last=True)
out_dir = os.path.join(root, "ckpt")
small = dict(ch=16, mult=(1, 2, 4, 4), nres=1)

# ---- every trainer runs a few steps and returns a working predictor
for name, fn in [
    ("unet", lambda: L.train_regressor("unet", L.UNet(1, 3, **small), loader, va_ds, dev, 1, 4, out_dir, val_every=2, warmup=2, use_season=False)),
    ("pix2pix", lambda: L.train_pix2pix("pix2pix", L.UNet(1, 3, **small), loader, va_ds, dev, 1, 3, out_dir, val_every=2)),
    ("cyclegan", lambda: L.train_cyclegan("cyclegan", L.UNet(1, 3, **small), L.UNet(3, 1, **small), loader, va_ds, dev, 1, 3, out_dir, val_every=2)),
    ("ddpm", lambda: L.train_ddpm("ddpm", L.UNet(4, 3, time=True, **small), loader, va_ds, dev, 1, 3, out_dir, ddim_steps=4, val_every=2)),
    ("resshift", lambda: L.train_resshift("resshift", L.UNet(4, 3, time=True, **small), loader, va_ds, dev, 1, 3, out_dir, val_every=2)),
]:
    pred, meta = fn()
    dfe, fid = L.evaluate(pred, va_ds, 8, dev, lpips_fn=None, fid=False, desc=name)
    check(len(dfe) == len(va_ds) and np.isfinite(dfe[["psnr", "ssim", "sam", "cc", "deltaE", "lf_rmse", "grad_err"]].values).all(),
          f"{name}: trains {meta['steps']} steps, eval finite (PSNR {dfe.psnr.mean():.2f})")
    check(os.path.exists(os.path.join(out_dir, f"{name}.pt")), f"{name}: checkpoint saved")

m, pconst = L.train_mean_colour(tr_ds, n=20)
dfc, _ = L.evaluate(pconst, va_ds, 8, dev, fid=False, desc="const")
r = L.summarise(dfc, float("nan"), "const")
check(r["PSNR_lo"] <= r["PSNR"] <= r["PSNR_hi"], f"summary + scene bootstrap {r['PSNR']:.2f} [{r['PSNR_lo']:.2f},{r['PSNR_hi']:.2f}]")
dm, lo, hi = L.paired_delta(dfc, dfe)
check(lo <= dm <= hi, "paired delta CI")

# ---- PSNR definition: identical images -> capped large, known MSE -> known PSNR
x = torch.zeros(1, 3, 32, 32); y = torch.full((1, 3, 32, 32), 0.2)
check(abs(L.image_metrics(x, y)["psnr"].item() - 10 * math.log10(4 / 0.04)) < 1e-4, "PSNR matches 10log10(4/MSE)")

# ---- sLSTM stabiliser gives the same output as the unstabilised recurrence (float64)
torch.manual_seed(0)
zs = [torch.randn(4, 3, 5, 5, dtype=torch.float64) * 3 for _ in range(12)]
h, c, n, mm = None, torch.zeros(4, 3, 5, 5, dtype=torch.float64), torch.zeros(4, 3, 5, 5, dtype=torch.float64), None
c2, n2 = c.clone(), n.clone()
for k in range(3):
    zi, zf, zo, zz = zs[4 * k: 4 * k + 4]
    h, c, n, mm = L.slstm_step(zi, zf, zo, zz, c, n, mm)
    i2, f2 = torch.exp(zi), torch.sigmoid(zf)
    c2, n2 = f2 * c2 + i2 * torch.tanh(zz), f2 * n2 + i2
    h2 = torch.sigmoid(zo) * c2 / n2
check(torch.allclose(h, h2, atol=1e-10), "sLSTM stabilised == unstabilised")
big = [torch.full((1, 1, 2, 2), v, dtype=torch.float64) for v in (80.0, 0.0, 0.0, 1.0)]
hb, *_ = L.slstm_step(*big, torch.zeros(1, 1, 2, 2, dtype=torch.float64), torch.zeros(1, 1, 2, 2, dtype=torch.float64), None)
hb, *_ = L.slstm_step(*big, *L.slstm_step(*big, torch.zeros(1, 1, 2, 2, dtype=torch.float64), torch.zeros(1, 1, 2, 2, dtype=torch.float64), None)[1:])
check(torch.isfinite(hb).all(), "sLSTM finite with extreme input gate (exp(80))")

# ---- selective SSM: gradcheck + causality
ssm = L.SelectiveSSM(8, d_state=4).double()
xs = torch.randn(2, 6, 8, dtype=torch.float64, requires_grad=True)
check(torch.autograd.gradcheck(lambda t: ssm(t).sum(), (xs,), eps=1e-6, atol=1e-5), "SelectiveSSM gradcheck")
y1 = ssm(xs.detach()); x2 = xs.detach().clone(); x2[:, 4:] += 1.0
check(torch.allclose(ssm(x2)[:, :4], y1[:, :4]), "SelectiveSSM is causal")

# ---- variants: equal to base at init, train, gradients reach the block
base_kw = dict(in_ch=1, out_ch=3, n_cond=4, **small)
base = L.UNet(**base_kw)
sar, season = torch.randn(2, 1, IMG, IMG), torch.tensor([0, 3])
for kind in L.REFINE_BLOCKS:
    v = L.make_variant(kind, base_kw, level=2)
    missing, unexpected = v.load_state_dict(base.state_dict(), strict=False)
    check(all(k.startswith("refine.") for k in missing) and not unexpected, f"{kind}: base weights load (new keys only in refine)")
    check(L.max_output_diff(base, v, sar, season) < 1e-5, f"{kind}: identical to base at initialisation")
    v.train(); out = v(sar, cond=season, aux=sar); out.mean().backward()
    g = [p.grad for n_, p in v.named_parameters() if n_.startswith("refine.") and p.grad is not None]
    check(kind == "base" or (len(g) > 0 and all(torch.isfinite(t).all() for t in g)), f"{kind}: finite gradients into the block")
    print(f"      {kind:10s} extra params {L.n_params(v) - L.n_params(base):.3f} M")
pv, meta = L.train_regressor("ft_mamba", L.make_variant("mamba", base_kw), loader, va_ds, dev, 1, 3, out_dir, val_every=2, warmup=1)
check(meta["steps"] == 3, "variant fine-tunes through train_regressor (with season FiLM)")

shutil.rmtree(root)
print("ALL SMOKE TESTS PASSED")
