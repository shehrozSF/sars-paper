"""Builds kaggle/01_baselines_comparison_kaggle.ipynb and kaggle/02_spatial_blocks_kaggle.ipynb.

Both notebooks embed kaggle/src/s2o_lib.py verbatim (%%writefile) and reuse the .env loader and the
KAGGLE SETUP cell from kaggle/freq_split_kaggle.ipynb, so they behave like the other Kaggle notebooks.
Run from the repo root:  python scripts/build_experiment_notebooks.py
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = open(os.path.join(ROOT, "kaggle", "src", "s2o_lib.py"), encoding="utf-8").read()
_ref = json.load(open(os.path.join(ROOT, "kaggle", "freq_split_kaggle.ipynb"), encoding="utf-8"))
ENV_CELL, SETUP_CELL = "".join(_ref["cells"][0]["source"]), "".join(_ref["cells"][2]["source"])
assert "[env] load .env" in ENV_CELL and "KAGGLE SETUP" in SETUP_CELL, "reference cells moved; update indices"
# Kaggle now mounts datasets at /kaggle/input/datasets/<user>/<dataset>/, so also look there
_GLOB = '*sorted(pathlib.Path("/kaggle/input").glob("*/.env"))'
assert _GLOB in ENV_CELL, "env loader changed; update the glob patch"
ENV_CELL = ENV_CELL.replace(_GLOB, _GLOB + ',\n           *sorted(pathlib.Path("/kaggle/input").glob("datasets/*/*/.env"))')


def nb(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


LIB_CELL = "%%writefile s2o_lib.py\n" + LIB

IMPORTS = r'''
import os, sys, json, time, random, glob, shutil
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
import importlib, s2o_lib as L
importlib.reload(L)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.environ.get("SAR_DATA_ROOT", "")
assert DATA_ROOT, "attach the Kaggle dataset shambac/augmented-sentinel-1-2 (the setup cell prints SAR_DATA_ROOT)"
os.makedirs(OUT, exist_ok=True)
import lpips
LPIPS = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
WORKERS = min(12, os.cpu_count() or 4)
L.log(f"GPU {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'none'} | CPU workers {WORKERS} | {CLOCK}")
'''

FIND_PRIOR = r'''
def find_prior(fname, depth=6):
    """Look for a file produced by an earlier run (attached as a Kaggle input) without walking image folders."""
    level = ["/kaggle/input"]
    for _ in range(depth):
        nxt = []
        for d in level:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                return p
            try:
                with os.scandir(d) as it:
                    kids = [e.path for i, e in zip(range(300), it) if e.is_dir()]
                nxt += kids if len(kids) < 300 else []
            except OSError:
                pass
        level = nxt
    return None
'''

DATA = r'''
# ---------------- data: scene-disjoint split, saved; exact-duplicate audit; evaluation subsets ----------------
prior = find_prior(SPLIT_FILE)
df, info = L.build_index(DATA_ROOT)
df, digest = L.scene_split(df, SPLIT_FRACS, SPLIT_SEED)
L.log(f"index: {info} | split digest {digest} | " + str(df.split.value_counts().to_dict()))
if prior:
    saved = json.load(open(prior))
    assert saved["digest"] == digest, f"split differs from {prior}: rebuild produced {digest}, file has {saved['digest']}"
    audit, excluded = saved["duplicate_audit"], set(saved["excluded_from_eval"])
    L.log(f"split verified against {prior}")
elif AUDIT_DUPLICATES:
    audit, excluded = L.audit_duplicates(df, DATA_ROOT, workers=32)
    L.log(f"duplicate audit: {audit}")
else:
    audit, excluded = None, set()
SPLIT = L.save_split(f"{OUT}/{SPLIT_FILE}", df, digest, info, audit, excluded)

train_df = df[df.split == "train"]
val_df = df[(df.split == "val") & ~df.s2.isin(excluded)]
test_df = df[(df.split == "test") & ~df.s2.isin(excluded)]
val_df = val_df.sample(n=min(N_VAL, len(val_df)), random_state=1)
test_df = test_df.sample(n=min(N_TEST, len(test_df)), random_state=2)
TRAIN = L.PairDS(train_df, DATA_ROOT, augment=True)
VAL, TEST = L.PairDS(val_df, DATA_ROOT), L.PairDS(test_df, DATA_ROOT)
L.log(f"train {len(TRAIN):,} pairs / {train_df.scene.nunique()} scenes | val {len(VAL)} | "
      f"test {len(TEST)} from {test_df.scene.nunique()} scenes | excluded duplicates {len(excluded)}")
'''

# =================================================================================================== notebook 1
NB1 = [
("markdown", r'''
# SAR→optical baselines on a scene-disjoint SEN1-2 split

One Kaggle session (RTX Pro 6000, Internet off), about **11 hours**. Every method is trained **from scratch** on the
same scene-disjoint split with the same data pipeline, the same U-Net backbone where the method has one, and the
same evaluation code. Nothing here reuses the old checkpoints: those saw about 95% of all patches during training.

| Method | Family | What it is here | Selection |
|---|---|---|---|
| Constant colour | floor | mean training RGB everywhere | — |
| U-Net (L1+SSIM) | regressor | SAR→RGB, single forward pass | best validation PSNR (EMA) |
| Pix2Pix | conditional GAN | U-Net generator + 70×70 PatchGAN, LSGAN + 100·L1 | final EMA |
| CycleGAN | unpaired GAN | two U-Net generators, two PatchGANs, cycle-L1 10; trained without pairs | final EMA |
| DDPM | diffusion | SAR-concatenated conditional DDPM, cosine schedule, DDIM 50 steps | final EMA |
| ResShift-style | residual-shifting bridge | x₀-prediction, T = 15 (the paper's bridge family, pixel space) | final EMA |

These are compact re-implementations of each family on a shared backbone, not the authors' official code. Label them
"-style" in the paper.

**Protocol.**
- Split by SEN1-2 source scene (`ROIs…_season_N`), 80/10/10 per season, built deterministically and saved with a digest.
- Only files with SEN1-2 names are used; the extra `img_*` files can't be assigned to a scene.
- Exact duplicate optical files are hashed across splits, and any val/test copy of a training image is excluded.
- Metrics are per image, then averaged. Uncertainty is a 95% bootstrap over **test scenes**.
- Test images are scored once, after training.

**Seeds.** The main table uses **one training seed per method**, with scene-bootstrap CIs. With about 25 test scenes,
which scenes you test on is the larger source of uncertainty. Training-seed variance is then measured on the cheap
methods: if time remains, the U-Net and Pix2Pix get seeds 1 and 2, so the table can say how large seed noise is next
to the gaps between methods. Three seeds of DDPM and CycleGAN would need roughly another 11-hour session.

**Inputs to attach:** `sar-assets-base` (for the LPIPS/FID weights and wheels), the Kaggle dataset
`shambac/augmented-sentinel-1-2`; a `.env` dataset is optional. **Outputs:** `/kaggle/working/nb1_baselines/`. Attach
this run's output to notebook 2 so it reuses the same split file.
'''),
("code", ENV_CELL),
("code", SETUP_CELL),
("code", r'''
# ---------------- configuration ----------------
import time
CLOCK_HOURS   = 11.0           # stop starting new work after this; Kaggle kills sessions at 12 h
RESERVE_MIN   = 20             # kept free at the end for tables and figures
OUT           = "/kaggle/working/nb1_baselines"
SPLIT_FILE, SPLIT_FRACS, SPLIT_SEED = "s2o_split_scene_v1.json", (0.8, 0.1, 0.1), 2026
AUDIT_DUPLICATES = True        # hashes every optical PNG once (~10-20 min); skipped when a saved split is attached
N_VAL, N_TEST = 256, 3000      # validation subset (model selection/monitoring) and the fixed test subset
BATCH, BATCH_CYCLE = 16, 8
UNET = dict(ch=64, mult=(1, 2, 4, 4), nres=2)
# (method, minutes, max_steps) in run order; minutes is the training wall-clock budget
PLAN = [("const", 0, 0), ("unet", 45, 60000), ("pix2pix", 70, 60000), ("cyclegan", 90, 60000),
        ("ddpm", 120, 300000), ("resshift", 100, 300000)]
EVAL_MIN = {"const": 1, "unet": 3, "pix2pix": 3, "cyclegan": 3, "ddpm": 10, "resshift": 5}   # rough, for scheduling
EXTRA_SEEDS = [("unet", 1), ("unet", 2), ("pix2pix", 1), ("pix2pix", 2)]                    # only while time remains
DDIM_STEPS, RESSHIFT_KAPPA = 50, 1.0
'''),
("code", LIB_CELL),
("code", "CLOCK = None\n" + IMPORTS.replace("os.makedirs(OUT, exist_ok=True)", "os.makedirs(OUT, exist_ok=True)\nCLOCK = L.SessionClock(CLOCK_HOURS)")),
("code", FIND_PRIOR + DATA),
("code", r'''
# ---------------- runner: train -> evaluate on TEST -> save; skips work that is already saved ----------------
RES_DIR = f"{OUT}/results"; CK_DIR = f"{OUT}/ckpt"
os.makedirs(RES_DIR, exist_ok=True)
PRED = {}

def build_and_train(method, seed, minutes, max_steps):
    L.seed_all(seed)
    tag = f"{method}_s{seed}"
    bs = BATCH_CYCLE if method == "cyclegan" else BATCH
    loader = L.make_loader(TRAIN, bs, True, WORKERS, drop_last=True)
    if method == "const":
        m, pred = L.train_mean_colour(TRAIN)
        return pred, {"name": tag, "mean_rgb": m.tolist(), "params_M": 0, "steps": 0, "minutes": 0}
    if method == "unet":
        return L.train_regressor(tag, L.UNet(1, 3, **UNET), loader, VAL, dev, minutes, max_steps, CK_DIR, use_season=False)
    if method == "pix2pix":
        return L.train_pix2pix(tag, L.UNet(1, 3, **UNET), loader, VAL, dev, minutes, max_steps, CK_DIR)
    if method == "cyclegan":
        return L.train_cyclegan(tag, L.UNet(1, 3, **UNET), L.UNet(3, 1, **UNET), loader, VAL, dev, minutes, max_steps, CK_DIR)
    if method == "ddpm":
        return L.train_ddpm(tag, L.UNet(4, 3, time=True, **UNET), loader, VAL, dev, minutes, max_steps, CK_DIR,
                            ddim_steps=DDIM_STEPS, seed=seed)
    if method == "resshift":
        return L.train_resshift(tag, L.UNet(4, 3, time=True, **UNET), loader, VAL, dev, minutes, max_steps, CK_DIR,
                                kappa=RESSHIFT_KAPPA, seed=seed)
    raise ValueError(method)

def run(method, seed, minutes, max_steps):
    tag = f"{method}_s{seed}"
    if os.path.exists(f"{RES_DIR}/{tag}.json"):
        L.log(f"{tag}: already done, skipping"); return True
    need = minutes + EVAL_MIN[method]
    if CLOCK.left_min() - RESERVE_MIN < need:
        L.log(f"{tag}: SKIPPED, needs {need} min, {CLOCK}"); return False
    L.log(f"=== {tag}: train {minutes} min | {CLOCK}")
    pred, meta = build_and_train(method, seed, minutes, max_steps)
    per, fid = L.evaluate(pred, TEST, 32, dev, lpips_fn=LPIPS, fid=True, desc=tag)
    per.to_csv(f"{RES_DIR}/{tag}.csv", index=False)
    row = L.summarise(per, fid, method, {"seed": seed, "params_M": meta.get("params_M"), "steps": meta.get("steps"),
                                         "train_min": meta.get("minutes"), "sec_per_step": meta.get("sec_per_step")})
    json.dump({"summary": row, "meta": meta}, open(f"{RES_DIR}/{tag}.json", "w"), indent=1, default=str)
    PRED[tag] = pred
    torch.cuda.empty_cache()
    return True
'''),
("code", r'''
# ---------------- main plan: one seed per method ----------------
for method, minutes, steps in PLAN:
    run(method, 0, minutes, steps)
L.log(f"main plan finished | {CLOCK}")
'''),
("code", r'''
# ---------------- extra seeds (training-seed variance), only while time remains ----------------
for method, seed in EXTRA_SEEDS:
    minutes, steps = {m: (mi, st) for m, mi, st in PLAN}[method]
    run(method, seed, minutes, steps)
L.log(f"extra seeds finished | {CLOCK}")
'''),
("code", r'''
# ---------------- results table ----------------
rows = [json.load(open(p))["summary"] for p in sorted(glob.glob(f"{RES_DIR}/*.json"))]
R = pd.DataFrame(rows)
per = {f"{r.model}_s{int(r.seed)}": pd.read_csv(f"{RES_DIR}/{r.model}_s{int(r.seed)}.csv") for r in R.itertuples()}
order = [m for m, _, _ in PLAN]
main = R[R.seed == 0].set_index("model").reindex([m for m in order if m in set(R.model)])
for m in main.index:                                  # paired difference to the U-Net regressor, same test images
    if "unet_s0" in per and m != "unet":
        d, lo, hi = L.paired_delta(per["unet_s0"], per[f"{m}_s0"])
        main.loc[m, "dPSNR_vs_unet"], main.loc[m, "dPSNR_lo"], main.loc[m, "dPSNR_hi"] = d, lo, hi
seedvar = R.groupby("model").agg(seeds=("seed", "nunique"), PSNR_seed_mean=("PSNR", "mean"), PSNR_seed_sd=("PSNR", "std"),
                                 SSIM_seed_sd=("SSIM", "std"), FID_seed_sd=("FID", "std"))
TABLE = main.join(seedvar)
pd.set_option("display.float_format", "{:.3f}".format)
cols = ["PSNR", "PSNR_lo", "PSNR_hi", "SSIM", "LPIPS", "FID", "SAM", "CC", "deltaE", "lf_rmse", "grad_err",
        "dPSNR_vs_unet", "dPSNR_lo", "dPSNR_hi", "seeds", "PSNR_seed_sd", "params_M", "steps", "train_min"]
print(f"TEST: {len(TEST)} images from {test_df.scene.nunique()} unseen scenes | split {digest}")
print(TABLE[[c for c in cols if c in TABLE]].to_string())
TABLE.to_csv(f"{OUT}/table_main.csv")
season = {k: v.groupby("season")[["psnr", "ssim"]].mean() for k, v in per.items() if k.endswith("_s0")}
print("\nPSNR by season:\n", pd.DataFrame({k: v.psnr for k, v in season.items()}).round(2).to_string())
'''),
("code", r'''
# ---------------- figures: PSNR-FID plane, PSNR with scene CIs ----------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m, r in main.iterrows():
    if np.isfinite(r.FID):
        ax[0].scatter(r.FID, r.PSNR, s=60); ax[0].annotate(m, (r.FID, r.PSNR), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax[0].invert_xaxis(); ax[0].set_xlabel("FID (better →)"); ax[0].set_ylabel("PSNR (dB)"); ax[0].grid(alpha=.3)
ax[0].set_title("distortion vs perception, scene-disjoint test", fontsize=9)
y = np.arange(len(main))
ax[1].errorbar(main.PSNR, y, xerr=[main.PSNR - main.PSNR_lo, main.PSNR_hi - main.PSNR], fmt="o", capsize=3)
ax[1].set_yticks(y, main.index); ax[1].set_xlabel("PSNR (dB), 95% scene-bootstrap CI"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_main.png", dpi=200); plt.show()
'''),
("code", r'''
# ---------------- visual grid: same test images for every method (seed 0) ----------------
import matplotlib.pyplot as plt
picks = [test_df.index.get_loc(test_df[test_df.season == s].index[0]) for s in L.SEASONS]
tags = [f"{m}_s0" for m in main.index if f"{m}_s0" in PRED]
fig, axs = plt.subplots(len(picks), len(tags) + 2, figsize=(2 * (len(tags) + 2), 2 * len(picks)))
with torch.no_grad():
    for r, i in enumerate(picks):
        b = TEST[i]; sar, gt = b["sar"][None].to(dev), b["opt"][None].to(dev)
        ims = [sar.repeat(1, 3, 1, 1), gt] + [PRED[t](sar, torch.tensor([b["season"]], device=dev)) for t in tags]
        for k, im in enumerate(ims):
            axs[r, k].imshow(((im[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy())
            axs[r, k].set_xticks([]); axs[r, k].set_yticks([])
            if k >= 2:
                axs[r, k].set_xlabel(f"{float(L.image_metrics(im.float(), gt)['psnr']):.1f} dB", fontsize=7)
        axs[r, 0].set_ylabel(L.SEASONS[b["season"]], fontsize=8)
for k, t in enumerate(["SAR", "ground truth"] + tags):
    axs[0, k].set_title(t, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_grid.png", dpi=150); plt.show()
L.log(f"done | {CLOCK} | outputs in {OUT}")
'''),
("markdown", r'''
**Reading the result.**
- `PSNR_lo/hi` are 95% CIs over test scenes.
- `dPSNR_vs_unet` is paired on identical images. A difference whose CI includes 0 is not a result.
- `PSNR_seed_sd` (U-Net and Pix2Pix, if extra seeds ran) shows how much training noise moves the score. Compare it with
  the gaps between methods before trusting a single-seed ranking.
- FID with 3000 images is biased upward and comparable only within this table.
- The constant-colour row is the floor. A method that is not clearly above it has learned little on unseen scenes.
'''),
]

# =================================================================================================== notebook 2
NB2 = [
("markdown", r'''
# Improving spatial fidelity: fixes first, then decoder blocks (Conv2D, ConvLSTM, xLSTM, Mamba, SPADE)

## Part 0: fixes from the audit, applied here

| Problem in the original pipeline | Fix in this notebook |
|---|---|
| Training scripts shuffled before splitting and evaluation scripts didn't, so ~95% of "validation" images were training images; the old split cannot be recovered | One **scene-disjoint split** built deterministically from SEN1-2 scene ids, saved with a digest, verified against notebook 1's file when attached |
| Patches overlap 50% inside a scene; augmented corpus has extra `img_*` files of unknown origin | Split by scene; only SEN1-2-named files; exact-duplicate audit across splits, duplicates excluded from evaluation |
| v3 was evaluated with the untuned decoder (`adapted16_final`) instead of its own `base_vae_tuned.pth` | Optional reference row evaluates v3 with **both** decoders on the new test images (labelled EXPOSED: v3 saw most of them) |
| Validation PSNR from batch-averaged MSE in some scripts (~2 dB low) | Per-image PSNR everywhere |
| `best` checkpoint silently fell back to `last`; flow built on a failed run | Every run saves the checkpoint it actually selected, with its validation history; selection uses validation only |
| Old checkpoints are contaminated | All models here are trained from scratch on the clean split |

The improvement experiments use a **pixel-space U-Net** (no VAE). On unseen scenes a 50M pixel network matched the
860M latent pipeline, and it costs a fraction of the compute, so it is the right test bed for architectural changes.
If the blocks help here, porting the winner into the latent model is the next step.

## Why these blocks: evidence from other image-translation domains
The residual error is mostly **low-frequency colour placed per region**. A purely local CNN decoder sees a limited
neighbourhood, and normalisation layers can wash out the SAR's spatial layout. Each block below comes from a domain
where it helped keep spatial structure during image-to-image translation or restoration:

| Block | Source domain and evidence | Hypothesis for SAR→optical |
|---|---|---|
| **SPADE** spatially-adaptive normalisation | Semantic image synthesis ([Park et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html)): ordinary normalisation "washes away" the layout of the input map; predicting per-pixel scale/shift from it fixes that | Re-inject the SAR layout into the decoder so colour regions follow SAR boundaries |
| **Mamba** selective scans | NIR→RGB colourisation ([ColorMamba](https://arxiv.org/html/2408.08087v1)) and restoration ([MambaIRv2, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html)): long-range context at linear cost | Scene-level context to choose consistent colour across a tile |
| **ConvLSTM** recurrent refinement | Deraining ([PReNet, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Ren_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline_CVPR_2019_paper.html)), mechanism from [Shi et al. 2015](https://arxiv.org/abs/1506.04214) | Repeated refinement of the same features aggregates spatial evidence |
| **ConvxLSTM** (sLSTM) | [xLSTM](https://arxiv.org/abs/2405.04517); vision variants in medical segmentation ([xLSTM-UNet](https://arxiv.org/abs/2407.01530)) | Exponential gating keeps more long-range state than ConvLSTM |
| **Conv2D** stack | control | Same depth and width, no recurrence or global context: tells you whether a gain is just extra capacity |

Each block sits in the decoder at the 64×64 level (`REFINE_LEVEL = 2`) and adds a residual through a **zero-initialised**
output projection, so every variant starts exactly equal to the base network. This is checked in code. If your
diagram places the blocks elsewhere, change `REFINE_LEVEL` (0 = full resolution, 3 = 32×32).

## Design
1. Train one **base** U-Net (SAR + season FiLM, L1 + 0.15·(1−SSIM)) on the clean split.
2. Fine-tune **every variant from the same base weights for the same number of steps**, including `base` itself
   (continued training), so a variant only counts as better if it beats more training of the base.
3. **Screen** (this session): one seed per variant; pick the top two by **validation** PSNR.
4. **Confirm** (next session, `PHASE = "confirm"`): base, conv and the top two with seeds 1 and 2, then report
   mean ± SD over three seeds and paired scene CIs.

`PHASE = "all"` runs screen and confirm in one session. To fit in 11 hours it halves fine-tuning to 5k steps per
variant (`FT_STEPS_ALL`). Same design and controls, but shorter fine-tuning, so small gains may not show.

Metrics include colour and structure diagnostics:
- **ΔE** (CIE76 colour error);
- **LF-RMSE** (error after a σ=4 Gaussian low-pass, i.e. coarse colour);
- **GradErr** (Sobel-edge error).

**Inputs:** `sar-assets-base`, `shambac/augmented-sentinel-1-2` (a `.env` dataset is optional), and ideally notebook 1's
output (for the shared split file). For `PHASE = "confirm"` also attach this notebook's screen output.
**Outputs:** `/kaggle/working/nb2_improve/`.
'''),
("code", ENV_CELL),
("code", SETUP_CELL),
("code", r'''
# ---------------- configuration ----------------
PHASE         = "screen"        # "screen": base + all variants, seed 0 | "confirm": top variants with seeds 1, 2 | "all": both, one session
CLOCK_HOURS   = 11.0
RESERVE_MIN   = 20
OUT           = "/kaggle/working/nb2_improve"
SPLIT_FILE, SPLIT_FRACS, SPLIT_SEED = "s2o_split_scene_v1.json", (0.8, 0.1, 0.1), 2026   # same as notebook 1
AUDIT_DUPLICATES = True
N_VAL, N_TEST = 1000, 3000      # validation subset (selection) and fixed test subset
BATCH         = 16
BASE_KW       = dict(in_ch=1, out_ch=3, n_cond=4, ch=64, mult=(1, 2, 4, 4), nres=2)
REFINE_LEVEL  = 2               # decoder level for the block: 2 = 64x64 features
BASE_MIN, BASE_MAX_STEPS = 150, 200000
FT_STEPS, FT_MAX_MIN     = 10000, 50   # equal steps for every variant; minutes is a safety cap
FT_STEPS_ALL, FT_MAX_MIN_ALL = 5000, 25  # used instead when PHASE = "all", so 14 fine-tunes fit in one session
if PHASE == "all":
    FT_STEPS, FT_MAX_MIN = FT_STEPS_ALL, FT_MAX_MIN_ALL
FT_LR_BASE, FT_LR_BLOCK  = 5e-5, 2e-4
VARIANTS      = ["base", "conv", "convlstm", "convxlstm", "mamba", "spade"]
CONFIRM_SEEDS = (1, 2)
RUN_V3_REFERENCE, N_V3 = True, 1000    # old v3 with tuned vs untuned decoder on the new test images (needs sar-assets-base)
os.environ.setdefault("HF_REPO_ID", "SotaSF/sar-to-optical-diffusion")   # v3 weights; served offline from sar-assets-base
'''),
("code", LIB_CELL),
("code", "CLOCK = None\n" + IMPORTS.replace("os.makedirs(OUT, exist_ok=True)", "os.makedirs(OUT, exist_ok=True)\nCLOCK = L.SessionClock(CLOCK_HOURS)")),
("code", FIND_PRIOR + DATA),
("code", r'''
# ---------------- Part 0b (optional): v3 with the wrong vs the right decoder, on the new test images ----------------
# EXPOSED: v3 was trained on a per-patch split of this corpus and has seen most of these images. This row only
# demonstrates the decoder fix; it is not a clean result and must not be compared with the models trained below.
V3_ROWS = []
if RUN_V3_REFERENCE and os.environ.get("SD15_PATH") and os.environ.get("HF_REPO_ID") and not os.path.exists(f"{OUT}/v3_reference.csv"):
    import torch.nn as nn
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    HF_REPO, SD_ID = os.environ["HF_REPO_ID"], os.environ["SD15_PATH"]

    class DetailEncoder(nn.Module):
        def __init__(s, ic=3, oc=12, nf=64):
            super().__init__()
            s.net = nn.Sequential(nn.Conv2d(ic, nf, 3, 2, 1), nn.GroupNorm(8, nf), nn.SiLU(), nn.Conv2d(nf, nf * 2, 3, 2, 1),
                                  nn.GroupNorm(8, nf * 2), nn.SiLU(), nn.Conv2d(nf * 2, nf * 4, 3, 2, 1), nn.GroupNorm(8, nf * 4),
                                  nn.SiLU(), nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), nn.GroupNorm(8, nf * 4), nn.SiLU(), nn.Conv2d(nf * 4, oc, 3, 1, 1))
        def forward(s, x): return s.net(x)

    def load_vae(base_file):
        bv = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae")
        o = bv.decoder.conv_in
        bv.decoder.conv_in = nn.Conv2d(16, o.out_channels, o.kernel_size, o.stride, o.padding)
        de = DetailEncoder(); de.load_state_dict(torch.load(hf_hub_download(HF_REPO, "adapted16_final/detail_encoder.pth"), map_location="cpu"))
        bv.load_state_dict(torch.load(hf_hub_download(HF_REPO, base_file), map_location="cpu"))
        return bv.to(dev).float().eval(), de.to(dev).float().eval()

    cfg = json.load(open(hf_hub_download(HF_REPO, "regressor_v3_final/bridge_config.json")))
    u = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet")
    oi, oo = u.conv_in, u.conv_out
    u.conv_in = nn.Conv2d(32, oi.out_channels, oi.kernel_size, oi.stride, oi.padding)
    u.conv_out = nn.Conv2d(oo.in_channels, 16, oo.kernel_size, oo.stride, oo.padding)
    u.load_state_dict(load_file(hf_hub_download(HF_REPO, "regressor_v3_final/unet.safetensors"))); u = u.to(dev).float().eval()
    tok = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer"); txt = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder").to(dev)
    with torch.no_grad():
        null = txt(tok("", padding="max_length", max_length=77, return_tensors="pt").input_ids.to(dev))[0]
    del txt
    S, TB = float(cfg["latent_scale"]), int(cfg["fixed_tb"])
    v3_test = L.PairDS(test_df.head(N_V3), DATA_ROOT)          # empty prompt for both rows, so only the decoder differs
    for label, base_file in (("v3 + adapted16 decoder (as in Table II)", "adapted16_final/base_vae.pth"),
                             ("v3 + its tuned decoder (correct)", "regressor_v3_final/base_vae_tuned.pth")):
        bv, de = load_vae(base_file)
        @torch.no_grad()
        def pred(sar, season, bv=bv, de=de):
            x = sar.repeat(1, 3, 1, 1)
            z = torch.cat([bv.post_quant_conv(bv.encode(x).latent_dist.mode()), de(x)], 1) * S
            out = u(torch.cat([z, z], 1), torch.full((x.shape[0],), TB, device=dev), encoder_hidden_states=null.expand(x.shape[0], -1, -1)).sample
            return bv.decoder(out / S)
        perv, fid = L.evaluate(pred, v3_test, 16, dev, lpips_fn=LPIPS, fid=True, desc=label)
        V3_ROWS.append(L.summarise(perv, fid, label, {"status": "EXPOSED (trained on most of these images)"}))
        del bv, de; torch.cuda.empty_cache()
    pd.DataFrame(V3_ROWS).to_csv(f"{OUT}/v3_reference.csv", index=False)
    del u; torch.cuda.empty_cache()
if os.path.exists(f"{OUT}/v3_reference.csv"):
    print(pd.read_csv(f"{OUT}/v3_reference.csv")[["model", "PSNR", "SSIM", "LPIPS", "FID", "status"]].to_string(index=False))
else:
    print("v3 reference skipped (set RUN_V3_REFERENCE and attach sar-assets-base)")
'''),
("code", r'''
# ---------------- Part 1: the shared base model (seed 0), reused by every variant ----------------
RES_DIR, CK_DIR = f"{OUT}/results", f"{OUT}/ckpt"
os.makedirs(RES_DIR, exist_ok=True); os.makedirs(CK_DIR, exist_ok=True)
VAL_MON = L.PairDS(val_df.head(256), DATA_ROOT)          # cheap validation monitor during training
BASE_CK = f"{CK_DIR}/base_pretrain.pt"
if not os.path.exists(BASE_CK) and (p := find_prior("base_pretrain.pt")):
    shutil.copy(p, BASE_CK); shutil.copy(p + ".json", BASE_CK + ".json"); L.log(f"base restored from {p}")
if not os.path.exists(BASE_CK):
    assert PHASE != "confirm", "confirm phase needs the screen output (base_pretrain.pt) attached"
    L.seed_all(0)
    loader = L.make_loader(TRAIN, BATCH, True, WORKERS, drop_last=True)
    _, meta = L.train_regressor("base_pretrain", L.UNet(**BASE_KW), loader, VAL_MON, dev, BASE_MIN, BASE_MAX_STEPS,
                                CK_DIR, use_season=True, select="best")
    L.log(f"base: {meta['steps']} steps, best val PSNR {meta['best_val_psnr']:.3f} | {CLOCK}")
print(json.load(open(BASE_CK + ".json"))["val_hist"][-5:])
'''),
("code", r'''
# ---------------- Part 2: variants fine-tuned from the same base for the same number of steps ----------------
FT_EVAL_MIN = 6

def fine_tune(kind, seed):
    tag = f"{kind}_s{seed}"
    if os.path.exists(f"{RES_DIR}/{tag}.json"):
        L.log(f"{tag}: done, skipping"); return
    if CLOCK.left_min() - RESERVE_MIN < FT_MAX_MIN + FT_EVAL_MIN:
        L.log(f"{tag}: SKIPPED for time | {CLOCK}"); return
    L.seed_all(seed)
    base = L.UNet(**BASE_KW); L.load_ckpt(BASE_CK, base)
    model = L.make_variant(kind, BASE_KW, REFINE_LEVEL)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith("refine.") for k in missing) and not unexpected
    b = next(iter(L.make_loader(VAL_MON, 4, False, 0)))
    diff = L.max_output_diff(base.to(dev), model.to(dev), b["sar"].to(dev), b["season"].to(dev))
    assert diff < 1e-4, f"{tag}: variant differs from base at init ({diff})"
    blk = [p for n, p in model.named_parameters() if n.startswith("refine.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("refine.")]
    groups = [{"params": rest, "lr": FT_LR_BASE}] + ([{"params": blk, "lr": FT_LR_BLOCK}] if blk else [])
    L.log(f"=== {tag}: +{L.n_params(model) - L.n_params(base):.2f} M params, identical at init (max diff {diff:.1e}) | {CLOCK}")
    loader = L.make_loader(TRAIN, BATCH, True, WORKERS, drop_last=True)
    pred, meta = L.train_regressor(tag, model, loader, VAL_MON, dev, FT_MAX_MIN, FT_STEPS, CK_DIR, lr=FT_LR_BASE,
                                   use_season=True, select="best", val_every=1000, warmup=200, param_groups=groups)
    pv, _ = L.evaluate(pred, VAL, 32, dev, lpips_fn=LPIPS, fid=False, desc=f"{tag} VAL")
    pt, fid = L.evaluate(pred, TEST, 32, dev, lpips_fn=LPIPS, fid=True, desc=f"{tag} TEST")
    pv.to_csv(f"{RES_DIR}/{tag}_val.csv", index=False); pt.to_csv(f"{RES_DIR}/{tag}_test.csv", index=False)
    extra = {"seed": seed, "variant": kind, "extra_params_M": round(L.n_params(model) - L.n_params(base), 3),
             "steps": meta["steps"], "sec_per_step": meta["sec_per_step"], "peak_vram_gb": meta["peak_vram_gb"],
             "val_PSNR": pv.psnr.mean(), "val_deltaE": pv.deltaE.mean()}
    json.dump({"summary": L.summarise(pt, fid, kind, extra), "meta": meta}, open(f"{RES_DIR}/{tag}.json", "w"), indent=1, default=str)
    del model, base; torch.cuda.empty_cache()

if PHASE in ("screen", "all"):
    for kind in VARIANTS:
        fine_tune(kind, 0)
    S0 = pd.DataFrame([json.load(open(f"{RES_DIR}/{k}_s0.json"))["summary"] for k in VARIANTS if os.path.exists(f"{RES_DIR}/{k}_s0.json")])
    cand = S0[~S0.variant.isin(["base", "conv"])].sort_values("val_PSNR", ascending=False).variant.tolist()[:2]
    json.dump({"confirm": ["base", "conv"] + cand, "rule": "top-2 non-control variants by VALIDATION PSNR"},
              open(f"{OUT}/confirm_list.json", "w"), indent=1)
    L.log(f"selected for confirmation (by validation PSNR): {cand}")
if PHASE in ("confirm", "all"):
    p = f"{OUT}/confirm_list.json" if os.path.exists(f"{OUT}/confirm_list.json") else find_prior("confirm_list.json")
    confirm = json.load(open(p))["confirm"]
    if os.path.dirname(p) != OUT:
        for f in glob.glob(os.path.join(os.path.dirname(p), "results", "*_s0*")):
            shutil.copy(f, RES_DIR)                     # seed-0 screen results join the confirmation table
    for seed in CONFIRM_SEEDS:
        for kind in confirm:
            fine_tune(kind, seed)
L.log(f"phase {PHASE} finished | {CLOCK}")
'''),
("code", r'''
# ---------------- results: paired against continued base training, scene CIs, seeds ----------------
rows = [json.load(open(p))["summary"] for p in sorted(glob.glob(f"{RES_DIR}/*_s[0-9].json"))]
R = pd.DataFrame(rows)
per = {(r.variant, int(r.seed)): pd.read_csv(f"{RES_DIR}/{r.variant}_s{int(r.seed)}_test.csv") for r in R.itertuples()}
for i, r in R.iterrows():
    ref = per.get(("base", int(r.seed)))
    if ref is not None and r.variant != "base":
        for m in ("psnr", "ssim", "deltaE", "lf_rmse", "grad_err"):
            d, lo, hi = L.paired_delta(ref, per[(r.variant, int(r.seed))], m)
            R.loc[i, f"d_{m}"], R.loc[i, f"d_{m}_lo"], R.loc[i, f"d_{m}_hi"] = d, lo, hi
pd.set_option("display.float_format", "{:.3f}".format)
show = ["variant", "seed", "PSNR", "PSNR_lo", "PSNR_hi", "SSIM", "LPIPS", "FID", "deltaE", "lf_rmse", "grad_err",
        "d_psnr", "d_psnr_lo", "d_psnr_hi", "d_deltaE", "d_grad_err", "val_PSNR", "extra_params_M", "sec_per_step", "steps"]
print(f"TEST {len(TEST)} images, {test_df.scene.nunique()} unseen scenes | d_* = variant - continued base, same seed, 95% scene CI")
print(R.sort_values(["variant", "seed"])[[c for c in show if c in R]].to_string(index=False))
if R.seed.nunique() > 1:
    agg = R.groupby("variant").agg(seeds=("seed", "nunique"), PSNR=("PSNR", "mean"), PSNR_sd=("PSNR", "std"),
                                   d_psnr=("d_psnr", "mean"), d_psnr_sd=("d_psnr", "std"), d_deltaE=("d_deltaE", "mean"),
                                   SSIM=("SSIM", "mean"), FID=("FID", "mean"))
    print("\nmean ± SD over seeds:\n", agg.to_string())
    agg.to_csv(f"{OUT}/table_seeds.csv")
R.to_csv(f"{OUT}/table_runs.csv", index=False)
'''),
("code", r'''
# ---------------- figure: paired gains over continued base training ----------------
import matplotlib.pyplot as plt
P = R[R.variant != "base"].dropna(subset=["d_psnr"])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, m, lab in zip(ax, ("psnr", "deltaE", "grad_err"), ("ΔPSNR (dB) ↑", "ΔdeltaE ↓", "ΔGradErr ↓")):
    y = np.arange(len(P))
    a.errorbar(P[f"d_{m}"], y, xerr=[P[f"d_{m}"] - P[f"d_{m}_lo"], P[f"d_{m}_hi"] - P[f"d_{m}"]], fmt="o", capsize=3)
    a.axvline(0, color="grey", lw=1); a.set_yticks(y, [f"{v} s{s}" for v, s in zip(P.variant, P.seed)])
    a.set_xlabel(lab + " vs continued base"); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_variants.png", dpi=200); plt.show()
L.log(f"done | {CLOCK} | outputs in {OUT}")
'''),
("markdown", r'''
**Reading the result.**
- A block helps only if its **paired** ΔPSNR (or ΔdeltaE / ΔGradErr) CI excludes 0 **and** it beats the `conv` control.
  Beating `base` alone can be extra capacity.
- Screen results are single-seed: use them to choose, not to claim.
- Claims come from the confirm phase: three seeds, mean ± SD, paired scene CIs.
- `sec_per_step` and `extra_params_M` are the cost side of the trade-off. Report them with any gain.
- If nothing beats `conv`, that is a valid negative result: on this data the bottleneck is information (colour that
  SAR doesn't determine), not decoder architecture.
'''),
]

TEST = open(os.path.join(ROOT, "kaggle", "src", "test_s2o_lib.py"), encoding="utf-8").read()
NB0 = [
("markdown", r'''
# Smoke test for `s2o_lib` (run before notebooks 01 and 02)

CPU is enough, no inputs needed, about 2–5 minutes. It builds a tiny fake copy of the dataset layout and checks:
- the scene-disjoint split, the duplicate audit and the dataset loader;
- every trainer and the evaluation code;
- the PSNR definition;
- the sLSTM stabiliser and the Mamba scan;
- that every decoder block starts identical to the base network.

It must end with `ALL SMOKE TESTS PASSED`. Generated by `scripts/build_experiment_notebooks.py`; edit the sources in
`kaggle/src/`, not this notebook.
'''),
("code", LIB_CELL),
("code", "%%writefile test_s2o_lib.py\n" + TEST),
("code", "!python test_s2o_lib.py"),
]

for name, cells in (("00_smoke_test_kaggle.ipynb", NB0), ("01_baselines_comparison_kaggle.ipynb", NB1),
                    ("02_spatial_blocks_kaggle.ipynb", NB2)):
    path = os.path.join(ROOT, "kaggle", name)
    json.dump(nb(cells), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", path)
