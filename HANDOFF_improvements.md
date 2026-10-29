# Handoff: SAR→optical paper — audit findings and change plan

Audience: an AI coding assistant continuing this work in a new session. Read this whole file before touching anything. Everything below was verified by reading the notebooks' code **and their saved outputs** on 2026-10-13; cell numbers are 0-based indices into each notebook's `cells` array.

---

## 1. What this project is

- **Paper:** `SAR_paper_IEEE_latex.pdf` — "Diffusability-Aware Latent Adaptation for SAR-to-Optical Translation: An Oracle Decomposition of the SEN1-2 Error Budget" (IEEE JSTARS draft, 11 pages). The LaTeX source is **not** in this folder; paper changes are written up as instructions, not edited directly.
- **Code:** three Colab notebooks (A100), all in this folder:

| Notebook | Paper sections | Data source |
|---|---|---|
| `SAR-to-Optic.ipynb` (82 cells) | III–VII: 16-ch VAE, ResShift bridge, regressors v1–v3, FLUX swap, calibration, oracles, failed interventions | `kagglehub` `shambac/augmented-sentinel-1-2` (48 GB, = SEN1-2 ROIs1158/1868/1970/2017 + extra `img_*` files). Per-season CSVs `s1_fileName, s2_fileName, season, region`. |
| `complete_pipeline.ipynb` (36 cells) | VIII, Table VI (scene-disjoint 16k corpus) | `kagglehub` `requiemonk/sentinel12-image-pairs-segregated-by-terrain` (`<class>/s1`, `<class>/s2`). Splits saved to HF `data_meta/`. |
| `S2O_Complete_Benchmark.ipynb` (59 cells) | IX (PMRF on official SEN1-2), X / Table VII (`v3x.py`, cells 55–57) | rsync from TUM mediaTUM `rsync://m1436631@dataserv.ub.tum.de/m1436631/`, 30% of scenes by MD5 hash, memmaps in `/content/pmflow_work/data`, scene-disjoint `split.json`. |

## 2. Already done (do not redo)

- Repo IDs switched from `AliMusaRizvi/...` to **`SotaSF/sar-to-optical-diffusion`** (all three notebooks) and **`SotaSF/pmflow-s2o-work`** (benchmark cell 16). Old run-log text in outputs still mentions `AliMusaRizvi`; that is intentional history.
- Hardcoded HF token removed everywhere; code now reads Colab Secret **`HF_TOKEN`** (`from google.colab import userdata; ... userdata.get("HF_TOKEN")`). `v3x.init()` now uses its `hf_token` argument. The old token `hf_CQef…` must be treated as compromised (user to revoke).
- Copied to `SotaSF/sar-to-optical-diffusion`: `adapted16_final`, `bridge_final`, `regressor_final`, `regressor_v2_final`, `regressor_v3_final` (includes `base_vae_tuned.pth`), `regressor_modernvae_final`, `afcrps_final`.
- **Not yet copied** (still in public `AliMusaRizvi/sar-to-optical-diffusion`; copy with the snapshot_download→upload_folder loop if needed): `data_meta`, `fyp_runs` (needed by `complete_pipeline`), `detail_vae_final`, `bridge_resume`, `hetero_final`, `eval_results*`. `AliMusaRizvi/pmflow-s2o-work` is **private** (Stage A/B checkpoints of the benchmark notebook) — unavailable unless shared.
- Backups of the pre-edit notebooks exist only in a temporary scratchpad of the previous session; assume none.

## 3. Working rules for this codebase

- Notebooks are stored as **compact JSON**. To edit programmatically, `json.load` then write with `json.dumps(nb, ensure_ascii=False, separators=(',',':'))` — this round-trips byte-identically. Most scripts are `%%writefile` cells; edit the cell source, not the generated `.py`.
- On this Windows machine `python` may not be on PATH in Bash; use `python3` (`/c/Users/aotos/AppData/Local/Python/bin/python3`). `sklearn` is not installed locally — nothing here can actually be run locally; all runs happen in Colab.
- Metric protocol to keep everywhere: **per-image** PSNR on [-1,1] with `10*log10(4/mse)` then averaged. Never `mse` over a batch (that reads ~2 dB low and has already corrupted several results — see §5).
- Any number that goes into the paper must come from one evaluation on a **saved, versioned split file**, on the **same image set** for every row it is compared with, with n stated.

---

## 4. P0 — bugs that invalidate headline numbers

### P0-1. Train/eval split mismatch (`SAR-to-Optic.ipynb`) — most important
- Every **training** script's `collect*()` does `random.shuffle(pairs)` (after `set_seed(42)`) before `train_test_split(test_size=0.05, stratify=season_terrain, random_state=42)`: cells 21, 25, 28, 32, 34, 40, 42, 46, 48, 56, 60, 62, 64, 66, 68, 70, 74.
- Every **evaluation** script does **not** shuffle: cells 19, 30, 37, 44, 50, 52, 54, 58, 76 (`final_eval_full.py`), 78 (`diag_error_budget.py`), 79 (`free_gains.py`).
- `train_test_split` output depends on input order → the evaluation "validation set" is a different random 5%; ≈95% of its 15,394 images were **training** images. Affects Table II (bridge + v3 rows), Table IV / Fig. 4 (oracles), Table V / Fig. 5 (calibration), Sec. VII item 4–5 numbers, Fig. 3.
- Evidence: v3 scores 17.668 on its true held-out monitor set (`regressor_v3_final/bridge_config.json` → `final_psnr`) vs 18.28–18.32 in the non-shuffled eval scripts.
- **Fix:**
  1. Add one cell that reconstructs the *training-time* split exactly: `set_seed(42)`; the same `collect` (with shuffle); `train_test_split(...)`. Save `{"train":[s1 paths...], "val":[...]}` to `splits/main_split.json`, push to HF.
  2. **Verification gate:** with this split, `random.Random(7).sample(val, 150)` must reproduce v2 = 17.449 PSNR / 0.2905 SSIM (`regressor_v2_final`, original decoder) and v3 = 17.668 / 0.3110 (`regressor_v3_final` + `base_vae_tuned.pth`). If not, RNG state or CSV glob order differs — try sorting `csv_files` and/or replicating calls between `set_seed` and `collect`; do not proceed until it reproduces.
  3. Make every eval script load the JSON instead of re-splitting.
  4. Re-run: `final_eval_full.py` (Table II), `diag_error_budget.py` (Table IV, Fig. 4), `sen12prob_harness.py` + `ensemble_diversity_gate.py` (Table V, Fig. 5, Sec. VI diversity numbers), `x1_calibration_probe.py` (Sec. VII-4/5).
- **UPDATE 2026-10-19 — the training-time split is NOT recoverable.** Tested on Kaggle with the same data (row counts match): rebuilding shuffle-then-split gives v3 monitor PSNR 18.330 (Kaggle CSV order) / 17.995 (sorted) vs the logged 17.668. An exact rebuild would reproduce 17.668 to ±0.01; both values are 'mostly training images'. The shuffle used Colab's directory-listing order of the CSVs, which was never saved. Consequence: clean numbers for the existing models are impossible; existing-model evaluations can only be *relative comparisons* on the contaminated split (label them so). Clean headline numbers need retraining on a saved (ideally scene-disjoint) split — the step below is no longer optional.
- Longer term: the main split is per-patch on SEN1-2 (50% patch overlap). Scene IDs are recoverable from paths (`spring/s2_65/ROIs1158_spring_s2_65_p1.png` → scene `ROIs1158_spring_65`). A scene-disjoint re-split + retrain is the honest version; at minimum report the per-scene audit.

### P0-2. v3 evaluated with the wrong decoder
- v3 (cell 70) fine-tunes the decoder and saves `regressor_v3_final/base_vae_tuned.pth` ("MUST be used with this UNet"). `final_eval_full.py` (cell 76) loads `adapted16_final/base_vae.pth` instead.
- **Fix:** in cell 76 use the `load_vae_for(tag)` pattern from cell 78 (detail encoder from `adapted16_final`, full `base_vae` state from `base_vae_tuned.pth`) for v3; keep the untuned VAE for the bridge. Re-run on the P0-1 split. With the tuned decoder v3 read SSIM ≈ 0.34 on (contaminated) n=400 — headline SSIM may change materially; report whatever comes out.

### P0-3. Leakage figure (3.00 dB, Sec. VIII-A, abstract) measures memorisation
- `complete_pipeline` Table VI per-patch row was evaluated on the 1,657 scene-test images; under the published 80/10/10 patch split ≈80% of those are in `patchsplit`'s **training** set.
- The proper audit `evaluate_experiments(leak_audit=True)` (cell 34; images in neither model's training set, `split_leak_eval.json`) **crashed**: `evaluate.py` (cell 32) filters experiments by *local* `RUNS/<exp>/best/config.json` before ever pulling from the Hub.
- **Fix:** in `evaluate_experiments`, call `hf_pull_file(f"fyp_runs/{e}/best/config.json", RUNS/e/"best"/"config.json", required=False)` before the existence filter. Copy `fyp_runs` + `data_meta` to SotaSF first. Re-run cell 33 (all rows) and cell 34. Note: the cell-33 run that did execute pushed a `summary.json` containing only `curated` to `eval_results/scene_test/` — check whether the original multi-row results survive in Ali's repo history.

### P0-4. `sceneref` uses ground truth
- `sceneref` input = leave-one-out mean RGB of the *optical ground truth* of other patches in the same (test) scene (`data_prep.py`, cell 5, `scene_loo_rgb`). It must be labelled an **oracle**, not grouped with "non-leaked models" (Table VI, Sec. VIII-B).

### P0-5. Contamination in Sec. X / Table VII and Sec. VIII
- `v3x.py` (benchmark cell 56) states v3 had seen ~95% of the official-release **test** patches (it was trained per-patch on the Kaggle copy of the same scenes); its own LaTeX caption says so; the paper's Table VII caption dropped this. v3+'s −1.41 dB is plausibly forgetting of memorised scenes, not a perception–distortion move.
- The 16-ch VAE (`adapted16_final`) was trained on SEN1-2 including the Table VI test scenes; `complete_pipeline` precheck labels v3 there "CONTAMINATED" (15.05 dB). Latent models in Table VI are therefore not fully scene-disjoint; the pixel-space baseline is.

### P0-6. PMRF-latent was built on v3+, not v3
- Benchmark cell 57: `REG_DIR` = `v3plus_reg/best` (exists after the gate-fail save) → the flow is initialised from v3+, conditions on v3+'s ẑ, and blends interpolate around v3+ (log: "regressor 15.059 dB"). Against its own posterior mean the flow costs **0.40 dB** (15.19→14.79), not 1.8 dB. Rewrite Sec. X accordingly, or re-run Exp 3 with `REG_DIR=None`.

### P0-7. Security
- Token already removed (§2). User must revoke `hf_CQef…` and clear outputs before publishing notebooks.

---

## 5. P1 — results the paper misreports (fix text/tables; some need re-runs)

| # | Paper claim | What the code/logs show | Action |
|---|---|---|---|
| 1 | Sec. VII-7: heteroscedastic 17.17 vs 17.45 (failed) | 17.17 is the step-15k gate value (cell 69); run finished **17.50** (> baseline), SSIM 0.2867 < 0.2905 | Report as null/marginal; it is a third "gate said FAIL" case for the gating note |
| 2 | Sec. VII-3: colour supervision → 14.0 dB | E1/E2 (cells 40–43) validate with **batch-MSE** PSNR; the no-colour control E2 also reads ~14.2. Both hardcode `latent_scale=1.4171` but `bridge_final` uses 1.4010 | Null result, or re-run with per-image PSNR + scale from config |
| 3 | Sec. IV-C / conclusion: FLUX 17.56 vs v3 18.32, "same recipe", "better AE → worse translator" | FLUX 17.56 is on the n=150 monitor set; same set: v3 17.67, v2 17.45. FLUX run (cell 74) has no MIND loss, no decoder FT, bs 6, lr 3e-5, init from v2 | Matched comparison; drop "worse translator" |
| 4 | Table IV / Sec. V: best shift +0.14 dB | Cropped vs uncropped mismatch; script's matched value **+0.08** (18.368→18.449) | Fix |
| 5 | Sec. VII-4: 18.31→14.52→12.60→9.92 | That chain is the **bridge**, from **16.26** (cell 45). Despeckle numbers (VII-5) are also the bridge | Fix; also note despeckle was inference-only on a model trained on speckle (distribution shift, not evidence "speckle is signal") |
| 6 | Table II caption n=15,394 for all rows | v1, v2, FLUX rows are n=150 monitor numbers | Re-evaluate on the saved split or relabel |
| 7 | Fig. 2 "largest jump from 16-ch latent" | Mixes 512 px (4-ch row) with 256 px; matched 512 px: 12.90→13.61 (+0.71) | Redo figure at one resolution/n |
| 8 | Sec. III-A, Fig. 1, abstract: SD encoder frozen; one-stage Eq. 2 | Two phases: cell 21 (512 px, frozen encoder, L1+LPIPS+align, 40k) then cell 25 (256 px, **everything unfrozen**, Eq. 2, 20k) | Describe both phases; fix Fig. 1 colouring |
| 9 | Sec. III-B: SD-1.5 U-Net reused as regressor | Initialised from trained `bridge_final`, then 30k (v1) + 40k (v2) + 80k (v3) steps | State it; the "beats the bridge" claim is continued training with extra losses + decoder FT |
| 10 | Sec. IX: Stage B on best Stage-A posterior means | `best.pt` 404 → cached from `last.pt` (step 60k, val 12.57) | State it, or re-cache from best |
| 11 | Sec. VII-6 rectified flow | Initialised from the ResShift bridge (x0-predictor) and validated with batch-MSE PSNR | State; soften mechanism claim |
| 12 | Sec. V: "registration error exists" | Mean signed best shift (+0.09, +0.28) px ≈ 0 | Remove/soften |
| 13 | Conclusion: "per-scene offset worth 2.27 dB cannot be predicted" | 2.27 is the 4×4 *grid* oracle; only the *global* affine predictability was tested (on v2) | Reword |

## 6. P2 — arithmetic and text errors (paper only)

- 30.97 − 27.40 = **3.57**, not 3.94 (3.94 was vs 27.02 on another 200-image sample; measure all Table I rows on one set). FLUX LPIPS 0.135 in Table I is untraceable.
- "Sampler pays about twice the MMSE, matches 1.6 dB": 2× MSE = **3.01 dB**.
- DoRA cost 2.21 dB = **0.94 dB more** than LoRA, not "2.22 dB more".
- Table II: bold SAM 6.27 but v2 = 6.02; published DDPM FID 116.9 identical to ACD-16 116.9 (verify source).
- Abstract says three contributions; intro lists seven.
- "Six terrain classes": only **3** occur (tropical, temperate, arctic; 10 season×terrain cells).
- Red note in Table II (FID 168.8 vs 128.2): 168.8 not found in any notebook; full-val runs gave 128.15 (cell 77) and 125.54 (cell 38). Report mean±sd over ≥3 sampler seeds.
- ERGAS (Sec. XI): values 1732/9377 in cell 77 show per-image band means blowing up on dark tiles, not just "doubled". Use dataset-level means or drop.
- "Decoder noise floor 0.000000" is trivial (deterministic decoder); DOIM ceiling check (cell 52) is trivially 1.0 for the same reason — don't cite either as evidence.

## 7. Numbers in the paper with no source in these notebooks

Table III (all rows), Table II row 1 (12.90/0.154/0.997; in-notebook 4-ch eval at 256 px = 12.49/0.120/0.986, cell 20), FID 168.8, FLUX LPIPS 0.135, Stage B curve (12.16/12.53/12.69), and every Table VI row except `curated` (code for pixel-space baseline, LoRA/DoRA/MiCA, distillation, INT8, TTA, weight soup is absent — `complete_pipeline` cell 35 says so). Either locate the source runs or re-run before submission.

## 8. Additions worth making

- **Constant-colour floor.** Benchmark cell 29 diagnostic: on unseen scenes Stage A = 12.53 dB vs **constant mean colour 12.34 dB** (+0.19), train scenes 17.40, perfect global colour shift +4.77 dB. Add a `Const` row to every table; this strongly supports the paper's thesis.
- Data audit of the main corpus: hash images to find exact/near duplicates across train/val (the check in cell 24 found location keys shared across seasons — 1,070 in all four, 7,209 summer∩winter — and within-season duplicate keys, e.g. winter 81,895 rows vs 77,823 unique).
- Calibration stats (cell 58): SSR uses mean of per-pixel std with `ddof=0` (biased low; use RMS spread, ddof=1); rank-histogram χ² pooled over ~300k correlated pixels (report as descriptive only); conformal exchangeability holds per image, not per pixel.
- Table VI CIs come from a bootstrap over only 8 test scenes (grassland: 108 images) — state the limitation or add scenes.
- Rotation TTA costs v3 1.33 dB (cell 80) because v1–v3 were trained with flips only; mention next to Sec. VIII-G.

## 9. Suggested order of work

1. P0-1 split reconstruction + verification gate → P0-2 decoder fix → re-run Table II / IV / V evaluations.
2. Copy `data_meta` + `fyp_runs`, fix `evaluate.py`, re-run Table VI and the leak audit (P0-3); relabel `sceneref` (P0-4).
3. Decide on Sec. X: re-run Exp 3 on original v3 or rewrite (P0-5, P0-6).
4. Locate or re-run untraceable numbers (§7).
5. Apply §5–§6 text/table corrections; add §8 items.
6. Keep a changelog in this file (append below) of every run: script, split file hash, n, numbers.

---

## 10. Running on Kaggle

The user runs on **Kaggle competition GPUs (RTX Pro 6000; Internet off)**. Kaggle copies of all three notebooks live in `kaggle/` (`*_kaggle.ipynb` plus `00_build_assets_kaggle.ipynb`). `kaggle/README.md` covers attaching inputs, the setup cell and the offline loader shim. Make the §4 fixes **in the Kaggle copies**: they are the notebooks that will actually be run. Their cell indices are the original index **+2**, because a header cell and the setup cell were inserted at the top.

## Changelog
- 2026-10-13 — audit completed; repo IDs → SotaSF; token → Colab Secret `HF_TOKEN`; 7 final checkpoint tags copied to `SotaSF/sar-to-optical-diffusion`.
- 2026-10-19 — Kaggle copies created in `kaggle/` (offline shim, asset-builder notebook, README); originals untouched.
- 2026-10-19 — `kaggle/freq_split_kaggle.ipynb` added (LP(v3)+HP(bridge) fusion vs pixel blends). Split rebuild failed its gate → notebook defaults to `PROTOCOL="relative"` (contaminated, comparison only).
