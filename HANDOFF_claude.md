# Handoff for the next Claude session (SAR→optical project)

Last updated 2026-11-02. Read this first; deeper detail lives in `HANDOFF_improvements.md` (audit findings with cell references) and `memory-handoff.md` (another agent's 28 Oct consolidation; useful, but it predates and contradicts several of our findings, see §4).

## 1. Who and what
- The user works on Ali Musa Rizvi's SAR→optical paper (`SAR_paper_IEEE_latex.pdf`, IEEE JSTARS draft). Git identity `shehrozSF`; HF account `SotaSF`.
- Goal: accurate (clean) results and real improvements in colour/spatial fidelity, run on **Kaggle RTX Pro 6000, Internet OFF, ~11 h per session**.
- Repo: this folder, branch `main`, **no remote yet** (user asked to "set remote"; waiting for the URL; recommend private). History was scanned: no tokens committed. `.env` is git-ignored.

## 2. Files
| Path | What |
|---|---|
| `SAR-to-Optic.ipynb`, `complete_pipeline.ipynb`, `S2O_Complete_Benchmark.ipynb` | Original Colab notebooks (Ali's), saved outputs; repo IDs switched to `SotaSF/...`, token removed |
| `sars-attempt-1.ipynb` | User's Kaggle run of the freq-split test (results in §3) |
| `kaggle/*_kaggle.ipynb` | Offline Kaggle ports of the three originals + `freq_split_kaggle.ipynb` |
| `kaggle/00_build_assets_kaggle.ipynb` | Builds `sar-assets-<bundle>` Kaggle datasets (weights, wheels, torch-hub weights) with Internet on |
| `kaggle/src/s2o_lib.py`, `kaggle/src/test_s2o_lib.py` | Shared library + CPU smoke test for the new experiments |
| `kaggle/00_smoke_test_kaggle.ipynb`, `01_baselines_comparison_kaggle.ipynb`, `02_spatial_blocks_kaggle.ipynb` | New experiment notebooks, **generated** by `scripts/build_experiment_notebooks.py` (edit `kaggle/src/` + the builder, then rebuild; never hand-edit the generated notebooks) |
| `.env.sample` | Template: HF repo ids/tokens; the first cell of every notebook loads `.env` (also from `/kaggle/input/*/.env`) |
| `scripts/move_hf_config_to_env.py` | Someone else's script that introduced the `.env` loader |

Kaggle mounts: `/kaggle/input/datasets/<user>/<dataset>/...` (one level deeper than older layouts). Assets dataset: `/kaggle/input/datasets/shehrozfarooq/sar-assets-base/sar-assets/{repos,wheels,torch}`. Data: `/kaggle/input/datasets/shambac/augmented-sentinel-1-2` (row counts match the training logs: 72,019 / 59,764 / 94,202 / 81,895).

## 3. Established findings (all verified from code + saved outputs)
1. **Split bug, unrecoverable.** Training scripts `random.shuffle` before `train_test_split`; eval scripts don't → ~95% of "validation" images in Tables II/IV/V were training images. Rebuilding the training split on Kaggle gave v3 monitor PSNR 18.330 / 17.995 vs the logged 17.668 (exact rebuild would match to ±0.01): Colab's CSV listing order was never saved. **No clean numbers are possible from the existing checkpoints.**
2. **Decoder bug.** Table II evaluated v3 with `adapted16_final` instead of `regressor_v3_final/base_vae_tuned.pth`. With the right decoder (contaminated split, n=3000): v3 PSNR 18.38, SSIM 0.342, LPIPS 0.672, **FID 117.8** (Table II: 0.306 / 0.823 / 215.5). v3 then beats the bridge on PSNR, FID and LPIPS (bridge FID 138.5) → the paper's perception–distortion story does not hold for these models.
3. Frequency-split fusion LP(v3)+HP(bridge): **negative** (FID 140–153, worse than v3 alone). A λ=0.25 pixel blend is marginal (FID 115.5, −0.14 dB).
4. Misreported/invalid in the paper: heteroscedastic finished 17.50 (17.17 was a gate value); colour supervision 14.0 dB = batch-MSE artefact; FLUX compared on different image sets; shift oracle +0.08 not +0.14; 3.00 dB leakage measured on images in the patch model's training set (leak audit crashed); `sceneref` uses test-scene optical GT (oracle); class-prior chain is the bridge from 16.26; only 3 terrain classes; PMRF-latent built on v3+; several numbers have no source in any notebook.
5. Clean evidence: on unseen scenes models ≈12.5–13 dB vs **constant mean colour 12.34 dB**; train scenes 17.40; perfect colour shift +4.77 dB; 50M pixel U-Net ≈ 860M latent pipeline.

## 4. Where `memory-handoff.md` is out of date
It still treats Table II's FID 215.5, heteroscedastic 17.17, colour-supervision 14.0, FLUX 17.56 vs 18.32 and the +0.14 shift oracle as valid, and does not know about the split being unrecoverable or the decoder bug. It also references a newer benchmark notebook (with overlap-lookup audit outputs, Stage B to 40k, cache-label fault), an N2 notebook `S2O_ConvXLSTM_Mamba_600CU.ipynb`, and externally verified literature numbers. **The user confirmed (2026-11-02) all three are unavailable**: don't ask for them again; quote external numbers only from `memory-handoff.md` §8 and label them as taken from that document.

## 5. New experiment notebooks (built 2026-11-02, not yet run)
- **00 smoke test**: run first on Kaggle CPU; must print `ALL SMOKE TESTS PASSED`. It could not run locally (Windows Application Control blocks torch DLLs; no torch/numpy on this machine).
- **01 baselines** (one ~11 h session): constant colour, U-Net (L1+SSIM), Pix2Pix, CycleGAN (unpaired via previous-batch targets, no identity loss), conditional DDPM (DDIM 50), ResShift-style (pixel, T=15, κ=1.0). Time-budgeted training (`TimedRun`), one seed each + extra seeds for U-Net/Pix2Pix while time remains. Scene-disjoint split `s2o_split_scene_v1.json` (MD5-ordered scenes, 80/10/10 per season, digest), exact-duplicate audit, per-image metrics incl. ΔE, LF-RMSE, GradErr, scene-bootstrap CIs.
- **02 spatial blocks**: Part 0 fixes + optional v3 reference with both decoders (EXPOSED); pixel U-Net base (SAR + season FiLM) 150 min; variants base/conv/convlstm/convxlstm/mamba/spade fine-tuned from the same base for 10k steps each (zero-init residual block at decoder level 2, identity checked in code); `PHASE="screen"` then `"confirm"` (top-2 by **validation** PSNR + base + conv, seeds 1, 2).
- `.env` loader fix (2026-11-02): the old loader only searched `/kaggle/input/*/.env` and missed the new `datasets/<user>/<dataset>/` mount (`sars-attempt-1` printed "no .env found"). The builder now also searches `datasets/*/*/.env`, and 02 defaults `HF_REPO_ID` to `SotaSF/sar-to-optical-diffusion` (served offline from `sar-assets-base`), so no `.env` dataset is needed for 00/01/02.
- `PHASE="all"` (added 2026-11-03): screen + confirm in one session, fine-tuning cut to 5k steps / 25 min per variant (`FT_STEPS_ALL`) so 14 fine-tunes + 150 min base fit in ~10.5 h. Added because RTX Pro 6000 access may end. 02 does not depend on 01: the split is deterministic and rebuilt; 01's output only skips the duplicate audit and verifies the digest.
- Design choice to defend if asked: improvements are tested on a **pixel** U-Net, not the latent v3, because v3 is contaminated and the clean evidence shows the pixel network matches it at a fraction of the cost.

## 5a. Results: 01 baselines (`kaggle-competition/baselines-01.ipynb`, 2026-11-03, ~10 h)
Same split and same 3000 test images (23 scenes) as 02. PSNR (scene CI), Δ vs U-Net:
- U-Net SAR-only **14.56** (13.88–15.25), 3 seeds 14.56/14.45/14.44, FID 238/317/325 (FID unstable across seeds).
- Pix2Pix 13.85, Δ −0.71 [−1.49, −0.07]; **constant colour 13.70**, Δ −0.86 [−1.60, −0.24]; ResShift-style 12.29 (final-EMA selection; val peaked ~22k then fell); CycleGAN 11.18 but best FID 106 (SSIM 0.04); DDPM 8.53, SSIM ≈ 0 = failed run.
- 02's base (SAR + season FiLM, otherwise same backbone/loss/EMA/selection) = 13.57 on the same images: ~1 dB below 01's U-Net and below constant colour; season helped on val (13.76 vs ~13.58) but hurt on test. Likely cause, unproven; the 02 block comparison sits on this weak base.
- Report for Ali: `reports/report_for_ali_2026-11-03.txt`.

## 5b. Results: 02 screen (`kaggle-competition/spatialpart1.ipynb`, 2026-11-03, RTX Pro 6000)
- Split verified against 01's output: digest `b26d0fd165ee9e1f`, 248 scenes, train 218,692 pairs / 201 scenes, test 3000 imgs / 23 unseen scenes, 0 duplicates excluded.
- v3 decoder check (EXPOSED, n=1000): adapted16 decoder PSNR 16.80 / SSIM 0.227 / LPIPS 0.887 / FID 280.3 vs tuned decoder 16.92 / 0.264 / 0.703 / **163.1**. Confirms the decoder bug on new images. This cell took ~90 min (model loading), so set `RUN_V3_REFERENCE=False` on reruns.
- Base U-Net: val PSNR peaks 13.657 at step 8k, then falls to 13.21 by 31.6k (overfits the 201 training scenes); best checkpoint kept. Speed 0.165 s/step once data is cached (0.43 s/step for the first ~14k steps).
- Variants (10k fine-tune steps, seed 0): **every variant's val PSNR peaks at step 1000 and declines after**, so `select="best"` returns ≈ base + 1k steps for all of them. Test PSNR 13.54–13.57 (scene CI ≈ ±0.7); paired Δ vs continued base all within ±0.03 dB, all CIs cross 0; convlstm ≈ convxlstm to 3 decimals (same init, both picked at 1k). Selected for confirm: spade, convxlstm (val +0.015 dB = noise).
- Interpretation: null at screen; fine-tuning from an already-overfitting base can't test the blocks (they barely train before selection). A from-scratch or frozen-base design would be the informative follow-up.

## 6. Open items / next actions
1. User runs 00 → 01 → 02 (attach 01's output to 02 for the shared split). Expect clean PSNR ≈ 13 dB; that is the point, not a bug.
2. Set the git remote once the user gives the URL; first push `main`.
3. Message to Ali (artifact https://claude.ai/artifact/WE8xu68vZ9VonhSs7wbo1u) does **not** yet include the decoder/FID finding; offer to update it.
4. Paper rewrite needs the LaTeX source (not in the folder).
5. The old HF token `hf_CQef…` must be revoked by the owner (unconfirmed).

## 7. Working notes for this machine
- Use `python3 -I -X utf8 …` (`-I` ignores `PYTHONIOENCODING`, so `-X utf8` is needed for non-ASCII output). There is no `python` on PATH in Bash.
- Some Bash commands with many chained `head`/`sed` get permission-denied; split them or use the Read/Grep tools.
- Notebook JSON: originals are compact (`json.dumps(nb, ensure_ascii=False, separators=(',',':'))` round-trips byte-identically); generated notebooks use `indent=1`.
- Scratchpad generators (`convert.py`, `kaggle_setup_cell.py`, `make_freq_nb.py`) live in the session scratchpad and may be gone; the source of truth for the setup cell is now cell 2 of `kaggle/freq_split_kaggle.ipynb` (the builder copies it from there).
