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
- Design choice to defend if asked: improvements are tested on a **pixel** U-Net, not the latent v3, because v3 is contaminated and the clean evidence shows the pixel network matches it at a fraction of the cost.

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
