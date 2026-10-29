# SAR-Optic — Research Memory and Agent Handoff

**Last consolidated:** 28 October 2026 (Asia/Karachi).  
**Owner:** Ali Musa Rizvi.  
**Scope:** SAR-to-optical image translation (S2O), Sentinel-1 → Sentinel-2.  
**Purpose:** Preserve the accessible project research, reported experiments, negative results, hypotheses, cross-domain reasoning, limitations, and actionable next steps so a future agent can continue without rebuilding the project history.

> This is a detailed consolidation of the accessible shared history, the 11-page project manuscript, and two project notebooks. It is not a claim that every past chat, experiment artifact, or remote checkpoint was recovered. Research results below were read from sources; no new training or metric reproduction was performed during this consolidation. Unknowns are explicit. Recommendations introduced here are labelled as proposed work.

## 1. Read this first

1. The user wants measurable progress toward strong/SOTA S2O results, particularly better colour and spatial appearance while retaining scene fidelity. A defensible benchmark and useful diagnostic paper remain valuable if architectural gains do not materialize.
2. This project is separate from the user's fetal-ultrasound/FYP work. Do not import those datasets, H1–H4 definitions, architectures, or clinical objectives.
3. The headline historical result is **v3: 18.32 dB PSNR, 0.306 SSIM**, evaluated on **15,394 validation images from a per-patch split**. It is not a strict unseen-geography result.
4. A 16-channel residual-shifting bridge has worse fidelity but better reported perceptual metrics. Its **FID is unresolved: 128.2 in the manuscript versus 168.8 in a repository evaluation file quoted by the manuscript**. Do not silently select one.
5. A scene-disjoint smaller-corpus comparison puts fidelity near 13 dB. A **49.9 M pixel U-Net** matches the **859.6 M latent system** on PSNR and improves several perceptual metrics. This supports prioritizing data/protocol/generalization work over simply enlarging the backbone.
6. Oracle colour corrections show meaningful headroom: **+1.20 dB global affine; +2.27 dB spatial 4×4 affine** on 400 images. They use the target and are diagnostics, not deployable baselines. A failed affine predictor does **not** establish that all such information is impossible to predict.
7. Never compare SEN1-2, SEN12MS, BigEarthNet, and cloud-removal scores as one leaderboard. Splits, RGB rendering, inputs, sample counts, label availability, and metric implementations differ.
8. The budgeted ConvLSTM/ConvXLSTM/Mamba notebook has reported synthetic software verification, **not real-data accuracy gains**. The complete benchmark notebook does contain saved real-run outputs for some earlier experiments; do not collapse these two statuses.
9. Prior user constraint: approximately **600 Colab compute units**, avoid costly custom dependency builds, and reuse relevant trained weights. Current balance and runtime pricing are unknown. Never assume the historic 5.4 CU/hour estimate is current.
10. Before spending compute, fix checkpoint provenance, split identity, evaluation inconsistencies, and hard-coded credentials. Existing v3 weights have exposure to patches later labelled as test scenes; they cannot establish a fresh scene-disjoint claim.

## 2. Evidence levels and provenance

Use these labels when updating this memory:

| Label | Meaning |
|---|---|
| `PAPER-REPORTED` | Appears in the user's manuscript; not independently rerun here |
| `NOTEBOOK-OBSERVED` | Present in saved notebook source/output; an output is historical evidence, not fresh execution |
| `HISTORY-RECALLED` | Retrieved from shared conversation memory; may be incomplete |
| `EXTERNAL-VERIFIED` | Checked against an original external paper or official implementation during this consolidation |
| `CODE-ONLY` | Implemented or planned in source, without recovered real-run evidence |
| `PROPOSED` | Recommended future work or a new audit inference in this handoff |
| `UNRESOLVED` | Missing source, conflicting number, unclear provenance, or incomplete result |

### 2.1 Accessible primary project sources

| ID | Source | What it establishes |
|---|---|---|
| P1 | [SAR_paper_IEEE_latex.pdf](https://chatgpt.com/library?file_id=libfile_417bf7119e1c81918f9e8001d958887a&account_id=personal) | 11-page manuscript, methods, Tables I–VII, oracles, calibration, negative results, limitations |
| N1 | [S2O_Complete_Benchmark.ipynb](https://chatgpt.com/library?file_id=libfile_a83d91458a1481919d374af6211f83ad&account_id=personal) | Complete benchmark implementation, audit, saved A/B training outputs, v3 continuation/latent-flow outputs |
| N2 | [S2O_ConvXLSTM_Mamba_600CU.ipynb](https://chatgpt.com/library?file_id=libfile_6618dbac2b58819180f284d8b4e787e1&account_id=personal) | Controlled budgeted decoder experiments, configuration, safeguards, documented synthetic verification |
| H1 | Shared project conversation, 14–16 October 2026 | User budget/workflow constraints, architecture requests and notebook delivery context |
| H2 | Shared project conversation, 26 October 2026 | Colour/spatial gap, cautious oracle interpretation, Aydin/OSCAR cross-paper comparison |

The project source attachment list in this conversation was empty. The sources above were recovered separately. No existing `memory.md` was resolved by the searches performed. A CV was also surfaced by retrieval; its approximately 250 A100-hours / 12+ gated experiments and public-release descriptions are secondary summaries, not a substitute for original logs.

### 2.2 Project identity and publication status

- Working project name: **SAR-Optic / S2O**.
- Manuscript title in P1: **“Diffusability-Aware Latent Adaptation for SAR-to-Optical Translation: An Oracle Decomposition of the SEN1-2 Error Budget.”**
- A secondary CV uses **“SAR-to-Optical Image Translation via Diffusability-Aware Latent Adaptation.”** Treat this as a title variant, not a second project.
- Historical target venues: IEEE JSTARS / MDPI Remote Sensing. Submission/acceptance status is not established.
- Referenced model repository: [AliMusaRizvi/sar-to-optical-diffusion](https://huggingface.co/AliMusaRizvi/sar-to-optical-diffusion). P1 and N1 refer to released checkpoints and evaluation code. The full current remote artifact tree was not audited here.

## 3. Problem formulation and what success means

Given SAR observation Y, predict an optical-like RGB image X. Radar backscatter responds to geometry, roughness, moisture/dielectric properties, and acquisition configuration; optical RGB also depends on reflectance, atmosphere, illumination, vegetation state and timing. A single SAR patch does not uniquely identify its paired optical image.

There are three distinct objectives:

1. **Fidelity:** match the paired reference on PSNR, SSIM, colour/spectral-angle and spatial metrics.
2. **Perception:** produce realistic optical-like images, measured by distributional/perceptual metrics and fixed qualitative examples.
3. **Utility and reliability:** retain information useful for downstream tasks and express ambiguity honestly.

The project repeatedly observes a fidelity–perception trade-off. A better FID does not prove that an individual output has the right objects, boundaries, or colours. A higher PSNR does not guarantee optical realism or useful uncertainty.

### 3.1 Theoretical guardrails

- Under expected squared error, the Bayes-optimal deterministic estimator is the conditional mean m(Y)=E[X|Y]. A finite trained MSE network only approximates it.
- For an exact independent posterior sample X' conditional on Y, E[||X−X'||²|Y]=2 tr Var(X|Y), while the exact conditional mean has MSE tr Var(X|Y). This is a theoretical comparison, not proof that the trained bridge is an exact posterior sampler.
- Expected MSE optimality does not imply exact optimality for mean per-image PSNR, which contains a logarithm. Nor does it prove that v3, trained with mixed losses, is the posterior mean.
- PMRF is motivated by minimum-distortion restoration under a perception constraint. Its idealized theory is not a guarantee that this finite, conditioned, imperfectly trained implementation will dominate every control.
- An oracle correction measures headroom within the correction family tested. Oracle improvements are not additive, and no oracle in this project certifies a universal model-performance ceiling.
- Failure of one affine head, one loss schedule, one backbone or one data budget cannot establish information-theoretic impossibility.

## 4. Data, splits, rendering and leakage

### 4.1 Keep the corpora separate

| Corpus / experiment | Size / split | Purpose and limitation |
|---|---|---|
| Large augmented corpus in P1 §§III–VII | 307,880 paired 256×256 images; 292,486 train / 15,394 validation; four seasons × six terrain classes; fixed stratified per-patch split | Historical headline numbers. Source-scene overlap not audited on this exact split. Why the augmented count differs from the official release requires lineage documentation. |
| Smaller terrain corpus in P1 §VIII | 16,000 pairs, four land-cover classes, 30 source scenes; 16/6/8 scenes train/val/test; test n=1,657 | Controlled scene-level ablations. Much smaller training distribution; absolute scores cannot be subtracted from the headline result to estimate leakage. |
| Official SEN1-2 release in P1 §IX | 282,384 pairs, four seasonal ROI sets; deterministic hash selection of whole scenes; 80/10/10 by scene within season | Pixel Stage A + PMRF experiment. Source-scene separation is stronger than patch separation but does not itself certify geographic independence across dates/ROIs. |
| Saved official-subset run in N1 | 95,649 retained pairs; train 73,284 / val 10,935 / test 11,430 in an appended v3 diagnostic output | Actual historical manifest counts. Training/source exposure differs by model. Do not infer scene count from the logged `S=256`; inspect the metadata definition. |
| N2 budgeted experiments | Reuses actual ingested dataset and manifests; default inherited data configuration must be inspected at execution | Setting `scene_frac` after ingestion does not resize existing arrays. New corpus → new work directory/manifests. |

Sources: official [SEN1-2 release](https://mediatum.ub.tum.de/1436631); smaller terrain corpus [Kaggle source](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain). Availability, licensing and checksums should be checked before a new run.

### 4.2 Scene context reconstruction

- Official patches have 50% overlap: 256-pixel patches with stride 128.
- N1 reconstructs SAR scene mosaics using overlapping SAR observations and extracts a **512×512 SAR context**, downsampled to **128×128**.
- Optical images must not be used to form inference context. Targets are permitted for training losses/teacher supervision on the training split only.
- Validate overlap consistency, border handling, missing neighbors, valid-context masks, scene keys, and absence of cross-split optical lookups.
- Extra neighboring SAR context is a distinct input condition. Compare patch-only and context-assisted methods explicitly; a gain over patch-only competitors may reflect extra information.
- If the end deployment has only isolated SAR tiles, context-assisted evaluation represents a different input assumption.

### 4.3 Strong saved leakage diagnostic — N1, not a translation model

N1 implements a no-SAR/no-training lookup that copies overlapping **training optical pixels** onto a test footprint. Historical output reports:

| Diagnostic | Exact-copy pixel coverage | Mean per-image PSNR | Pooled PSNR | SSIM | FID |
|---|---:|---:|---:|---:|---:|
| Random per-patch split lookup | 98.0% | 94.44 dB, exact matches capped at 100 dB | 32.24 | 0.986 | 1.6 |
| Scene-disjoint lookup → train-mean colour | 0.0% | 13.10 | 12.22 | 0.289 | 488.6 |

The 94.44 dB value is dominated by exact-copy handling and the PSNR cap. It is neither a learned S2O score nor a sensible number to compare directly with normal model PSNR. The result demonstrates a leakage mechanism in this particular split/lookup setup, not proof that every historical S2O paper has identical contamination.

N1 reports a separate per-patch split of 76,519/9,564/9,566 images, and a fixed 4,096-image test evaluation drawn from 11 scenes. These are run-specific counts, not the P1 headline validation set.

### 4.4 Rendering and metric protocol

- Document RGB bands/order, SAR polarization and units, source PNG generation, clipping/stretch/gamma, dtype and normalization.
- A fixed reflectance divisor, per-image percentile stretch and histogram equalization change the target and can mechanically alter PSNR. Do not treat them as neutral display changes.
- Historical convention: image-wise PSNR/SSIM on 8-bit RGB, then mean across images. Taking the logarithm after pooling batch MSE is a different statistic; an early experiment was invalidated by about a 2 dB discrepancy from this mistake.
- Keep 0–255, 0–1 and −1–1 ranges explicit throughout training and evaluation.
- Label SAM units; the P1 tables use degrees. Do not compare external SAM values without checking units and implementation.
- ERGAS had an acknowledged scale/unit bug; corrected Table VI values do not repair other outputs automatically.

## 5. Architecture history

### 5.1 Adapted 16-channel autoencoder — P1

- Start with SD-1.5's four-channel, spatial-stride-8 autoencoder.
- Freeze the base encoder; add a trainable **12-channel detail encoder** with three strided convolutions, group normalization and SiLU.
- Concatenate base and detail representations into **32×32×16** at 256×256 input.
- Widen the decoder input convolution 4→16. New-channel weights start at zero, preserving the original decoder function at initialization.
- Loss reported:

  `L_VAE = L1 + LPIPS + 1e-6 KL + 0.5 L_align + 0.25 L_SE + 0.1 L_adv`.

- Alignment constrains grouped detail channels against the base representation. Scale equivariance compares decoding downsampled latents with downsampled images at scales {2,4}. KL is small/annealed; adversarial PatchGAN begins after 1,000 steps.
- P1 reports a DCT-spectrum analysis supporting reduced high-frequency latent energy. A matched factorial ablation is still needed before attributing every downstream gain specifically to scale equivariance.

### 5.2 Deterministic latent regressor v1/v2/v3

- Reuse the SD-1.5 U-Net as a direct latent translator.
- Input convolution: 32 channels, receiving `[z_sar || z_sar]`; output: 16 channels.
- Fixed timestep embedding **t=933**; no inference sampling; **NFE=1**.
- Season×terrain text conditioning enters through CLIP cross-attention.
- v3 loss:

  `min_(|dx|,|dy|<=1) ||z_pred − T_dx,dy(z_opt)||² + 0.5 L1 + 0.3(1−SSIM) + 0.08 MIND + 0.05 SAM`.

- Shift is cyclic at latent resolution; one latent cell corresponds to eight image pixels. Audit boundary artifacts and whether loss alignment hides displaced details.
- MIND compares the decoded prediction with unfiltered SAR. It is a modality-independent descriptor borrowed from medical image registration.
- Pixel terms ramp from steps 2,000 to 5,000. Decoder is jointly fine-tuned at a lower learning rate to accommodate predicted latents, which differ from encoded target latents.
- Regressor v1/v2 exact change-by-change configurations were not fully recovered; do not invent them from their score progression.

### 5.3 Residual-shifting bridge

- Same adapted latent, residual-shifting diffusion inspired by ResShift.
- **T=15, kappa=2.0**, geometric eta schedule **0.04→0.99**.
- CLIP season×terrain prompts; 10% conditioning dropout for classifier-free guidance.
- Perceptual operating point rather than fidelity winner.
- One-step use fails; NFE and resolution must accompany each result.

### 5.4 FLUX autoencoder experiment

FLUX.1-schnell's 16-channel codec improves encode/decode PSNR but yields a worse translator under the tried recipe. This is evidence against that particular swap, not proof that all codec changes are irrelevant. Codec scaling, latent statistics, initialization and optimization remain potential confounders.

### 5.5 Pixel Stage A + PMRF Stage B

**Stage A:** 53.5 M context/season-conditioned pixel U-Net. Season FiLM; convolutional context encoder supplies a global modulation vector and a bottleneck map. Pure MSE estimates a conditional-mean starting point. No VAE bottleneck.

**Stage B:** approximately 92 M pixel U-Net, inputs `[Z_t, pm, SAR]`. Start `Z_0 = pm + 0.10 epsilon`; interpolate `Z_t = t X + (1−t) Z_0`. Predict clean X with velocity-equivalent loss:

`L_B = ||g([Z_t,pm,SAR],t) − X||² / max(1−t,0.05)²`.

Sample t from a half uniform / half logit-normal mixture. Euler update:

`Z_(t+dt) = Z_t + dt/(1−t) * (g − Z_t)`.

At the last step this lands on the predicted clean image; K=1 is mathematically defined, not automatically accurate. Both stages historically use AdamW, cosine decay, EMA 0.9999, bf16, batch 32, one A100.

Out-of-fold posterior-mean caches (`B_oof`) address the fact that Stage A fits training scenes better than unseen scenes. N1 implements this option; a completed real result was not recovered.

### 5.6 Continued latent regression and latent PMRF

- v3+: continue original mixed-loss recipe, U-Net LR 1e-5, decoder LR 5e-6, effective batch 8, EMA 0.9995. Gate at 20k requires +0.15 dB paired PSNR over v3.
- Latent PMRF: initialize a second U-Net from regressor weights; inputs `[x_t, SAR latent, predicted optical latent]`; start from regressor output + sigma 0.1 noise; train velocity; NFE {8,16}; latent blends lambda {0.25,0.5,0.75}.
- **Important N1 lineage:** appended run selected `v3plus_reg/best` if that directory existed, even after v3+ failed its gate. The recorded latent flow therefore starts from a degraded continued regressor. It does not cleanly isolate flow-on-original-v3.

## 6. Historical numerical results

All tables in this section are P1 unless explicitly marked N1. `n/a` means not recorded, not zero. Keep table protocols attached when quoting them.

### 6.1 Autoencoder round-trip only — 256 px, no translation

| Codec | PSNR | LPIPS | Note |
|---|---:|---:|---|
| Frozen SD-1.5, 4 channels | 25.23 | 0.120 | Baseline |
| Adapted 16 channels | 27.40 | 0.061 | +2.17 dB |
| FLUX.1-schnell, 16 channels | 30.97 | 0.135 | SSIM 0.865; better PSNR does not mean better LPIPS |

### 6.2 Main large-corpus validation — n=15,394

| Method | PSNR | SSIM | LPIPS | FID | SAM degrees | CC | NFE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4-channel bridge, 512 px | 12.90 | 0.154 | 0.997 | n/a | n/a | n/a | 15 |
| Adapted 16-channel bridge, 256 px | 16.69 | 0.257 | 0.768 | **128.2 disputed** | 7.59 | 0.265 | 15 |
| Deterministic v1 | 17.06 | 0.285 | n/a | n/a | n/a | n/a | 1 |
| Deterministic v2 | 17.45 | 0.291 | n/a | n/a | 6.02 | n/a | 1 |
| Deterministic v3 | **18.32** | **0.306** | 0.823 | 215.5 | 6.27 | 0.390 | 1 |
| FLUX-latent regressor | 17.56 | 0.283 | n/a | n/a | n/a | n/a | 1 |

The manuscript explicitly quotes a bridge repository evaluation of **PSNR 16.66, SSIM 0.256, LPIPS 0.769, FID 168.8**. Reconcile checkpoint, seed, sample set, image preprocessing and FID implementation before reporting 128.2 as canonical. Small bridge PSNR differences 16.64/16.66/16.69 also require configuration identity.

The 4-channel/16-channel bridge rows differ in resolution as well as codec. Their large score difference is not a pure controlled estimate of channel adaptation alone.

### 6.3 Bridge steps, season and resolution

| Setting | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| NFE 1 | 10.02 | 0.034 | 0.758 |
| NFE 5 | 16.87 | 0.256 | 0.777 |
| NFE 15 | 16.64 | 0.257 | 0.769 |
| Spring | 15.69 | 0.223 | 0.792 |
| Summer | 17.34 | 0.280 | 0.745 |
| Fall | 17.38 | 0.285 | 0.753 |
| Winter | 16.22 | 0.235 | 0.783 |
| 512 px inference | 13.61 | 0.191 | 0.799 |

More sampling steps and larger inference resolution did not automatically improve fidelity. The NFE=1 LPIPS result illustrates why a single metric can be misleading.

### 6.4 Oracle error decomposition — v3, n=400

| Target-assisted correction | PSNR | Gain |
|---|---:|---:|
| None | 18.31 | 0 |
| Best global translation within ±4 px | 18.45 | +0.14 |
| Per-image, per-channel affine | 19.51 | +1.20 |
| Spatial 4×4 grid affine | 20.57 | +2.27 |
| Best Gaussian blur | 18.33 | +0.02 |
| Best unsharp mask | 18.31 | +0.00 |

- Mean absolute selected shifts: (1.72,1.92) px.
- Replacing predicted low frequencies with target low frequencies at sigma 2.5 gives **+7.06 dB**; replacing high frequencies gives **+2.19 dB**. These are different target-assisted constructions, not a partition that can be added to the affine gains.
- Global affine head was trained against an older v2 oracle gain of **1.40 dB**, not v3's 1.20 dB. It captured **−1.0%** of available gain in that experiment.
- Safe conclusion: coarse radiometry is a major residual component for this model/subset. Small gains from blur/sharpen or rigid shifts do not prove correct local geometry or optimal detail placement.
- The manuscript's wording “cannot be predicted” / “noise floor” is stronger than this evidence supports. Preserve the negative result and weaken the universal inference.

### 6.5 Ensemble verification / calibration

K=16 stochastic draws; calibration/test sets of 150 images each. Diversity check on 30 images: per-pixel ensemble SD **0.098**, deterministic decoder noise floor **0.000000**, mean pairwise LPIPS **0.218**.

| Metric | Original bridge | Almost-fair CRPS fine-tune |
|---|---:|---:|
| fair CRPS ↓ | 0.1029 | 0.0972 |
| Spread–skill ratio, ideal 1 | 0.291 | 1.206 |
| Rank histogram chi-square ↓ | 565,442 | 39,516 |
| AUSE ↓ | 0.0399 | 0.0459 |
| Spearman(sigma,error) ↑ | 0.369 | 0.338 |
| Conformal multiplier q | 5.64 | 1.30 |
| Coverage, target 90% | 90.1% | 89.9% |
| Single-sample PSNR | 16.5 | 10.4 |

Interpretation: native spread is insufficient; CRPS optimization improves dispersion/scoring but harms single-sample fidelity and uncertainty ranking. Coverage and discrimination are separate. Conformal coverage requires the relevant exchangeability assumptions; spatially related pixels are not independent calibration examples. No across-domain coverage guarantee was demonstrated.

### 6.6 Smaller-corpus scene-disjoint test — n=1,657

PSNR uncertainty below is the half-width of a scene-bootstrap 95% CI. Point estimates and overlapping individual CIs do not substitute for paired difference CIs.

| Model | PSNR ± CI half-width | SSIM | LPIPS | FID | SAM deg | ERGAS | CC | Params M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 recipe on this corpus | 13.16 ±1.14 | .129 | .829 | 311.0 | 7.24 | 61.27 | .110 | 859.6 |
| Dual-SAR | 13.17 ±.94 | .130 | .699 | 243.9 | 7.26 | 61.86 | .136 | 859.6 |
| Wavelet low-frequency loss | 13.33 ±.98 | .135 | .721 | 252.5 | 7.46 | 60.49 | .151 | 859.6 |
| Scene-reference colour | 13.45 ±.61 | .121 | 1.015 | 365.7 | 6.44 | 58.95 | .094 | 859.6 |
| ResShift bridge | 12.76 ±.75 | .086 | .766 | 179.9 | 8.65 | 64.39 | .049 | 859.6 |
| Curated training, 3% removed | 13.13 ±.96 | .136 | .694 | 242.1 | 7.34 | 61.21 | .176 | 859.6 |
| Perceptual LoRA | 11.89 ±.89 | .072 | .709 | 303.2 | 9.92 | 70.72 | .006 | 859.6 |
| Perceptual DoRA | 10.95 ±.91 | .071 | .687 | 287.5 | 10.86 | 78.73 | .051 | 859.6 |
| Perceptual MiCA | 11.83 ±.65 | .041 | .712 | 305.6 | 9.52 | 71.81 | .049 | 859.6 |
| Distilled half-width/depth student | 13.28 ±.96 | .134 | .860 | 325.9 | 7.54 | 60.16 | .129 | 148.1 |
| Pixel-space baseline | 13.31 ±.98 | .130 | .638 | 190.9 | 7.52 | 60.42 | .215 | 49.9 |
| Per-patch split, diagnostic only | 16.16 ±.79 | .232 | .720 | 235.3 | 4.92 | 44.17 | .509 | 859.6 |

- Reported patch-split inflation: **+3.00 dB, 95% CI [2.06,4.04]**, SSIM approximately +0.10. Do not apply exactly 3.00 dB to the separate large corpus or call it a proven lower bound there.
- Dual-SAR, low-frequency loss and curation effects on PSNR are below 0.2 dB and paired intervals cross zero.
- Scene-reference colour has nominal best PSNR but worst FID/LPIPS; reference construction and optical availability must be audited before treating it as a deployable SAR-only baseline. P1's suspected block artifact was not visually confirmed.
- Student-vs-teacher PSNR difference CI: **[−0.17,+0.31] dB**. Distillation offers efficiency with somewhat worse perception.
- LoRA/DoRA/MiCA used one untuned configuration and 6k steps versus 15k for the core recipe. This is not a fair general ranking of the methods. MiCA's original evidence concerns language-model knowledge injection, not this exact U-Net task.
- Eight-way dihedral TTA and last/best weight soup change PSNR <0.05 dB. Simulated INT8 weight-only quantization costs 0.01 dB; this is not a measured deployed INT8 speedup.
- A bridge-blend/posterior-mean sweep and fp32/bf16 latency comparison used a biased fixed-position test subset scoring >2 dB below the full test set. Their quantitative conclusions were withheld pending random-sample reruns.

### 6.7 Official-release pixel A/B: paper versus later saved outputs

P1: Stage A peaks at **12.94 dB around step 10k**, falls to **12.74 at 25k**, despite training MSE falling 0.143→0.115. Pixel PMRF improves **12.16 at 5k → 12.53 at 35k**, versus cached mean **12.69**; no recovered test-scene FID/LPIPS for this study.

N1 later saved output:

- A_main resumed at 35,543 and reached **60,000 steps**; final validation **12.5747**, logged best **12.938**.
- Diagnostic on last weights: val **12.529**, seen-training-scenes **17.400**, constant mean **12.338**, oracle colour-shift val **17.294**. Train/val gap **4.87 dB**, colour oracle gain **4.77 dB**, model-over-constant only **0.19 dB** on that diagnostic subset.
- Per-scene val range **10.32–15.17 dB** over ten scenes.
- **Checkpoint fault:** missing remote/local `ckpt/A_main/best.pt` caused a fallback to `last.pt`. Cache output then stated `source A_main/best @ step 60000`. Actual identity and label disagree.
- Cache: 95,649 patches, about 425 s, approximately 18.8 GB; OOF cache temporarily requires another 18.8 GB in the logged run.
- B_main resumed at 37,288 toward planned 80k. Recovered validation at 40k: **12.4367**, cached mean **12.5744**. Completion to 80k is not established by the recovered outputs.

Do not replace the earlier paper curves with later outputs without identifying the exact checkpoint/cache/configuration. The fallback creates a material confound in PMRF lineage.

### 6.8 v3+, latent PMRF and blends — 4,096 shared test-list patches

These rows compare methods within the same diagnostic list. Original v3 was trained on a per-patch corpus containing approximately 95% of the same patches according to N1, so “test scenes” does not make its weights cleanly scene-disjoint.

| Method | PSNR | SSIM | LPIPS | FID | SAM deg | CC |
|---|---:|---:|---:|---:|---:|---:|
| Original v3 | 16.60 | .253 | .706 | 142.5 | 7.26 | .372 |
| Continued v3+ | 15.19 | .192 | .703 | 132.8 | 8.04 | .342 |
| Latent PMRF NFE8 | 14.79 | .159 | .667 | 208.7 | 8.34 | .276 |
| Latent PMRF NFE16 | 14.44 | .145 | .673 | 205.5 | 8.53 | .250 |
| Blend lambda .25 | 15.15 | .188 | .664 | 130.3 | 8.08 | .330 |
| Blend lambda .50 | 15.01 | .176 | .651 | 153.6 | 8.15 | .305 |
| Blend lambda .75 | 14.77 | .161 | .660 | 181.6 | 8.29 | .276 |

- N1 paired v3+−v3 difference: **−1.407 ±0.046 dB** on 4,096 patches. The appended uncertainty should not be presented as scene-cluster uncertainty without checking the bootstrap implementation.
- v3+ planned 120k but stopped at **20k**: gate observed best-current RAW/EMA gain **−1.105 dB**, required +0.15.
- Latent PMRF planned 40k but stopped at **10k**: required PSNR drop ≤2 dB and LPIPS gain ≥0.03 relative to its regressor; gate failed. This is an early-stopped result, not a converged model.
- `best` in the appended v3 continuation code chooses RAW versus EMA **at finalization**, not the best historical checkpoint. Earlier EMA monitor PSNR **15.223 at 5k** exceeded saved EMA **14.962 at 20k**. Audit/repair selection before interpreting this as best-of-training v3+.
- In 600 validation images, empty/null prompt **16.021 dB** exceeded season+“temperate” **15.530**. Prompt selection is a validation choice, not proof that metadata never helps. Original v3's own logged held-out value **17.668** is another historical checkpoint/protocol figure, not a replacement for the final 18.32 result.
- N1 raw ERGAS in this appended output: approximately 105,137,143,149,137,140,143 by row. These were not in P1 Table VII and should remain quarantined until the known unit bug is audited.
- Qualitatively, P1 shows a dark/desaturated brown tendency; greener spring targets remain poorly matched. Latent flow adds texture without reliably placing it correctly. Four displayed tiles are illustrative, not an estimate of aggregate gain.

## 7. Negative-result ledger: preserve the scope of each failure

| Intervention | Historical result | Appropriate inference / next audit |
|---|---|---|
| ControlNet-style conditioning change | PSNR fell to 10.5 | Architecture surgery disrupted initialization; not evidence against all ControlNet methods |
| VGG/SSIM loss on stochastic bridge | −1.7 dB PSNR | Objective conflict in this run; SSIM helped deterministic v3 |
| Low-frequency colour supervision | 14.0 dB, no SAM improvement | That loss/schedule failed; audit normalization, weighting and conditioning |
| Class-prior calibration, stronger blends | 18.31→14.52→12.60→9.92 | Coarse class means can be harmful; per-scene variation matters |
| Input despeckling | −1.8 dB; CC .230→.105 | Distribution shift or lost useful signal in that pipeline; not a universal ban on filters |
| Direct rectified flow matching | Stopped at 60k; loss .005–.015 but val 9.9–11.5 | Training loss did not track source-endpoint sampling quality |
| Heteroscedastic regression | 17.17 vs 17.45 | One confidence-weighting run failed; uncertainty calibration and point prediction differ |
| SAR-conditioned affine head | −1% oracle capture | Head/features/protocol failed; no impossibility proof |
| FLUX codec swap | 17.56 vs 18.32 translation | Better round-trip codec does not guarantee learnable/predictable latents |
| Perceptual PEFT | See Table VI | Short untuned LoRA/DoRA/MiCA runs do not establish broad method rankings |
| More steps on Stage A | Validation declines while train improves | Scene generalization is a priority; restore genuine best weights |
| v3 continuation | −1.407 dB on diagnostic list | Check selection, distribution, prompt, decoder drift and exposure; not clean generalization evidence |
| Latent PMRF | Fidelity loss, inconsistent FID | Early stopped, degraded starting mean, missing clean control |

The manuscript notes two near-miss gates: v3 and FLUX printed FAIL early but later became relatively strong runs. Historical recommendation: do not kill before twice the loss-ramp length; use at least three flat/falling validations. This is a useful heuristic, not permission to ignore a hard compute cap. Gate criteria must consider the objective and the cost of confirmation.

## 8. External baselines and cross-domain verification

This section separates three different uses of “cross-domain”: external S2O papers on other datasets; mechanisms borrowed from restoration/colourization/segmentation/weather; and actual train-on-one-domain/test-on-another experiments. The first two have evidence below. A completed project-level zero-shot cross-dataset validation was **not recovered**.

### 8.1 External SEN1-2/SEN12 reference numbers

`EXTERNAL-VERIFIED`: original Bai & Xu papers report the following. These are literature rows under their protocols, not measurements by this project.

| Reference model | PSNR | SSIM | FID | Sampling |
|---|---:|---:|---:|---|
| CycleGAN, reported in comparator table | 13.33 | .2156 | 192.13 | — |
| NiceGAN, same table | 12.48 | .2181 | 211.62 | — |
| CRAN, same table | 14.13 | .2289 | 214.46 | — |
| S2ODPM | 18.76 | .3020 | 136.38 | — |
| Colour-supervised DDPM | 19.72 | .3119 | 116.93 | 1,000 diffusion training steps T |
| Adversarial consistency distillation (ACD) | 19.93 | .3108 | 126.76 | 8 inference steps |
| ACD | 20.57 | .3225 | 116.93 | 16 inference steps |

Colour-supervised DDPM adds a blurred-image colour loss; ACD adds adversarial consistency distillation. Both address known colour problems, so adding colour supervision alone is not a new contribution. Reproduce splits/rendering before claiming superiority. Their use of the name “SEN12” must be mapped to the exact release rather than confused with SEN12MS.

Sources: [colour-supervised DDPM](https://arxiv.org/html/2407.16921v1), [ACD](https://arxiv.org/html/2407.06095v1), [official ACD implementation](https://github.com/Coordi777/ACD_S2ODPM). The official repository advertises available pretrained weights; load compatibility and training exposure still require inspection.

### 8.2 Aydin, Hanna & Borth — SEN12MS

`EXTERNAL-VERIFIED`: [SAR-to-RGB Translation with Latent Diffusion for Earth Observation](https://arxiv.org/html/2504.11154v1), Tables 1 and 3:

| Translation setup | SSIM | FID |
|---|---:|---:|
| Standard diffusion without class | .818 | 85.409 |
| Standard diffusion with class | .824 | 84.681 |
| Cold diffusion | .876 | 150.923 |

Translation metrics use 200 images after a random training selection from SEN12MS. Separate SEN12MS-CR cloud-removal PSNRs are **31.85 / 32.84 dB**; they are not SEN1-2 SAR-Optic translation scores. Audit that task's actual inference path rather than assuming all cloud-removal methods use identical optical inputs. Class-label availability and the high-SSIM/poor-FID cold-diffusion result matter for interpreting transfer. None of these values establishes that this project's scene-disjoint implementation should reach .824 SSIM.

### 8.3 OSCAR — semantic optical-teacher guidance

`EXTERNAL-VERIFIED`: [OSCAR v1](https://arxiv.org/html/2601.06835v1) combines optical-teacher/SAR-student alignment with DINOv3-SAT, semantic prompts and uncertainty weighting.

| Dataset | SSIM | FID | LPIPS |
|---|---:|---:|---:|
| BigEarthNet-v2 | .3564 | 56.32 | .3831 |
| SEN12MS | .3625 | 34.73 | .4362 |

Protocol: random 80/20 splits; final samples 1,000 / 3,000 respectively; VV/VH-derived inputs; Refined Lee filtering; 3-sigma clipping and scaling; SD2.1; DDIM-50. Therefore these are not scene-disjoint or matched-input SAR-Optic baselines. Optical supervision in training is distinct from using the target at inference. Its ablation table also has individual partial configurations outperforming the full version on some metrics; isolate mechanisms rather than assuming every component helps. Our despeckling failure and OSCAR's preprocessing are different experiments, not a logical contradiction.

### 8.4 Cross-domain mechanisms already researched

| Source domain / method | Evidence recovered | SAR-Optic hypothesis | Transfer limitation and required control |
|---|---|---|---|
| Image restoration: PMRF | Original [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9de62e421d58234dbf773abf43268630-Abstract-Conference.html) predicts a mean then transports it; project has pixel/latent trials | Move along the perception–distortion frontier more efficiently than posterior sampling | Conditional mean is approximate; source data ambiguity differs. Need flow-from-noise, ordinary residual flow and OOF mean controls. |
| NIR→RGB: ColorMamba | [Original paper](https://arxiv.org/html/2408.08087v1): boundary tokens, local convolution, attention and HSV guidance alongside Mamba; reports +1.02 dB over its source-task comparator | Long-range context plus a local colour branch may improve spatially consistent colour | NIR retains optical relationships absent from SAR. +1.02 is not an isolated Mamba gain or predicted SAR gain. Ablate colour branch, local convolutions and scan separately. |
| Image restoration: MambaIRv2 | [CVPR source](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html) describes attentive SSM and semantic-guided neighboring | Connect distant semantically similar locations without relying only on local convolutions | It is not the notebook's small axial reference. Source-task gains do not prove SAR colour recovery; compare measured compute and local controls. |
| Deraining: PReNet | [CVPR source](https://openaccess.thecvf.com/content_CVPR_2019/html/Ren_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline_CVPR_2019_paper.html) uses progressive recurrent feature refinement | Repeated decoder refinement can aggregate useful spatial evidence | Rain removal retains much of the desired optical content. Repetition adds no new SAR observation; compare same-capacity convolution and iteration counts. |
| Medical segmentation: xLSTM-UNet / Vision-LSTM | [Original technical report](https://arxiv.org/abs/2407.01530) supports a spatial long-range backbone in segmentation | Recurrent spatial representations might help boundaries/global coherence | Segmentation is not RGB reconstruction; Vision-LSTM is not custom convolutional sLSTM. No project SAR improvement recovered. |
| Weather forecasting: ensemble verification | P1 cites CRPS/fair-CRPS, rank histograms, spread–skill and AIFS-CRPS | Evaluate distributions with proper scores instead of assuming diverse images are calibrated | P1 does report calibration results, despite a less complete chat-memory summary suggesting otherwise. Spatial dependence and small image counts limit generalization. |
| Distribution-free imaging uncertainty: conformal | P1 uses split-conformal calibration | Quantify how much interval scaling is needed for coverage | Exchangeability and domain shift matter; pixel coverage is not scene correctness. |
| Semantic image synthesis: spatial modulation | N2 cites [SPADE](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html) | Preserve spatial SAR cues when global context/FiLM is insufficient | A SAR-conditioned feature map is not a ground-truth segmentation map; compare spatial branch against global conditioning and plain convolution. |
| Super-resolution: PiSA-SR | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Pixel-level_and_Semantic-level_Adjustable_Super-resolution_A_Dual-LoRA_Approach_CVPR_2025_paper.html) uses pixel and semantic LoRA modules for adjustment | Separate fidelity and perceptual adaptation with an inference control | Proposed only; source task has more target information. Not the same as the project's failed single-objective PEFT runs. |
| Medical multimodal registration: MIND | P1 uses modality-independent descriptors | Encourage cross-modal local structure agreement | Modality invariance does not make radar edges identical to RGB boundaries; test weight/removal and actual geometric errors. |

Further names preserved from prior history: **CM-Diffusion** (colour memory), **DualCDM**, **Seg-LSTM**, **E3Diff**, **C-DiffSET**, and **TerraMind**. CM-Diffusion's title and [IEEE paper](https://ieeexplore.ieee.org/document/10623721/) were located. N1 proposes official E3Diff/C-DiffSET comparisons and TerraMind initialization. The exact intended DualCDM/Seg-LSTM references and their numerical evidence were not recovered reliably; do not attach guessed papers or rankings. The recalled statement “Seg-LSTM is weaker” is insufficient without a named comparison/protocol. These remain literature-verification tasks, not established baselines.

## 9. Hypothesis register and falsifiable tests

The hypothesis IDs below are local to this memory. They do not refer to any other project.

| ID | Hypothesis and present status | Smallest informative experiment / falsification |
|---|---|---|
| S2O-H01 | Coarse radiometry dominates the current residual. Supported for P1 v3 oracle subset, not universal. | Repeat frequency/global/grid oracles on random unseen scenes and seasons. Weaken if the error budget shifts substantially. |
| S2O-H02 | A better SAR-conditioned spatial correction can recover some oracle gain. Open; one global affine head failed. | Frozen predictions + small conventional residual/grid head on SAR features and season; OOF training inputs. Require held-out improvement over identity and mean correction. |
| S2O-H03 | Local spatial conditioning is more useful than global prompts for colour placement. Open. | Base vs spatial vs conv; same initializer/loss/data; compare low-frequency colour and boundary errors. |
| S2O-H04 | ConvLSTM refinement helps beyond added capacity. Code-only in N2. | Base/conv/ConvLSTM, equal budget, three seeds. No repeatable advantage over conv → no recurrence-specific support. |
| S2O-H05 | Custom stabilized sLSTM helps more than ordinary ConvLSTM. Synthetic mechanics checked; accuracy open. | Match decoder locations/iterations; compare ConvLSTM and custom sLSTM, numerical stability, cost and held-out quality. |
| S2O-H06 | Selective state-space context helps more than local processing. Synthetic reference checked; accuracy open. | Mamba reference vs conv/spatial, with compute and scan/pooling controls. Gains disappearing against controls reject the specific mechanism claim. |
| S2O-H07 | Learned semantic SAR features improve colour/generalization. Open. | Training-only optical teacher, SAR-only student inference; compare no-teacher and metadata-only controls. Explicitly test held-out footprint contamination. |
| S2O-H08 | Predicting a low-resolution colour field then detail is better than generic residual refinement. Proposed/history-aligned. | Compare smooth correction, generic residual head and identity under same capacity; audit seams, gamut and boundaries. |
| S2O-H09 | More diverse independent scenes help more than longer training on existing scenes. Strong motivation, unmeasured causal gain. | Fixed architecture/budget, scene-diversity learning curve; separate number of scenes from repeated patches/steps. |
| S2O-H10 | Multi-temporal SAR/acquisition metadata supplies useful missing context. Proposed. | Current SAR vs temporal stack with timestamp controls and geographic/time-disjoint evaluation. Keep optical-assisted variants separate. |
| S2O-H11 | Strict geographic grouping lowers optimistic performance further than source-scene IDs. Open audit. | Group overlapping footprints across seasons/dates, deduplicate, rerun small baseline. Report actual overlap rather than assume. |
| S2O-H12 | OOF posterior means improve refinement generalization. Implemented option, completed result missing. | Same B network on in-sample vs OOF means; same inference A and matched B budget. Include cache-generation cost. |
| S2O-H13 | PMRF yields lower distortion at matched perceptual quality than posterior sampling. Not established by current incomplete comparisons. | Genuine best A; B_ref vs B_noise vs B_pmrf, K/lambda sweep chosen on validation, held-out FID/LPIPS and fidelity. |
| S2O-H14 | Constraining stochastic refinement to protected residual/frequency bands preserves structure. Prior proposal, untested. | Constrained residual flow vs ordinary residual flow and PMRF; assess low-frequency drift, edge displacement and perception. |
| S2O-H15 | Proper-score optimization improves calibration at acceptable fidelity cost. Calibration improvement observed; current cost unacceptable for fidelity goal. | Small coefficient sweep with mean/sample metrics and discrimination; held-out scene calibration. A better SSR alone is insufficient. |
| S2O-H16 | Pixel U-Net is a better budget allocation than the large latent system. Supported on one smaller corpus. | Matched-input clean-split baseline on official data; quality/cost/latency, not parameters alone. |
| S2O-H17 | Improved codec reconstruction translates to improved S2O. Contradicted by one FLUX swap. | Only revisit with codec scaling/statistics and matched initialization controlled; low priority now. |
| S2O-H18 | Registration or simple sharpening is the main bottleneck. Weak support from current oracle family. | Extend to local alignment/boundary diagnostics only if errors justify it; do not repeat expensive global cleanup without evidence. |
| S2O-H19 | Generated RGB improves downstream utility despite imperfect paired fidelity. Motivated externally, project result missing. | SAR-only / generated-RGB-only / SAR+generated-RGB task models, identical training data/budgets; real RGB is a labelled upper-reference input. |
| S2O-H20 | Independent fidelity/semantic adapters give a useful adjustable frontier. Proposed PiSA-inspired direction. | Compare dual adaptation with single adapter, scalar output blends and frozen baseline; measure the full Pareto curve. |

For colour-correction heads, report held-out gain relative to the matching oracle with careful denominator choice. A ratio of dB gains is descriptive, not a physical fraction of MSE explained. Prefer also reporting MSE reduction and validation/test regression error for correction parameters.

## 10. Baseline registry required for a defensible next study

| Tier | Baseline / control | Reason and status |
|---|---|---|
| Protocol | Constant train-mean RGB | Reveals whether scene-held-out predictor adds useful information |
| Protocol | Overlap optical lookup under patch and scene split | Quantifies leakage; diagnostic only, never a deployable model |
| Fidelity | Clean-split pixel MSE Stage A | Cheap primary reference and approximate posterior mean |
| Fidelity | SAR-only L1 U-Net | Controls loss and extra context/season information |
| Historical | Original v3 | Useful diagnostic, exposure-contaminated for some later scene lists |
| Historical | Clean-split v3 recipe | Architecture comparison without contaminated initialization |
| Correction | Identity / global bias / train-mean correction / small residual CNN | Determines whether complex colour module is necessary |
| Capacity | Parameter-similar decoder convolution | Controls ConvLSTM/ConvXLSTM capacity confound |
| Recurrent | Conventional ConvLSTM | Baseline for custom sLSTM claims |
| Spatial | SAR spatial modulation, with/without recurrence | Tests local conditioning and interaction |
| SSM | Build-free Mamba-style reference | Tests a limited selective-scan hypothesis, not official MambaIR/VMamba |
| Generative family | Pix2Pix, CycleGAN, DDPM, BBDM-style, ResShift-style | N1 compact reimplementations, not official SOTA reproductions |
| Flow | PMRF full / pure-PMRF / flow-from-noise / OOF means | Isolates mean initialization, extra conditioning and cache mismatch |
| Official competitor | Authors' ACD/colour-DDPM, plus verified relevant S2O methods | Required to strengthen SOTA claims; same manifests and input assumptions |
| Diagnostic upper reference | Optical round-trip codec, target-assisted correction | Bounds within specified construction, excluded from main deployable ranking |

N1's compact family baselines use a shared harness and mostly SAR patch-only inputs. Proposed A uses context/season; therefore report both a matched-input comparison and the best-information configuration. Equal steps alone are not equal compute, equal convergence, or faithful official reproduction. CycleGAN is unpaired by design and uses two generators; state this when comparing training cost.

N1 Stage-A ablations: `A_ref`, `A_noctx`, `A_nometa`, `A_plain`, `A_l1ssim`. Stage-B controls: `B_ref`, `B_pmrf`, `B_noise`, `B_oof`, optional sigma {0.05,0.20}. `A_patch` measures patch-protocol inflation. Presence in source does not establish completed runs; most corresponding cells have no saved outputs in the recovered N1.

## 11. Weaknesses and room for improvement

### 11.1 Critical correctness and reproducibility weaknesses

1. **Checkpoint exposure:** old v3 saw many later “held-out” patches. A fresh split cannot clean the training history of a pretrained checkpoint.
2. **Checkpoint identity:** missing best weights fell back to last, yet cache labels claimed best. Cache producer hash, actual step and split must be authoritative.
3. **Historical-best selection:** appended v3+ finalization picks current RAW/EMA rather than historical best. Save best model **and its corresponding decoder** together throughout training.
4. **Flow starting point:** latent PMRF used failed v3+ when that directory existed. Compare explicitly against original-v3 initialization or genuine best clean Stage A.
5. **Metric conflicts:** bridge FID 128.2/168.8; known ERGAS units; PSNR pooling differences; missing metadata for some metrics.
6. **Credential handling:** N1 contains credential-like HF literals in persistence/appended code. Do not copy, execute or publish them. Replace with secrets/environment input and have the owner rotate exposed credentials. Rotation was not performed here.
7. **Dataset lineage:** 307,880 augmented pairs versus 282,384 official pairs, original RGB rendering and augmentation ancestry are insufficiently documented.
8. **Cached evaluation identity:** matching filenames are insufficient. A cache key must include weights, decoder, source corpus, split, preprocessing, metric implementation and inference settings.

### 11.2 Scientific inference weaknesses

- “Colour is not a function of radar,” “13 dB ceiling,” “backbone capacity is not binding,” “detail placement is optimal,” and “no S2O work has done X” are stronger than the recovered evidence. Rephrase as observed model/data limitations unless stronger tests or a dated literature review support them.
- The 3 dB leakage estimate comes from a particular smaller-corpus comparison; it is not a correction factor or guaranteed lower bound for the headline experiment.
- Large reconstruction headroom above translation PSNR does not by itself establish that codec error is irrelevant; codec/translation error can interact.
- Small oracle/calibration samples, few scenes, one-seed comparisons and patchwise intervals limit statistical strength.
- Small metric differences need paired scene-level analysis and seed variability. An interval crossing zero means insufficient evidence for a difference, not equivalence.
- A low velocity loss can be uninformative near the source endpoint; monitor rollout quality and timestep-stratified errors.
- Diversity is not calibration, and calibration is not correct semantic uncertainty. Pixel-level CRPS does not fully score spatial joint distributions.
- RGB SAM/ERGAS do not establish recovery of unobserved multispectral reflectance. Avoid scientific-use claims beyond the actual output bands/task.
- Parameter-count and simulated quantization results do not establish end-to-end deployment speed, memory or energy improvements.

### 11.3 Model and data opportunities

- Test colour learnability with a small residual/spatial predictor before another large generator.
- Improve independent scene diversity and validate preprocessing instead of merely extending an overfitting run.
- Evaluate training-only optical-teacher semantics with a SAR-only student at inference.
- Recover dual polarization and acquisition metadata from appropriate source data where available; do not invent channels absent from the PNG release.
- Treat temporal SAR, historical optical, DEM/geolocation and external land-cover layers as explicit new input regimes. Historical optical may help but changes a SAR-only claim.
- For perception, benchmark a controlled frontier using OOF refinement and fixed spatial diagnostics. Do not optimize FID while ignoring hallucinated content.
- For practical value, evaluate a downstream task and compare with direct SAR features. Generated RGB cannot be assumed to add information beyond its input.

## 12. Recommended execution plan

This is an ordered plan, not a record of completed work. Tasks can be assigned to future agents separately, but this consolidation did not launch or complete research runs.

### P0 — Establish an authoritative, trustworthy state

**Task A: recover artifacts and sanitize execution.** Read P1/N1/N2, enumerate actual checkpoint files and source revisions, remove embedded credentials from working copies, keep originals unchanged unless an edit is requested. Determine whether true historical best A weights can be recovered from persistence. Do not run all notebook cells blindly: `%%writefile` cells overwrite generated modules.

**Deliverables:** artifact inventory; checkpoint lineage; redacted runnable configuration; missing-file list. **Done when:** every model intended for comparison has exact weight/decoder hashes, split exposure, training step and input specification, or is excluded explicitly.

**Task B: freeze data and splits.** Recover `items.json` and original split files, source-scene keys, augmentation ancestry and RGB conversion. Audit duplicates and overlapping footprints across dates/seasons. Produce one canonical scene/region-disjoint train/validation/test manifest. Preserve old manifests for historical reproduction.

**Deliverables:** `dataset_manifest.json`, `split_manifest.json`, duplicate/footprint audit, preprocessing specification. These filenames are recommendations; existing files may use different names. **Done when:** counts, hashes, group boundaries and model exposure are independently checkable.

**Task C: reconcile evaluator and cache lineage.** Re-evaluate existing bridge/v3 on one frozen subset first. Explain FID 128.2 versus 168.8 using checkpoint, seed, sample count, range, preprocessing and feature implementation. Fix ERGAS ranges and best/last identity. Reject stale caches when their producer metadata differs.

**Deliverables:** metric protocol, reconciliation table, authoritative prediction-cache manifest. **Done when:** a result can be regenerated from specified inputs/configuration without manual ambiguity. If FID cannot be recovered, mark the old figure unresolved rather than fabricate reconciliation.

### P1 — Reproduce the simple references and error budget

**Task D: constant and clean pixel baseline.** Score train-mean colour and a compatible clean Stage-A checkpoint. If no clean checkpoint exists, use a budgeted shared initializer only after explicitly changing the notebook configuration to permit it. Compare patch-only and context/season-assisted input regimes.

**Task E: repeat diagnostic oracles.** Use a random validation scene sample for global shift, global affine, smooth/grid affine, frequency replacement, and low-frequency/edge errors. Define oracle fitting/regularization before viewing test outcomes. Plot distributions by scene/season, not only the mean.

**Done when:** baseline metrics and image panels are reproducible; colour headroom and structural problems are measured on the intended generalization regime. Preserve the final test set while designing methods.

### P2 — Highest-value model experiment: learnable spatial colour correction

1. Freeze a provenance-clean first-stage predictor and form training predictions OOF or using an independent calibration/training partition. If old v3 is used for an exploratory screen, label it as exposed and do not report clean generalization.
2. Start with identity, train-mean/global correction, and a small residual CNN using prediction + SAR/context features + legitimate metadata.
3. Predict bounded smooth colour residuals or a low-resolution affine grid; compare against an unconstrained residual of similar capacity.
4. Train using targets only in training. Target-derived affine grids are supervision labels, never inference features. Avoid fitting test-target corrections inside the deployed path.
5. Track mean image PSNR, MSE, SSIM, low-frequency colour error, SAM, appropriate colour-space error, boundary consistency and fixed visual panels.
6. If the simple head works, add SAR spatial modulation or training-only semantic distillation. If it fails, examine head-target predictability, teacher forcing/cache mismatch and data diversity before adding recurrence.

**Acceptance:** positive paired scene-level gain, consistent seed/scene behavior, no material boundary/perception regression, and cost worth the gain. Pre-register practical thresholds on validation. The old +0.15 dB gate is historical, not automatically a suitable universal threshold.

### P3 — Run the budgeted architecture screen only after P0/P1

Use N2: `base`, `conv`, `convlstm`, `convxlstm`, `spatial`, `mamba_ref`; optional `spatial_lstm` and `crossattn` after evidence. Preserve the same initialized backbone, split, batch, losses, schedule, validation and augmentation. Confirm the candidate against **both base and conv**, not only base.

**Acceptance:** at least three training seeds; paired scene CIs; real CUDA throughput/VRAM measurements; fixed unbiased images; honest custom-module naming. If no variant beats the conventional control consistently, retain the simpler baseline and report the null result.

### P4 — Strong priors or added information

First verify that any proposed foundation-model weights can actually load affordably and have appropriate inputs. Candidate directions from history: TerraMind-initialized Stage A; optical-teacher semantic features distilled into a SAR student; multi-temporal SAR; legitimate acquisition/geospatial context. Freeze input assumptions and compare each added information source separately.

**Acceptance:** clean region/season generalization, no target optical at inference, provenance of pretrained data and labels documented, and improvement beyond a plain backbone/input control. Do not launch a heavyweight training program solely from a paper title.

### P5 — Perceptual refinement, then cross-dataset transfer

- Restore actual best A and retrain/reuse B with verified caches. Compare in-sample/OOF, flow-from-noise, ordinary residual flow, pure PMRF and full conditioned flow.
- Choose K, noise and blends on validation; score the frozen frontier on test. Include real FID/LPIPS, scene fidelity and object/boundary diagnostics.
- Then zero-shot test the frozen trained model on a separately prepared dataset such as SEN12MS. No target-domain fitting in the zero-shot row. If adaptation is allowed, add a separate adapted row and fresh target test split.
- Separate sensor/polarization, geography, season and rendering shifts where possible. A preprocessing mismatch can dominate a cross-domain result.
- Evaluate downstream SAR-only / generated RGB / combined input models if suitable independent labels exist.

**Acceptance:** cross-dataset result manifests specify input compatibility, no geographic overlap, normalization, no target-label selection, sample count and uncertainty. Until this runs, “cross-domain verification” means literature/mechanism comparison, not validated project transfer.

### P6 — Paper and release

Rewrite claims around demonstrated results. Keep historical patch-split measurements visible with caveats; add clean benchmarks rather than silently replacing tables. Resolve bridge FID, qualify oracle/noise-floor claims, repair citations, report missing metrics as unavailable, and separate literature tables from reruns. Release code/configs/manifests and negative results only after scanning for credentials and checking artifact completeness.

## 13. Compute plan and N2 implementation details

### 13.1 Historical 600-CU allocation

| Phase | Allocation | Planned work |
|---|---:|---|
| Screen | 100 CU | Six default variants, one seed, 10k fine-tuning steps; shared 20k initializer if explicitly enabled/needed |
| Confirmation | 320 CU | Candidate + base/conv, seeds 42/123/2026, 30k fine-tuning steps |
| Final evaluation | 60 CU | Locked comparison and perceptual metrics |
| Reserve | 120 CU | Recovery, data/metric overhead, unexpected cost |
| Total | 600 CU | A planning cap, not a billing guarantee |

N2 requires live `CU_PER_HOUR` and actual `REMAINING_CU`. It times full forward/backward/optimizer steps, includes 25% estimated overhead, and stops plans exceeding phase limits. Download, idle allocation and final FID/LPIPS can add cost; monitor actual balance. Do not treat a plan-fitting estimate as permission to consume beyond the user's budget.

### 13.2 N2 defaults and important correction to recalled summaries

- Work directory `/content/pmflow_work`; smoke directory `/tmp/s2o_colour_smoke`.
- Width 32; recurrent refinement steps 3; colour-weight default 0.0.
- LR 5e-5 for fine-tuning; warmup 1k; validate every 1k; checkpoint every 15 minutes; session cap 8h.
- **Actual non-smoke training config sets `cfg.bsA=16`. `BS=8` is evaluation batch size.** A recalled summary said batch 8 generally; source code distinguishes them. If OOM, lower training batch consistently for all compared variants and repeat the budget measurement.
- Evaluation default `EVAL_N=4096`, `EVAL_TEST=False`; final test only in confirmation after decisions are fixed. Prefer full held-out set for final distributional estimates if affordable.
- Missing compatible initializer stops by default. `ALLOW_SHARED_PRETRAIN=False` in real mode unless explicitly enabled in configuration. Shared initializer LR is 2e-4.
- Example initializer: `/content/pmflow_work/ckpt/A_main/best.pt` with **original** split/items manifests. Architectures and exact manifests must match; independent training-exposure provenance is still required.
- New branch interiors are randomly initialized. Zero output projections initially preserve the pretrained backbone function; they do not make the added modules pretrained.
- The SD v3 checkpoint is incompatible with the pixel architecture and must not be silently imported.
- Dependencies use binary wheels; no requested `mamba-ssm`, `causal-conv1d`, Triton or xLSTM extension build.

### 13.3 What the custom modules mean

- `convlstm`: repeated refinement of static decoder features, **not a temporal SAR sequence**.
- `convxlstm`: custom convolutional stabilized **sLSTM**, exponential input gate, sigmoid forget gate, scalar memory, normalizer and log stabilization. Not mLSTM, official Vision-LSTM or an official uniquely identified “ConvXLSTM” reproduction.
- `mamba_ref`: Mamba-1-style input-selective recurrence with learned negative A, delta/B/C, causal depthwise convolution, D skip and output gating; four shared-weight axial scans on a pooled map ≤16×16; residual decoder branch. Not complete VMamba or MambaIRv2. A Python/PyTorch reference can be slower than optimized implementations.
- `conv` is approximately parameter matched for recurrent controls, not perfectly matched for every spatial/SSM branch. Measure parameters, training step time, inference latency and VRAM.

### 13.4 Verification already documented for N2

N2 states CPU verification with PyTorch 2.14.1+cpu on synthetic data:

- All Python cells parse and notebook format validates.
- Stabilized convolutional sLSTM matches an independent direct recurrence and stays finite under extreme log gates.
- Selective SSM matches an independently indexed recurrence and passes double-precision autograd gradcheck.
- All eight variants pass shape, finite-gradient, zero-initialization equivalence and state-reset checks.
- Six default variants complete short training/EMA validation/checkpoint reload/evaluation/figure/export flow.
- Synthetic confirmation runs base/conv/candidate across three seeds and traverses the final-test path.
- Resume restores optimizer/model/EMA; incompatible schedule rejected.
- Imported Stage-A weights preserve the starting function in new variants.
- Full-size 256×256 forwards reported: ConvXLSTM **53,903,468 parameters**; Mamba reference **53,628,716**.

These are **documented prior checks**, not checks rerun during creation of this memory. Unverified there: A100 kernels/VRAM/speed, external server/credential access, perceptual-weight downloads, and real SEN1-2 improvements.

### 13.5 Saved N1 runtime measurements — historical only

N1 records A100-SXM4-40GB, torch 2.11.0+cu130. Representative batch-32 benchmark: A **0.298 s/step, 18.7 GB**, B **0.398 s/step, 24.9 GB**; approximately 53.5 M / 92.1 M parameters. Appended v3+ run: approximately **1.54 iterations/s effective batch 8, 26.6 GB**; latent flow approximately **3 iterations/s, 28.3 GB**. These are different models/runs from N2 and do not certify current throughput.

N1's broader initial plan includes A 60k, B 80k, A ablations 25k, B controls 30k, OOF A 25k, L1 regressor 40k, Pix2Pix/CycleGAN/diffusion 60k, patch diagnostic 25k. Some printed family-budget rows used 1,000-step settings. Read the active RUN/STEPS and measured plan rather than assuming every family ran for the nominal default.

## 14. Implementation map and artifact contracts

N1 generates these modules using notebook `%%writefile` cells:

| Module | Responsibility |
|---|---|
| `common.py` | Config presets, seeds/utilities, persistence store |
| `data.py` | Streaming/local/Kaggle ingestion, scene selection, arrays and context |
| `models.py` | Conditional pixel U-Nets, context/season components |
| `train.py` | Resumable generic training, EMA, validation and checkpointing |
| `baselines.py` | Compact family comparison implementations |
| `audit.py` | Mean colour / overlap lookup / patch protocol diagnostics |
| `evalm.py` | Common pixel/perceptual/distribution metrics and scene resampling |
| `registry.py` | Model loading and evaluation metadata |
| `pipeline.py` | Run orchestration and persistence integration |
| `experiments.py` in N2 | New decoder modules, shared initialization, controlled experiments |
| Appended `v3x.py` | Historical latent v3 reproduction/continuation/flow/reporting |

Historical paths, not claims these files currently exist in every environment:

- `/content/pmflow_work/data/{sar,rgb}.u8`, `items.json`, original split file (often `split.json`).
- `/content/pmflow_work/ckpt/A_main/{best,last}.pt`.
- Shared initializer `EX2_shared_base_s42_c0/best.pt`.
- Experiment names include phase, kind, seed and colour weight; fingerprints prevent incompatible resumes.
- `/content/v3x/v3plus_reg/best`, `/content/v3x/pmrf_flow/best`, `/content/v3x/v3x_results.zip`.
- Remote paths referenced by N1: `adapted16_final/base_vae.pth`, `adapted16_final/detail_encoder.pth`, `regressor_v3_final/base_vae_tuned.pth`, `regressor_v3_final/bridge_config.json`, `regressor_v3_final/unet.safetensors`, `v3plus/report`.
- N2 exports `experiment_budget.csv`, `screen_selection.json`, `colour_<split>_summary.csv`, `colour_<split>_paired_deltas.csv`, `colour_experiment_manifest.json`, fixed comparison figures and `colour_experiment_results.zip`.

### 14.1 Minimum run record for every new experiment

```yaml
run_id: unique descriptive name
status: planned | smoke_passed | running | paused | complete | failed | early_stopped
claim_status: exploratory | confirmed | negative | unresolved
source_revision: exact commit or source hash
dataset_manifest_hash: required
split_manifest_hash: required
grouping: patch | source_scene | region_and_time
train_val_test_counts: required
preprocessing: SAR units/channels + RGB conversion + normalization
input_regime: SAR_patch / SAR_context / metadata / auxiliary_sources
initial_weights_hashes: model + codec + decoder + adapters
prior_training_exposure: known / contaminated / unknown
training: seed + steps + batch + accumulation + losses + schedule + EMA
selection: validation metric + selected step + actual selected checkpoint hash
cache: actual producer hash + step + OOF fold + dtype + item order
inference: NFE + stochastic seeds + CFG + blend + resolution
evaluation: sample IDs + metric versions/ranges + FID features/sample count
uncertainty: scene CIs + training-seed variability, explicitly distinguished
cost: hardware + time + peak VRAM + measured/actual CU
results: paths and hashes for rows, predictions, logs and fixed figures
limitations: missing data, gate stop, exposure or metric warnings
```

No fallback may retain the old identity label. If `best` is unavailable, either recover it or record an explicit `last` run and invalidate dependent cached outputs. Persist selected U-Net and fine-tuned decoder atomically. Capture optimizer/scaler/scheduler/EMA/RNG/sampler state for resume where applicable; multiworker augmentation may prevent bitwise replay.

## 15. Evaluation and reporting checklist for future agents

1. Freeze the data, splits and input regime before comparing methods. Audit pretraining exposure and geographic overlap.
2. Reproduce initial checkpoint predictions at step zero before adding branches/losses. For adapters, verify zero-output equivalence and finite gradients on the actual runtime.
3. Select architectures, prompts, loss weights, NFE and blends on validation. Do not use test metrics or images for iterative tuning.
4. Use the same held-out examples and stochastic draws for paired comparisons. Report sample scores and ensemble-mean scores separately.
5. Compute per-image PSNR before averaging. Report pooled MSE/RMSE separately if used. Document exact-match caps.
6. Report PSNR/SSIM/MAE/RMSE/CC/SAM and corrected ERGAS only with units and range. Add colour/frequency/boundary diagnostics when making colour/spatial claims.
7. LPIPS must name backbone/range. FID must name feature implementation, sample counts and reference set; unavailable dependencies yield `n/a`, never zero. Consider KID or an EO-relevant feature metric as a complement, with explicit implementation.
8. Cluster resampling by scene is preferable to independent-patch intervals. Report the number of independent scenes and variability across at least three training seeds separately. Do not label a seed SD a confidence interval.
9. Small FID differences need uncertainty/repeated draws; FID has no per-image contribution suitable for ordinary paired pixel bootstrap.
10. Fixed visual grids should span seasons/land covers and show failure cases, low-frequency colour errors, boundary displacement and invented structures. Avoid selecting examples to fit the desired conclusion.
11. For probabilistic models include proper scores, diversity, spread–skill, ranking and coverage/width. Do not claim domain-shift guarantees from in-distribution conformal calibration.
12. Distinguish compact reimplementations from official reproductions, diagnostic oracles from deployable models, and historical literature numbers from shared-harness reruns.
13. Report gates, interrupted runs and missing artifacts honestly. Successful code execution is not successful research; failed fixed-budget experiments are not universal disproof.
14. Store durable configurations, manifests, checkpoints, raw per-image results, environment and figures. Update this memory with evidence and exact protocol identity.

## 16. Open questions and retrieval gaps

| Question | Why it matters | How to resolve |
|---|---|---|
| What creates 307,880 pairs from the official 282,384? | Duplicate/augmentation leakage and rendering lineage | Original download/preprocessing scripts and manifests |
| Can headline scene IDs and geography be reconstructed? | Strength of 18.32/.306 generalization claim | Original filename/ROI/acquisition metadata audit |
| Which checkpoint/metric pipeline gives FID 128.2? | Central perceptual claim | Recovered predictions, evaluation JSON and source revisions |
| Does true Stage-A best at 12.938 survive? | PMRF cache starting point | Local/remote checkpoint versions and hashes |
| Was B_main ever completed beyond recovered 40k output? | Avoid claiming planned 80k as completed | Training log/checkpoint metadata |
| Are OOF and official baseline runs complete? | Core PMRF and SOTA controls | Real output files, not empty code cells |
| What is the scene-reference colour source? | Possible optical-assisted input | Inspect original ablation construction and inference availability |
| Are raw ERGAS outputs correct outside Table VI? | Consistent spectral-distortion reporting | Unit tests plus rescoring cached predictions |
| What is the original affine head target/regularization/split? | Strength of the learnability conclusion | Source and training logs |
| Which DualCDM/Seg-LSTM references were intended? | Accurate cross-domain memory | Recover exact paper IDs and earlier comparison |
| Have any N2 variants been trained on real data since delivery? | Avoid stale “unverified” status | New logs, manifests and results from owner/source store |
| Current compute balance/hardware and artifact permissions? | Feasible execution plan | Live runtime/account information at execution |
| Current manuscript/repository revision and submission status? | Avoid editing stale copy | Authoritative source and revision history |
| Which downstream task has independent suitable labels? | Utility claim beyond visual metrics | Dataset/label availability and deployment requirement |

Missing information is not a reason to block reversible source inspection or protocol fixes. Continue independent recovery and only ask the owner for facts unavailable from artifacts when they materially block the next action.

## 17. Reference map for continued work

These are the relevant source families, not an exhaustive current SOTA survey. External numerical tables above were checked in original papers during this consolidation; the rest of P1's bibliography remains a research reading list unless stated otherwise.

- **Core S2O:** SEN1-2 (Schmitt/Hughes/Zhu); conditional diffusion S2ODPM; colour-supervised DDPM; ACD; colour-memory CM-Diffusion; Aydin SAR-to-RGB; OSCAR.
- **Core generative theory:** DDPM; latent diffusion; ResShift; Blau/Michaeli perception–distortion; Liu/Gong/Liu rectified flow; Ohayon/Michaeli/Elad PMRF.
- **Verification:** Matheson/Winkler CRPS; Gneiting/Raftery proper scores; Ferro fair ensemble scoring; Hamill rank histograms; Fortin et al. spread–skill; Price et al. probabilistic weather; Lang et al. AIFS-CRPS; Angelopoulos et al. distribution-free image regression.
- **Metrics/structure:** Heusel et al. FID; Zhang et al. LPIPS; Heinrich et al. MIND.
- **Adaptation:** Hu et al. LoRA; Liu et al. DoRA; P1's MiCA reference (arXiv:2604.01694, verify before reproducing); PiSA-SR dual LoRA.
- **Recurrent/SSM:** [ConvLSTM](https://arxiv.org/abs/1506.04214), [xLSTM](https://arxiv.org/abs/2405.04517), [Vision-LSTM code](https://github.com/NX-AI/vision-lstm), [Mamba](https://arxiv.org/abs/2312.00752), [selective scan reference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py), ColorMamba, MambaIRv2, PReNet, xLSTM-UNet.

For bibliographic work, prefer the original publisher/arXiv record over extracted PDF text; for example, the official PiSA-SR CVPR record gives pages **2333–2343**. Verify the actual LaTeX bibliography before submission.

## 18. Maintenance and next-agent handoff

**Best immediate assignment:** recover the genuine Stage-A best checkpoint and original manifests, fix cache/checkpoint identity and evaluator conflicts, then run a small clean spatial-colour residual baseline. This is more informative than starting another large flow model immediately.

Before changing this memory, preserve prior numbers with their source/protocol. Add a dated entry rather than overwriting conflicting evidence. Move a hypothesis to “supported” only when its specified control and generalization test succeed; move it to “negative” only for the tested configuration. Keep `unknown`, `not recorded`, `early stopped`, and `not independently reproduced` distinct.

### Consolidation log

- **28 October 2026:** Read accessible shared SAR-Optic history, P1 and both notebooks; consolidated numerical tables, failures, theory and future plans. Checked selected original external references. Recorded the 600-CU constraints and the actual N2 training/evaluation batch distinction.
- Added source-level findings not safely represented by a short project summary: N1 overlap-lookup leakage; missing-best fallback with misleading cache label; appended v3+ final-step versus historical-best selection; latent PMRF automatically using failed v3+; existing v3 exposure to later test-list patches.
- No new model training, checkpoint rescoring, credential rotation, manuscript editing, remote repository modification, or completed cross-dataset experiment occurred during this consolidation.

**Standing principle:** explain and improve the remaining colour/spatial error with matched controls, clean provenance and finite compute. Preserve the negative evidence, but do not turn an unsuccessful implementation into a claim that improvement is impossible.

