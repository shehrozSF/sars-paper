# Running on Kaggle (competition GPUs, Internet off)

Kaggle copies of the three Colab notebooks. The originals in the parent folder are unchanged and keep their saved outputs. These copies have **outputs cleared**.

## One-time: build the asset datasets (Internet ON, CPU)
Open `00_build_assets_kaggle.ipynb` in a normal Kaggle notebook with Internet enabled. Then, for each bundle you need:

1. Set `BUNDLE`.
2. **Save Version → Save & Run All**.
3. On the finished version's Output tab, choose **New Dataset** and name it `sar-assets-<bundle>`.

| Bundle | Size | Needed for |
|---|---|---|
| `base` | ~12 GB | everything |
| `regressors` | ~10 GB | v1/v2/afCRPS cells |
| `flux` | ~4 GB | FLUX cell 74; accept the FLUX licence on HF first |
| `fyp_runs` | ~13 GB | `complete_pipeline` |

Each run has to stay under Kaggle's ~20 GB output limit, which is why the assets are split.

## Each run
1. Choose the **RTX Pro 6000** accelerator. The scripts use a single GPU; L4×4 would give you one 24 GB L4.
2. Attach the inputs:

   | Notebook | Inputs |
   |---|---|
   | `SAR-to-Optic_kaggle` | `sar-assets-base` (+ `regressors` / `flux` as needed), Kaggle dataset `shambac/augmented-sentinel-1-2` |
   | `complete_pipeline_kaggle` | `sar-assets-base`, `sar-assets-fyp_runs`, Kaggle dataset `requiemonk/sentinel12-image-pairs-segregated-by-terrain` |
   | `S2O_Complete_Benchmark_kaggle` | `sar-assets-base`, a SEN1-2 copy (defaults to the augmented dataset; set `SEN12_DIRS` to use another) |

   Also attach a private dataset holding your `.env` (copy `../.env.sample` and fill in the HF repo ids). The first cell of every notebook loads `/kaggle/input/*/.env`. Keep the token in the Kaggle secret `HF_TOKEN`, not in the dataset.
3. Run the **KAGGLE SETUP** cell first, with `OFFLINE = True` when Internet is off. It does five things:
   - points the notebook at the attached assets;
   - installs any missing wheels;
   - sets the offline environment variables;
   - installs a small `sitecustomize.py` shim, so `hf_hub_download`, `snapshot_download` and `list_repo_files` read from the assets in every `!python` subprocess;
   - locates the attached datasets.
4. For long jobs, use **Save Version → Save & Run All**. Sessions stop at 12 h.

## What was changed
- **Paths:**
  - `/content/` → `/kaggle/working/`;
  - large work dirs → `/tmp` (`pmflow_work`, `sar_fyp_cache`, `v3x`);
  - Drive → `/kaggle/working`.
- **Model and dataset IDs:**
  - SD-1.5, FLUX and SAM IDs → local asset paths (env `SD15_PATH`, `FLUX_PATH`, `SAM_PATH`);
  - hardcoded Kaggle dataset path → `SAR_DATA_ROOT` / `SAR16K_ROOT`.
- **Colab-only code:**
  - Colab secrets → Kaggle secret `HF_TOKEN`, used only when online;
  - `!pip` lines and `runtime.unassign()` commented out.
- **`complete_pipeline`:**
  - pushes and `require_hf_token` are skipped offline;
  - `evaluate.py` now pulls configs before filtering experiments (it previously evaluated nothing in a fresh session);
  - the last cell copies eval results to `/kaggle/working`.
- **Benchmark notebook:**
  - offline it uses `SOURCE_MODE='local_archives'` instead of rsync;
  - checkpoints are mirrored to `/kaggle/working/pmflow_benchmark`. Attach a previous run's output to resume, and the setup cell restores it.
- **`SAR-to-Optic`:** the `list_repo_files` call in `stage4_final_eval.py` is made offline-safe.

## Not changed (see `../HANDOFF_improvements.md`)
These copies only port the notebooks to Kaggle; the audit fixes still have to be made:
- the train/eval split mismatch (P0-1);
- the v3 tuned-decoder evaluation (P0-2);
- relabelling `sceneref` and Sec. X.

Make those fixes in the Kaggle copies before re-running the evaluations.

## Known limitations
- `SAR-to-Optic` cells 1–17 (the old DoRA/W&B run) need Internet; skip them.
- Offline, nothing is pushed to the Hugging Face Hub. Results stay in `/kaggle/working` (the run's output). Upload them later from an online session if you want them on the Hub.
- Training scripts in `SAR-to-Optic` save full accelerator states into `/kaggle/working`, which can hit the 20 GB output limit. Evaluation scripts are small.
- The augmented Kaggle dataset used as the SEN1-2 source for the benchmark notebook has extra `img_*` files. The ingestor ignores files that don't match the SEN1-2 naming, but scene coverage differs from the official rsync subset.
