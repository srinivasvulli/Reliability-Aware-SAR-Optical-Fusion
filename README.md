# Reliability-Aware Cross-Modal Fusion of Sentinel-1 SAR and Sentinel-2 Optical Imagery for Robust Land-Cover Classification

Code for the *Remote Sensing* manuscript of the same title.

The project tests whether a fusion model that learns **how much to trust each modality** stays
more reliable than single-modality and simple-fusion baselines when Sentinel-2 optical imagery
is degraded by noise, missing bands, or cloud-like masking. The proposed model fuses
per-modality prediction logits using a **noise-aware reliability gate** driven by feature
representations, prediction-confidence descriptors, and optical-quality indicators.

The full workflow is in the notebook **`reliability_aware_fusion.ipynb`** — running it
downloads the dataset, pairs the SAR-optical images, trains the proposed model with
balanced-robustness checkpoint selection, evaluates it across clean and degraded settings, and
writes the result tables, confusion matrices, and figures. No results or figures are stored in
this repository; they are regenerated on each run.

## Dataset

The notebook downloads the Kaggle dataset
[`requiemonk/sentinel12-image-pairs-segregated-by-terrain`](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)
via `kagglehub` — a four-class SAR-optical subset derived from SEN12MS (16,000 paired
Sentinel-1 / Sentinel-2 patches; agriculture, barrenland, grassland, urban).

If `kagglehub` prompts for credentials, place your Kaggle API token at `~/.kaggle/kaggle.json`
(`chmod 600`). The token is never committed — `kaggle.json` is git-ignored.

## Run

**Google Colab (recommended, GPU):** open `reliability_aware_fusion.ipynb` in Colab, set the
runtime to a GPU, and run all cells. The first cell installs the dependencies.

**Local Jupyter:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook reliability_aware_fusion.ipynb
```

Running all cells writes `balanced_model_seed2026.pt`,
`reliability_aware_fusion_full_metrics.csv`, `table5_perclass_f1.csv`, the
`confusion_matrix_*.csv` files, and the figure PNGs to the working directory.

> **Note on reproducibility.** GPU training with cuDNN is not fully deterministic by default,
> so exact per-run numbers differ slightly across machines and runs (the checkpoint is selected
> and saved within a run, but the model is trained fresh each time). The qualitative findings —
> the proposed model is strongest under Gaussian noise, and the gate shifts toward SAR under
> channel drop — are stable.

## Use of generative AI

Claude (Anthropic) was used for manuscript drafting and organization. All content was
reviewed and edited by the authors, who take full responsibility for it.

## License

MIT — see [`LICENSE`](LICENSE).
