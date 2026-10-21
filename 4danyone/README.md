# 4DAnyone on Modal

4DAnyone turns one casual monocular video of a person into consistent multiview videos. This directory vendors that inference code and serves it from a scale-to-zero Modal app with a small Gradio test page. 3DGS / 4DGS reconstruction is not included.

- Upstream code: https://github.com/ant-research/4DAnyone (`6dcf626d74bd95c42792f86b1d55e3841075c10c`, Apache-2.0)
- Motion recovery: https://github.com/zju3dv/GVHMR (`6ec3ca39336c50492c0fae65fba2fb831fc7d866`), checked out under `upstream/third_party/GVHMR`
- Weights: https://huggingface.co/AntResearch/4DAnyone (public; about 22 GB including GVHMR, Wan, BiRefNet, and the Turbo LoRA)
- Project page: https://4danyone.github.io/

The vendored tree is the pinned snapshot. `pin_gvhmr.sh` turns GVHMR into a git checkout during the image build because upstream refuses to run unless `git rev-parse HEAD` works inside that directory. If that fetch cannot reach GitHub, the script records a local commit on the vendored tree instead. Inference still runs; the recorded SHA will not match upstream.

## What runs where

| Piece | Modal object | Resources |
| --- | --- | --- |
| Weight download | `prefetch_assets` | CPU, volume `models-4danyone-weights` |
| Inference | `Inference.run` | one `A100-80GB`, 8 CPU cores, 64 GiB RAM |
| Test page | `ui` | CPU, Gradio |

Nothing is kept warm. `scaledown_window` is 180 seconds and `min_containers` is unset. The GPU class allows one container. Turbo denoising is on (upstream default). Attention is `auto`, which uses PyTorch SDPA here: FlashAttention-3 and SageAttention are not installed, and FlashAttention-3 does not run on A100.

## SMPL-X license

GVHMR needs `SMPLX_NEUTRAL.npz` from the official SMPL-X release. That file is not in the public Hugging Face repo and is not committed here. Register and accept the license at https://smpl-x.is.tue.mpg.de/, download `models_smplx_v1_1.zip` in a browser, and upload it before the first inference. This is the expected path:

```bash
modal volume put models-4danyone-weights models_smplx_v1_1.zip incoming/models_smplx_v1_1.zip
```

The app keeps `models/smplx/SMPLX_NEUTRAL.npz` from that archive on the volume. `incoming/SMPLX_NEUTRAL.npz` is accepted as well. The 2026-10-21 verification used that file: `SMPLX_NEUTRAL.npz` uploaded to `incoming/SMPLX_NEUTRAL.npz`. Do not commit either file.

The functions also reference a Modal secret, `smplx-credentials`, with `SMPLX_USERNAME` and `SMPLX_PASSWORD`. Create it so `modal deploy` can resolve the secret:

```bash
modal secret create smplx-credentials \
  SMPLX_USERNAME='you@example.com' \
  SMPLX_PASSWORD='your-smplx-password'
```

Posting those credentials to `https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip` returns the HTML sign-in page even with a valid account, so the secret does not install SMPL-X. Use the `modal volume put` command above.

## Deploy and test

From the repository root, after the secret exists and the SMPL-X archive is on the volume:

```bash
modal deploy 4danyone/modal_app.py
```

```bash
modal run 4danyone/modal_app.py
```

The second command runs the upstream example `data/source/pexels/10331522-uhd_2160_4096_25fps.mp4` with `views_per_layer` 6 and writes `output.zip`, `grid.mp4`, and `videos/*.mp4` under `outputs/4danyone`. A local portrait clip:

```bash
modal run 4danyone/modal_app.py \
  --video /path/to/portrait.mp4 \
  --views-per-layer 6 \
  --layer-pitches '[-10,15,35]' \
  --start-yaw 0 \
  --yaw-span 360 \
  --output-dir outputs/4danyone
```

Quote `--layer-pitches`. Total views (`views_per_layer` times the number of pitches) must be divisible by 6. Each pitch is an integer from -15 to 45.

## Job log

UI runs and `modal run` write a job to the shared `models-jobs` Dict. The GPU container copies `grid.mp4`, a still frame, the view videos, `output.zip`, and an input thumbnail to the `models-job-outputs` Volume under `jobs/<job_id>/`. The models hub reads that log. If the Dict or Volume write fails, inference still returns the zip.

## Test UI

Deployed: https://dgfx--4danyone-ui.us-east.modal.run

The page takes a 9:16 upload, `views_per_layer` (default 6), `layer_pitches`, `start_yaw`, and `yaw_span`. Clicking the upstream example fills those fields and starts the run. Results are the dense view videos, a grid preview mp4, and a zip of the output directory (videos, skeletons, cameras, GVHMR motion, metadata, and `grid.mp4`).

## GPU, time, cost

- GPU: `A100-80GB` (8 CPU cores, 64 GiB RAM). No H100 and no multi-GPU. A GPU request does not raise Modal's default memory, so the CPU and RAM requests are set on the function. Ephemeral disk stays at Modal's default (explicit requests must be at least 512 GiB).
- Run time: a warm 6-view run of the upstream example took about 518 s wall clock. Inference was about 482 s: about 49 s preprocess, about 1 m 46 s denoise, and about 17 s per view to write. Peak VRAM was 21.9 GB allocated and 33.3 GB reserved. The first run also downloads the weights.
- Cost: about $0.50 per 6-view run at list price, up to about $0.85 with a region multiplier.
- Last verified: 2026-10-21, commit `6e780fd`, upstream example, `views_per_layer` 6, on one `A100-80GB`.
- Test UI: https://dgfx--4danyone-ui.us-east.modal.run

## Layout

- `upstream/` — pinned 4DAnyone and GVHMR source. Inference is `upstream/inference.py`; this wrapper does not reimplement it.
- `modal_app.py` — volume, CPU prefetch, GPU inference, Gradio UI, and `modal run` entrypoint.
- `requirements-modal.txt` — pins for the GPU image. GVHMR's own `requirements.txt` is intentionally not installed.
- `pin_gvhmr.sh` — image-build checkout of the pinned GVHMR commit.
