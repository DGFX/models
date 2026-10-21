# Models

One directory per model, plus a shared job log and a small hub that lists both.

| Path | What it is |
| --- | --- |
| `4danyone/` | 4DAnyone on Modal. Test page: https://dgfx--4danyone-ui.us-east.modal.run |
| `common/jobs.py` | Job log any model's `modal_app.py` can import |
| `hub/` | CPU hub, Modal app name `models-hub` |
| `hub/models.json` | Registry: name, one-liner, test link, GPU, last verified |

The hub URL is printed by `modal deploy`. For this workspace it should look like:

https://dgfx--models-hub-ui.us-east.modal.run

## Job log

`common/jobs.py` writes one record per run to the Modal Dict `models-jobs` and copies result files to the Volume `models-job-outputs`.

Each record has a job id, model id, status (`queued`, `running`, `succeeded`, `failed`), UTC timestamps (`created_at`, `started_at`, `finished_at`), an input summary (filename, parameters, thumbnail path), GPU type, duration, a cost estimate, an error message, and output paths. Files land under `jobs/<job_id>/` as `input.jpg`, `grid.mp4`, `still.jpg`, `videos/*.mp4`, and `output.zip`.

The cost is GPU list price times measured GPU seconds. The table is in `common/jobs.py` and was taken from https://modal.com/pricing on 2026-10-21. It does not include CPU, memory, idle scaledown, or region multipliers.

Inference calls the store with `strict=False`. If the Dict or Volume write fails, the run still returns its result.

## Deploy

Run these on a machine that has Modal credentials, from the repository root. Do not deploy from a checkout that cannot import this branch.

```bash
modal dict create models-jobs
modal volume create models-job-outputs
modal deploy hub/modal_app.py
modal deploy 4danyone/modal_app.py
```

`modal dict create` does nothing if the Dict already exists. `modal volume create` errors if the Volume already exists; continue if that happens. The apps also call `from_name(..., create_if_missing=True)`, so a first deploy creates either object that is still missing. Deploy the hub and 4DAnyone into the same Modal environment so they share the Dict and Volume.

4DAnyone still needs its own weights volume and the `smplx-credentials` secret. Those steps are in `4danyone/README.md`. This deploy does not change any other Modal app.

The hub is CPU-only and scales to zero (`scaledown_window` 60 seconds, no `min_containers`). Token auth is off. To require a token, set `ENABLE_HUB_TOKEN = True` in `hub/modal_app.py`, create the secret, and redeploy:

```bash
modal secret create models-hub-auth HUB_TOKEN='a-long-random-string'
```

A browser can open `/?token=...` once. After that an HttpOnly cookie covers the video requests. `requires_proxy_auth=True` on the hub's `@modal.asgi_app()` is the other switch and stays off.

## Backfill

Two 4DAnyone runs from 2026-10-21 live on the deploy machine (the upstream example and Dominik's clip). Import each directory that is still on disk. Reusing `--job-id` is safe: a second run leaves the existing job in place.

Upstream example (`views_per_layer` 6 and the default pitch `15`, unless that run used different flags). The clock times below are placeholders: use the real UTC start and finish. A finish 8 minutes 38 seconds later matches the verified warm run's wall clock, not a recorded start time:

```bash
python hub/backfill.py \
  --model 4danyone \
  --output-dir outputs/4danyone-example \
  --started 2026-10-21T00:00:00Z \
  --finished 2026-10-21T00:08:38Z \
  --gpu A100-80GB \
  --filename 10331522-uhd_2160_4096_25fps.mp4 \
  --params '{"views_per_layer":6,"layer_pitches":[15],"start_yaw":0,"yaw_span":360,"use_example":true}' \
  --job-id backfill-4danyone-example
```

Dominik's clip. Replace the directory, clock times, filename, and parameters with the values from that run:

```bash
python hub/backfill.py \
  --model 4danyone \
  --output-dir outputs/4danyone-clip \
  --started 2026-10-21T00:00:00Z \
  --finished 2026-10-21T00:08:38Z \
  --gpu A100-80GB \
  --filename clip.mp4 \
  --params '{"views_per_layer":6,"layer_pitches":[15],"start_yaw":0,"yaw_span":360}' \
  --input-video /path/to/clip.mp4 \
  --job-id backfill-4danyone-clip
```

`--started` and `--finished` are UTC, ISO-8601. The hub shows them in Europe/Warsaw. `--input-video` is optional and is only used to grab a thumbnail. The output directory needs `grid.mp4`, `videos/*.mp4`, or `output.zip`.

`modal run 4danyone/modal_app.py` writes to `outputs/4danyone` unless `--output-dir` was set. If both runs used that same directory, only the later files are still there.

## Wire a new model

1. Add `your-model/modal_app.py` and deploy it under its own Modal app name.
2. Add one object to `hub/models.json` with `id`, `name`, `summary`, `test_url`, `gpu`, and `verified`. The `id` is the job's `model` field (`4danyone`, not the display name).
3. Mount this repo's `common/` into the image that runs inference, at `/opt/models-common/common`, with `add_local_dir` last and `copy=False` so the model image's install layers stay cached. Put `/opt/models-common` on `sys.path` before `import common.jobs`.
4. In that same app module, keep module-level handles so deploy creates them if needed:

   `modal.Dict.from_name("models-jobs", create_if_missing=True)`

   `modal.Volume.from_name("models-job-outputs", create_if_missing=True)`

5. Mount the Volume only on the function that has the result files. Build a `JobStore` with `strict=False`.
6. `create` the job before the run (`queued`), call `running` when the GPU method starts, then `attach_outputs` and `succeed`. On an exception, call `fail` inside the GPU method and `fail_if_open` in the caller so a crash before the GPU starts is still recorded. Catch logging errors. They must not fail inference.
7. Copy the preview the hub should play to `jobs/<job_id>/grid.mp4`, a still to `still.jpg`, view videos to `videos/`, the archive to `output.zip`, and an input thumbnail to `input.jpg`.
8. Redeploy the model app and `modal deploy hub/modal_app.py`.
9. Keep secrets out of `params`. The hub has no login while `ENABLE_HUB_TOKEN` is false; the Modal URL is the only gate.
