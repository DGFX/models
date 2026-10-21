"""Modal wrapper for 4DAnyone.

Deploy from the repository root:

    modal deploy 4danyone/modal_app.py

Run the upstream example (6 views) and save the outputs locally:

    modal run 4danyone/modal_app.py

The Gradio test UI is a separate CPU endpoint. Inference runs on one A100-80GB
and scales to zero. Weights are downloaded into the `models-4danyone-weights`
volume on first use. This file does not deploy anything at import time.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import traceback
import uuid
import zipfile
from pathlib import Path

import modal

APP_DIR = Path(__file__).resolve().parent
UPSTREAM = APP_DIR / "upstream"

APP_NAME = "4danyone"
GPU = "A100-80GB"
VOLUME_NAME = "models-4danyone-weights"
SMPLX_SECRET_NAME = "smplx-credentials"
WEIGHTS_MOUNT = "/weights"
MODEL_DIR = Path(WEIGHTS_MOUNT) / "models"
GVHMR_ROOT = Path("/opt/4DAnyone/third_party/GVHMR")
INCOMING_DIR = Path(WEIGHTS_MOUNT) / "incoming"
EXAMPLE_NAME = "10331522-uhd_2160_4096_25fps.mp4"
EXAMPLE_PATH = Path(WEIGHTS_MOUNT) / "data" / "source" / "pexels" / EXAMPLE_NAME
REPO_ROOT = "/opt/4DAnyone"

SCALEDOWN_SECONDS = 180
PREFETCH_TIMEOUT_SECONDS = 4 * 60 * 60
INFERENCE_TIMEOUT_SECONDS = 3 * 60 * 60
WEB_TIMEOUT_SECONDS = 7 * 60 * 60
PREVIEW_MAX_HEIGHT = 640

app = modal.App(APP_NAME)
weights = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
smplx_secret = modal.Secret.from_name(
    SMPLX_SECRET_NAME,
    required_keys=["SMPLX_USERNAME", "SMPLX_PASSWORD"],
)

_BASE_ENV = {
    "PYTHONPATH": REPO_ROOT,
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": f"{WEIGHTS_MOUNT}/hf-home",
    "HF_HUB_CACHE": f"{WEIGHTS_MOUNT}/hf-home/hub",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates", "git")
    .env(_BASE_ENV)
)

asset_image = base_image.pip_install("huggingface-hub==1.33.0").add_local_dir(
    UPSTREAM,
    REPO_ROOT,
    copy=True,
    ignore=["**/.git/**", "**/__pycache__/**", "**/*.pyc", "**/.DS_Store"],
)

gpu_image = (
    asset_image.apt_install(
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .env(
        {
            # validate_gvhmr shells out to git. Mark the checkout safe even if
            # the runtime uid does not own the image layer.
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(GVHMR_ROOT),
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
        }
    )
    .add_local_file(APP_DIR / "requirements-modal.txt", "/opt/requirements-modal.txt", copy=True)
    .add_local_file(APP_DIR / "pin_gvhmr.sh", "/opt/pin_gvhmr.sh", copy=True)
    .run_commands(
        "python -m pip install --timeout 300 -r /opt/requirements-modal.txt",
        "bash /opt/pin_gvhmr.sh",
    )
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates")
    .pip_install("gradio==6.20.0")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "GRADIO_TEMP_DIR": "/tmp/gradio",
            "GRADIO_ALLOWED_PATHS": f"{WEIGHTS_MOUNT},/tmp",
        }
    )
)


def as_int(name: str, value: object) -> int:
    """Accept an integer from the CLI, Gradio, or a whole-number float."""

    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{name} must be an integer, got {value!r}.")


def parse_layer_pitches(value: object) -> list[int]:
    """Parse `15`, `[15]`, or `-10,15,35` into integer pitch degrees."""

    if isinstance(value, bool):
        raise ValueError("layer_pitches must be a list of integer degrees.")
    if isinstance(value, int):
        items: list[object] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text.strip():
            raise ValueError("layer_pitches must contain at least one pitch.")
        items = [part.strip() for part in text.split(",") if part.strip()]
    else:
        raise ValueError(f"layer_pitches must be a list of integer degrees, got {value!r}.")
    if not items:
        raise ValueError("layer_pitches must contain at least one pitch.")
    return [as_int("layer pitch", item) for item in items]


def validate_view_request(
    views_per_layer: object,
    layer_pitches: object,
    start_yaw: object,
    yaw_span: object,
) -> tuple[int, list[int], int, int]:
    """Check the view layout before spending a GPU. Upstream rechecks the same rules."""

    views = as_int("views_per_layer", views_per_layer)
    pitches = parse_layer_pitches(layer_pitches)
    yaw0 = as_int("start_yaw", start_yaw)
    span = as_int("yaw_span", yaw_span)
    if views <= 0:
        raise ValueError(f"views_per_layer must be positive, got {views}.")
    if len(set(pitches)) != len(pitches):
        raise ValueError(f"layer_pitches must not contain duplicates, got {pitches}.")
    invalid = [pitch for pitch in pitches if not -15 <= pitch <= 45]
    if invalid:
        raise ValueError(f"Each layer pitch must be between -15 and 45 degrees, got {invalid}.")
    if not 1 <= span <= 360:
        raise ValueError(f"yaw_span must be between 1 and 360 degrees, got {span}.")
    total = views * len(pitches)
    if total % 6:
        raise ValueError(f"Total target views ({total}) must be divisible by 6.")
    return views, pitches, yaw0, span


def safe_filename(name: str | None) -> str:
    base = Path(name or "input.mp4").name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not base:
        base = "input.mp4"
    if "." not in base:
        base += ".mp4"
    return base[:120]


def format_error(exc: BaseException) -> str:
    message = " ".join(line.strip() for line in str(exc).splitlines())
    return f"{type(exc).__name__}: {message}"


def format_summary(summary: dict) -> str:
    lines = ["Inference finished."]
    for key in (
        "num_target_videos",
        "fps",
        "attention_backend",
        "total_pipeline_elapsed_seconds",
        "peak_vram_allocated_bytes",
        "peak_vram_reserved_bytes",
    ):
        if key in summary:
            lines.append(f"{key}: {summary[key]}")
    return "\n".join(lines)


def _plain_summary(summary: dict) -> dict:
    plain: dict[str, object] = {}
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            plain[key] = value
        else:
            plain[key] = str(value)
    return plain


def _even(value: int) -> int:
    return max(2, value - (value % 2))


def _resize_rgb(frame, width: int, height: int):
    import numpy as np
    from PIL import Image

    image = Image.fromarray(frame, mode="RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


def write_grid(video_paths: list[Path], destination: Path) -> None:
    """Tile the target-view mp4s into one preview. Runs after inference, on CPU."""

    import av
    import numpy as np

    if not video_paths:
        raise RuntimeError("Inference published no target videos to preview.")
    containers = [av.open(str(path)) for path in video_paths]
    try:
        streams = [container.streams.video[0] for container in containers]
        iterators = [container.decode(stream) for container, stream in zip(containers, streams)]
        pending = [next(iterator) for iterator in iterators]
        sample = pending[0].to_ndarray(format="rgb24")
        tile_h = _even(min(sample.shape[0], PREVIEW_MAX_HEIGHT))
        tile_w = _even(max(2, int(round(sample.shape[1] * (tile_h / sample.shape[0])))))
        count = len(video_paths)
        cols = max(1, math.ceil(math.sqrt(count)))
        rows = max(1, math.ceil(count / cols))
        rate = streams[0].average_rate or 25
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = av.open(str(destination), mode="w")
        try:
            out_stream = output.add_stream("libx264", rate=rate)
            out_stream.width = tile_w * cols
            out_stream.height = tile_h * rows
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "20", "preset": "veryfast"}
            index = 0
            while True:
                canvas = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
                for position, frame in enumerate(pending):
                    rgb = _resize_rgb(frame.to_ndarray(format="rgb24"), tile_w, tile_h)
                    row, col = divmod(position, cols)
                    y0, x0 = row * tile_h, col * tile_w
                    canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = rgb
                video_frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
                video_frame.pts = index
                for packet in out_stream.encode(video_frame):
                    output.mux(packet)
                index += 1
                nxt = []
                for iterator in iterators:
                    try:
                        nxt.append(next(iterator))
                    except StopIteration:
                        nxt = []
                        break
                if not nxt:
                    break
                pending = nxt
            for packet in out_stream.encode():
                output.mux(packet)
        finally:
            output.close()
    finally:
        for container in containers:
            container.close()


def zip_directory(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                archive.write(path, path.relative_to(root).as_posix(), compress_type=zipfile.ZIP_STORED)


def materialize_result(result: dict, destination: Path) -> dict[str, object]:
    """Unpack a remote result into videos, a grid preview, and the output zip."""

    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "output.zip"
    zip_path.write_bytes(result["zip_bytes"])
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    videos = sorted((destination / "videos").glob("*.mp4"))
    grid = destination / "grid.mp4"
    if not videos or not grid.is_file():
        raise RuntimeError(f"Result archive is missing videos or grid.mp4 under {destination}.")
    return {"videos": videos, "grid": grid, "zip": zip_path, "summary": result.get("summary", {})}


def _weights_ready(include_example: bool) -> bool:
    from fdanyone.assets import BIREFNET_DIR, BIREFNET_FILES, MODEL_FILES, SMPLX_MODEL

    for relative in (*MODEL_FILES, SMPLX_MODEL):
        if not (MODEL_DIR / relative).is_file():
            return False
    foreground = MODEL_DIR / BIREFNET_DIR
    if any(not (foreground / name).is_file() for name in BIREFNET_FILES):
        return False
    if include_example and not EXAMPLE_PATH.is_file():
        return False
    return True


def ensure_licensed_smplx() -> str:
    """Install SMPL-X from the volume or the license holder's Modal secret.

    The official files are not redistributed. `_download_official` is upstream's
    POST against https://download.is.tue.mpg.de/ after the user has registered
    and accepted the SMPL-X license.
    """

    from fdanyone.assets import SMPLX_MODEL
    from fdanyone.download import _download_official, create_classic_gvhmr_links, install_smplx
    from fdanyone.errors import AssetError

    target = MODEL_DIR / SMPLX_MODEL
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        create_classic_gvhmr_links(MODEL_DIR, GVHMR_ROOT, require_models=False, require_smplx=True)
        return "cached"

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in (INCOMING_DIR / "SMPLX_NEUTRAL.npz", INCOMING_DIR / "models_smplx_v1_1.zip"):
        if candidate.is_file():
            install_smplx(candidate, MODEL_DIR, GVHMR_ROOT)
            return f"installed from volume file {candidate.name}"

    username = os.environ.get("SMPLX_USERNAME", "").strip()
    password = os.environ.get("SMPLX_PASSWORD", "").strip()
    if not username or not password:
        raise AssetError(
            "SMPL-X is not on the weights volume. Accept the license at "
            "https://smpl-x.is.tue.mpg.de/ and create the Modal secret "
            f"`{SMPLX_SECRET_NAME}` with SMPLX_USERNAME and SMPLX_PASSWORD. "
            "Or upload the official models_smplx_v1_1.zip (or SMPLX_NEUTRAL.npz) with "
            "`modal volume put models-4danyone-weights <file> incoming/<file>`."
        )

    with tempfile.TemporaryDirectory(prefix="fdanyone-smplx-") as temporary_dir:
        archive = Path(temporary_dir) / "models_smplx_v1_1.zip"
        try:
            _download_official(username, password, archive)
            install_smplx(archive, MODEL_DIR, GVHMR_ROOT)
        except AssetError as exc:
            raise AssetError(
                f"{exc} Scripted login is sometimes rejected by the SMPL-X site. "
                "Download models_smplx_v1_1.zip in a browser after accepting the license, then run "
                "`modal volume put models-4danyone-weights models_smplx_v1_1.zip incoming/models_smplx_v1_1.zip`."
            ) from exc
    return "downloaded with SMPLX_USERNAME"


def prepare_weights(*, include_models: bool, include_example: bool) -> dict[str, str]:
    """Download missing public weights and the licensed body model onto the volume."""

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    status: dict[str, str] = {}
    if include_models:
        status["smplx"] = ensure_licensed_smplx()
        # Persist the licensed file before the ~22 GB public download.
        weights.commit()
        from fdanyone.download import ensure_models

        ensure_models(MODEL_DIR, GVHMR_ROOT)
        status["models"] = "ready"
    if include_example:
        from fdanyone.download import ensure_example_video

        path = ensure_example_video(EXAMPLE_PATH)
        status["example"] = str(path)
    weights.commit()
    print(f"Weights volume ready: {status}", flush=True)
    return status


def coerce_video(value: object) -> Path | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, dict):
        for key in ("path", "name", "video"):
            if value.get(key):
                value = value[key]
                break
        else:
            return None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"Uploaded video is not a file: {path}")
    return path


@app.function(
    image=asset_image,
    volumes={WEIGHTS_MOUNT: weights},
    secrets=[smplx_secret],
    timeout=PREFETCH_TIMEOUT_SECONDS,
    startup_timeout=10 * 60,
    scaledown_window=SCALEDOWN_SECONDS,
    max_containers=1,
    cpu=2.0,
    memory=16384,
    ephemeral_disk=16384,
)
def prefetch_assets(include_models: bool = True, include_example: bool = True) -> dict[str, str]:
    """CPU-only first-use download. Keeps the A100 idle while weights transfer."""

    return prepare_weights(include_models=include_models, include_example=include_example)


@app.cls(
    image=gpu_image,
    gpu=GPU,
    volumes={WEIGHTS_MOUNT: weights},
    secrets=[smplx_secret],
    timeout=INFERENCE_TIMEOUT_SECONDS,
    startup_timeout=20 * 60,
    scaledown_window=SCALEDOWN_SECONDS,
    max_containers=1,
    # A GPU request does not raise the default RAM or disk. The 12 GB DiT plus
    # GVHMR checkpoints need host memory, and /tmp holds the clip and the zip.
    cpu=8.0,
    memory=65536,
    ephemeral_disk=65536,
)
class Inference:
    @modal.method()
    def run(
        self,
        video_bytes: bytes | None = None,
        video_name: str | None = None,
        use_example: bool = False,
        views_per_layer: int = 6,
        layer_pitches: list[int] | str = "15",
        start_yaw: int = 0,
        yaw_span: int = 360,
    ) -> dict:
        """Run one 4DAnyone inference on the single visible A100 and pack the outputs."""

        views, pitches, yaw0, span = validate_view_request(
            views_per_layer, layer_pitches, start_yaw, yaw_span
        )
        weights.reload()
        if not _weights_ready(include_example=use_example):
            # Caller should have prefetched on CPU. This is the fallback.
            prefetch_assets.remote(include_models=True, include_example=use_example)
            weights.reload()
        if not _weights_ready(include_example=use_example):
            raise RuntimeError("Model weights are still missing after the volume download.")

        input_dir: Path | None = None
        output_parent: Path | None = None
        archive_path: Path | None = None
        try:
            if use_example:
                video_path = EXAMPLE_PATH
            else:
                if not video_bytes:
                    raise ValueError("Pass a video or set use_example=True.")
                input_dir = Path("/tmp/fdanyone-input") / uuid.uuid4().hex
                input_dir.mkdir(parents=True, exist_ok=True)
                video_path = input_dir / safe_filename(video_name)
                video_path.write_bytes(video_bytes)

            output_parent = Path("/tmp/fdanyone-output") / uuid.uuid4().hex
            output_dir = output_parent / video_path.stem
            output_parent.mkdir(parents=True, exist_ok=True)
            os.chdir(REPO_ROOT)
            # inference() configures the CUDA allocator before the first PyTorch import.
            from inference import inference

            summary = inference(
                video_path=str(video_path),
                output_dir=str(output_dir),
                views_per_layer=views,
                layer_pitches=pitches,
                start_yaw=yaw0,
                yaw_span=span,
                enable_turbo=True,
                model_dir=str(MODEL_DIR),
                gvhmr_root=str(GVHMR_ROOT),
                gpu_ids=[0],
                attention_backend="auto",
                target_fps="auto",
                start_time=0.0,
                seed=42,
            )
            videos = sorted((output_dir / "videos").glob("*.mp4"))
            if not videos:
                raise RuntimeError(f"Inference finished without videos under {output_dir / 'videos'}.")
            grid_path = output_dir / "grid.mp4"
            write_grid(videos, grid_path)
            archive_path = Path("/tmp") / f"{output_parent.name}.zip"
            zip_directory(output_dir, archive_path)
            return {
                "zip_bytes": archive_path.read_bytes(),
                "summary": _plain_summary(summary),
                "video_names": [path.relative_to(output_dir).as_posix() for path in videos],
            }
        finally:
            if input_dir is not None:
                shutil.rmtree(input_dir, ignore_errors=True)
            if output_parent is not None:
                shutil.rmtree(output_parent, ignore_errors=True)
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)


def _execute(
    *,
    video_bytes: bytes | None,
    video_name: str | None,
    use_example: bool,
    views_per_layer: int,
    layer_pitches: list[int],
    start_yaw: int,
    yaw_span: int,
) -> dict:
    prefetch_assets.remote(include_models=True, include_example=use_example)
    return Inference().run.remote(
        video_bytes=video_bytes,
        video_name=video_name,
        use_example=use_example,
        views_per_layer=views_per_layer,
        layer_pitches=layer_pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
    )


def generate(video, views_per_layer, layer_pitches, start_yaw, yaw_span):
    """Gradio callback. Yields status updates, then the videos, grid, and zip."""

    try:
        views, pitches, yaw0, span = validate_view_request(
            views_per_layer, layer_pitches, start_yaw, yaw_span
        )
        video_path = coerce_video(video)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        yield format_error(exc), None, None, None
        return
    if video_path is None:
        yield "Upload a portrait video, or click the upstream example.", None, None, None
        return

    use_example = False
    video_bytes = None
    video_name = None
    try:
        use_example = video_path.resolve() == EXAMPLE_PATH.resolve()
    except OSError:
        use_example = False
    if not use_example:
        video_bytes = video_path.read_bytes()
        video_name = video_path.name

    yield (
        "Preparing weights on CPU. The first run downloads about 22 GB into the "
        "models-4danyone-weights volume, then inference starts on one A100-80GB.",
        None,
        None,
        None,
    )
    try:
        result = _execute(
            video_bytes=video_bytes,
            video_name=video_name,
            use_example=use_example,
            views_per_layer=views,
            layer_pitches=pitches,
            start_yaw=yaw0,
            yaw_span=span,
        )
        saved = materialize_result(result, Path("/tmp/fdanyone-ui") / uuid.uuid4().hex)
    except Exception as exc:
        traceback.print_exc()
        yield format_error(exc), None, None, None
        return
    videos = saved["videos"]
    yield (
        format_summary(saved["summary"]),
        [str(path) for path in videos],
        str(saved["grid"]),
        str(saved["zip"]),
    )


def build_demo(example_video: str | None):
    import gradio as gr

    with gr.Blocks(title="4DAnyone") as demo:
        gr.Markdown(
            """
# 4DAnyone

Upload one portrait **9:16** video of a single person (1080p or higher, at least **121 frames**, mild camera motion).
The default layout is **6 views** on one full orbit (`views_per_layer=6`, pitch `15`). Total views
(`views_per_layer` times the number of pitches) must be divisible by 6. Pitches are integers from -15 to 45.
3DGS / 4DGS reconstruction is not part of this test UI.

The first request downloads the public checkpoints (about 22 GB) onto a Modal Volume. Later requests reuse them.
Inference uses one **A100-80GB** and can take several minutes after the weights are cached.
            """.strip()
        )
        with gr.Row():
            video = gr.Video(label="Source video", sources=["upload"])
            with gr.Column():
                views = gr.Number(label="views_per_layer", value=6, precision=0, minimum=1)
                pitches = gr.Textbox(
                    label="layer_pitches",
                    value="15",
                    info="Comma-separated degrees, for example 15 or -10,15,35.",
                )
                start_yaw = gr.Number(label="start_yaw", value=0, precision=0)
                yaw_span = gr.Number(label="yaw_span", value=360, precision=0, minimum=1, maximum=360)
                run = gr.Button("Generate", variant="primary")
        status = gr.Textbox(label="Status", lines=8)
        videos_out = gr.File(label="Dense view videos", file_count="multiple")
        grid = gr.Video(label="Grid preview")
        archive = gr.File(label="Output directory (.zip)")
        inputs = [video, views, pitches, start_yaw, yaw_span]
        outputs = [status, videos_out, grid, archive]
        run.click(generate, inputs=inputs, outputs=outputs)
        if example_video:
            gr.Examples(
                examples=[[example_video, 6, "15", 0, 360]],
                inputs=inputs,
                outputs=outputs,
                fn=generate,
                cache_examples=False,
                run_on_click=True,
                label=f"Upstream example ({EXAMPLE_NAME})",
            )
    demo.queue(max_size=8)
    demo.allowed_paths = [WEIGHTS_MOUNT, "/tmp"]
    demo.max_file_size = 2 * 1024**3
    return demo


def _load_example_video() -> str | None:
    try:
        prefetch_assets.remote(include_models=False, include_example=True)
        weights.reload()
    except Exception as exc:
        print(f"Example clip is unavailable: {format_error(exc)}", flush=True)
        return None
    if EXAMPLE_PATH.is_file():
        return str(EXAMPLE_PATH)
    print(f"Example clip was not written to {EXAMPLE_PATH}", flush=True)
    return None


@app.function(
    image=web_image,
    volumes={WEIGHTS_MOUNT: weights},
    timeout=WEB_TIMEOUT_SECONDS,
    startup_timeout=10 * 60,
    scaledown_window=SCALEDOWN_SECONDS,
    cpu=2.0,
    memory=16384,
    ephemeral_disk=65536,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def ui():
    """CPU-only Gradio page. Inference is a remote call to `Inference`."""

    demo = build_demo(_load_example_video())
    return demo.app


@app.local_entrypoint()
def main(
    video: str = "",
    views_per_layer: int = 6,
    layer_pitches: str = "15",
    start_yaw: int = 0,
    yaw_span: int = 360,
    output_dir: str = "outputs/4danyone",
) -> None:
    """Run inference and save the zip, grid, and view videos on this machine.

    Omit `--video` to use the upstream example clip
    `data/source/pexels/10331522-uhd_2160_4096_25fps.mp4` with `views_per_layer` 6.
    """

    views, pitches, yaw0, span = validate_view_request(
        views_per_layer, layer_pitches, start_yaw, yaw_span
    )
    use_example = not str(video).strip()
    video_bytes = None
    video_name = None
    if not use_example:
        path = Path(video).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Video not found: {path}")
        video_bytes = path.read_bytes()
        video_name = path.name
    print(
        "Preparing weights on CPU, then running 4DAnyone-Turbo on one A100-80GB. "
        f"views_per_layer={views} layer_pitches={pitches} start_yaw={yaw0} yaw_span={span}",
        flush=True,
    )
    result = _execute(
        video_bytes=video_bytes,
        video_name=video_name,
        use_example=use_example,
        views_per_layer=views,
        layer_pitches=pitches,
        start_yaw=yaw0,
        yaw_span=span,
    )
    destination = Path(output_dir).expanduser().resolve()
    saved = materialize_result(result, destination)
    print(format_summary(saved["summary"]), flush=True)
    print(f"Saved outputs to {destination}", flush=True)
