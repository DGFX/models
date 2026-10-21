"""Shared job log for model apps.

Records live in the Modal Dict ``models-jobs``. Result files live in the Modal
Volume ``models-job-outputs`` under ``jobs/<job_id>/``. Both objects are created
on first use.

``JobStore`` methods that inference calls (``create``, ``running``, ``succeed``,
``fail``, ``fail_if_open``, ``get``, ``list_jobs``) swallow errors when
``strict`` is false. A logging failure must not fail a run. ``import_finished``
and ``attach_outputs`` raise, because backfill should stop when a write fails.

Pass ``records`` and ``files`` to test against an in-memory backend. ``open_store``
is the real Modal backend and is not used by the unit tests.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

DICT_NAME = "models-jobs"
VOLUME_NAME = "models-job-outputs"

# GPU list prices in USD per second, from https://modal.com/pricing on 2026-10-21.
# Keys match Modal's `gpu=` strings. This is GPU time only: CPU, memory, idle
# scaledown, and region multipliers are not included.
GPU_USD_PER_SECOND: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "A10G": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "B300": 0.001972,
    "RTX-PRO-6000": 0.000842,
}

JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
OPEN_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"succeeded", "failed"}


def utc_now() -> str:
    return _format_utc(datetime.now(timezone.utc))


def parse_time(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seconds_between(start: str, end: str) -> float:
    return round((parse_time(end) - parse_time(start)).total_seconds(), 1)


def estimate_cost(gpu: str | None, duration_seconds: float | None) -> float | None:
    """Rough USD cost from the GPU list-price table and a duration in seconds."""

    if gpu is None or duration_seconds is None:
        return None
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        return None
    if duration < 0:
        return None
    rate = GPU_USD_PER_SECOND.get(str(gpu).strip())
    if rate is None:
        return None
    return round(duration * rate, 4)


def check_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise ValueError(f"Invalid job id: {job_id!r}")
    return job_id


def new_job_id() -> str:
    return uuid.uuid4().hex


def empty_outputs() -> dict[str, Any]:
    return {"grid": None, "still": None, "zip": None, "videos": []}


def record_key(job_id: str) -> str:
    return f"job:{check_job_id(job_id)}"


def coerce_params(value: object, *, strict_json: bool = False) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        if strict_json:
            raise ValueError("params must be a JSON object")
        return {}
    try:
        encoded = json.dumps(value)
    except TypeError as exc:
        if strict_json:
            raise ValueError("params must be JSON-serializable") from exc
        return {str(key): str(item) for key, item in value.items()}
    if len(encoded) > 8000:
        if strict_json:
            raise ValueError("params are too large")
        return {"truncated": True}
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        raise ValueError("params must be a JSON object")
    return loaded


def coerce_summary(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            plain[str(key)] = item
        else:
            plain[str(key)] = str(item)
    if len(json.dumps(plain)) > 8000:
        return {"truncated": True}
    return plain


def _clip(text: object, limit: int) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def safe_relative(remote_path: str) -> Path:
    text = str(remote_path).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("/"):
        text = text[1:]
    pure = PurePosixPath(text)
    if not pure.parts or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe output path: {remote_path!r}")
    return Path(*pure.parts)


class MappingRecords:
    """Dict-shaped adapter. Works for a memory dict and for ``modal.Dict``."""

    def __init__(self, mapping: Any):
        self.mapping = mapping

    def get(self, key: str, default: Any = None) -> Any:
        value = self.mapping.get(key, default)
        if value is None or value is default:
            return default
        return copy.deepcopy(value)

    def put(self, key: str, value: dict) -> None:
        stored = copy.deepcopy(value)
        if isinstance(self.mapping, dict):
            self.mapping[key] = stored
            return
        self.mapping.put(key, stored)

    def items(self) -> Iterator[tuple[Any, Any]]:
        pairs = list(self.mapping.items())
        for key, value in pairs:
            yield key, copy.deepcopy(value)


class DirectoryFiles:
    """Copy files into a local directory. Tests and mounted volumes both use this."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def destination(self, remote_path: str) -> Path:
        relative = safe_relative(remote_path)
        root = self.root.resolve()
        dest = (root / relative).resolve()
        if dest != root and root not in dest.parents:
            raise ValueError(f"Unsafe output path: {remote_path!r}")
        return dest

    def put_file(self, remote_path: str, source: Path) -> None:
        src = Path(source)
        if not src.is_file() or src.is_symlink():
            raise FileNotFoundError(src)
        dest = self.destination(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    def commit(self) -> None:
        return None


class MountedFiles(DirectoryFiles):
    """DirectoryFiles plus ``volume.commit()`` for a Volume mounted in a container."""

    def __init__(self, root: Path, volume: Any):
        super().__init__(root)
        self.volume = volume

    def commit(self) -> None:
        self.volume.commit()


class BatchFiles:
    """Upload via ``Volume.batch_upload`` from a machine that has not mounted the volume."""

    def __init__(self, volume: Any):
        self.volume = volume
        self._pending: list[tuple[Path, str]] = []

    def put_file(self, remote_path: str, source: Path) -> None:
        src = Path(source)
        if not src.is_file() or src.is_symlink():
            raise FileNotFoundError(src)
        remote = "/" + safe_relative(remote_path).as_posix()
        self._pending.append((src, remote))

    def commit(self) -> None:
        pending = list(self._pending)
        self._pending = []
        if not pending:
            return
        with self.volume.batch_upload(force=True) as batch:
            for local, remote in pending:
                batch.put_file(str(local), remote)


def open_store(*, mount: str | None = None, strict: bool = False) -> "JobStore":
    """Real Modal Dict + Volume. Imports Modal only when called."""

    import modal

    records = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    files: DirectoryFiles | BatchFiles
    if mount:
        files = MountedFiles(Path(mount), volume)
    else:
        files = BatchFiles(volume)
    return JobStore(records=MappingRecords(records), files=files, strict=strict)


class JobStore:
    def __init__(
        self,
        records: MappingRecords,
        files: DirectoryFiles | BatchFiles | None = None,
        *,
        strict: bool = False,
        retry_pause: float = 0.2,
        retries: int = 5,
    ):
        self.records = records
        self.files = files
        self.strict = strict
        self.retry_pause = retry_pause
        self.retries = retries

    def _guard(self, fn):
        try:
            return fn()
        except Exception as exc:
            if self.strict:
                raise
            print(f"models job log ignored: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return None

    def _load(self, job_id: str) -> dict | None:
        key = record_key(job_id)
        attempts = max(1, self.retries)
        for attempt in range(attempts):
            found = self.records.get(key)
            if found:
                return found
            if attempt + 1 < attempts and self.retry_pause:
                time.sleep(self.retry_pause)
        return None

    def create(
        self,
        *,
        model: str,
        gpu: str,
        filename: str,
        params: dict | None = None,
    ) -> str | None:
        def op() -> str:
            if not MODEL_RE.fullmatch(model):
                raise ValueError(f"Invalid model id: {model!r}")
            job_id = new_job_id()
            record = {
                "job_id": job_id,
                "model": model,
                "status": "queued",
                "created_at": utc_now(),
                "started_at": None,
                "finished_at": None,
                "input": {
                    "filename": _clip(filename or "input", 180) or "input",
                    "params": coerce_params(params or {}),
                    "thumbnail": None,
                },
                "gpu": _clip(gpu or "", 64),
                "duration_seconds": None,
                "cost_usd": None,
                "error": None,
                "persist_error": None,
                "summary": None,
                "outputs": empty_outputs(),
            }
            self.records.put(record_key(job_id), record)
            return job_id

        return self._guard(op)

    def running(self, job_id: str | None) -> None:
        def op() -> None:
            if not job_id:
                return
            current = self._load(job_id)
            if current is None or current.get("status") in TERMINAL_STATUSES:
                return
            current["status"] = "running"
            current["started_at"] = current.get("started_at") or utc_now()
            self.records.put(record_key(job_id), current)

        self._guard(op)

    def succeed(
        self,
        job_id: str | None,
        *,
        outputs: dict | None = None,
        thumbnail: str | None = None,
        summary: dict | None = None,
        persist_error: str | None = None,
    ) -> None:
        def op() -> None:
            if not job_id:
                return
            self._close(
                job_id,
                status="succeeded",
                outputs=outputs,
                thumbnail=thumbnail,
                summary=summary,
                persist_error=persist_error,
            )

        self._guard(op)

    def fail(self, job_id: str | None, error: str) -> None:
        def op() -> None:
            if not job_id:
                return
            self._close(job_id, status="failed", error=_clip(error, 2000))

        self._guard(op)

    def fail_if_open(self, job_id: str | None, error: str) -> None:
        """Mark a queued or running job failed. Leave a finished job alone."""

        def op() -> None:
            if not job_id:
                return
            current = self._load(job_id)
            if current is None or current.get("status") not in OPEN_STATUSES:
                return
            self._close(job_id, status="failed", error=_clip(error, 2000), current=current)

        self._guard(op)

    def _close(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        outputs: dict | None = None,
        thumbnail: str | None = None,
        summary: dict | None = None,
        persist_error: str | None = None,
        current: dict | None = None,
    ) -> None:
        current = current or self._load(job_id)
        if current is None:
            return
        if status == "failed" and current.get("status") == "succeeded":
            return
        finished = utc_now()
        started = current.get("started_at")
        if status == "succeeded" and not started:
            started = current.get("created_at")
            current["started_at"] = started
        current["status"] = status
        current["finished_at"] = finished
        if started:
            duration = max(0.0, seconds_between(started, finished))
            current["duration_seconds"] = duration
            current["cost_usd"] = estimate_cost(current.get("gpu"), duration)
        if error:
            current["error"] = error
        if outputs is not None:
            current["outputs"] = {
                "grid": outputs.get("grid"),
                "still": outputs.get("still"),
                "zip": outputs.get("zip"),
                "videos": list(outputs.get("videos") or []),
            }
        if thumbnail:
            current.setdefault("input", {})
            current["input"]["thumbnail"] = thumbnail
        if summary is not None:
            current["summary"] = coerce_summary(summary)
        if persist_error:
            current["persist_error"] = _clip(persist_error, 2000)
        self.records.put(record_key(job_id), current)

    def get(self, job_id: str) -> dict | None:
        def op() -> dict | None:
            if not job_id or not JOB_ID_RE.fullmatch(job_id):
                return None
            return self.records.get(record_key(job_id))

        return self._guard(op)

    def list_jobs(self, model: str | None = None, limit: int = 100) -> list[dict]:
        def op() -> list[dict]:
            jobs: list[dict] = []
            for key, value in self.records.items():
                if not str(key).startswith("job:") or not isinstance(value, dict):
                    continue
                if model and value.get("model") != model:
                    continue
                jobs.append(value)
            jobs.sort(key=lambda job: (job.get("created_at") or "", job.get("job_id") or ""), reverse=True)
            return jobs[:limit]

        found = self._guard(op)
        return found or []

    def attach_outputs(
        self,
        job_id: str,
        *,
        grid: Path | None = None,
        still: Path | None = None,
        archive: Path | None = None,
        videos: list[Path] | None = None,
        thumbnail: Path | None = None,
    ) -> dict[str, Any]:
        """Copy result files into the outputs backend and commit them."""

        check_job_id(job_id)
        if self.files is None:
            raise RuntimeError("output storage is not configured")
        outputs = empty_outputs()
        errors: list[str] = []

        def copy(src: Path | None, remote: str) -> str | None:
            if src is None:
                return None
            path = Path(src)
            if not path.is_file() or path.is_symlink():
                errors.append(f"missing {path}")
                return None
            try:
                self.files.put_file(remote, path)
            except Exception as exc:
                errors.append(f"{remote}: {type(exc).__name__}: {exc}")
                return None
            return remote

        outputs["grid"] = copy(grid, f"jobs/{job_id}/grid.mp4")
        outputs["still"] = copy(still, f"jobs/{job_id}/still.jpg")
        outputs["zip"] = copy(archive, f"jobs/{job_id}/output.zip")
        saved_videos: list[str] = []
        used: set[str] = set()
        for index, video in enumerate(videos or []):
            name = _video_name(Path(video), index)
            while name in used:
                stem = name[:-4] if name.lower().endswith(".mp4") else name
                name = f"{stem}-{len(used)}.mp4"
            used.add(name)
            remote = f"jobs/{job_id}/videos/{name}"
            saved = copy(Path(video), remote)
            if saved:
                saved_videos.append(saved)
        outputs["videos"] = saved_videos
        thumb_remote = copy(thumbnail, f"jobs/{job_id}/input.jpg") if thumbnail else None
        self.files.commit()
        return {
            "outputs": outputs,
            "thumbnail": thumb_remote,
            "persist_error": "; ".join(errors)[:2000] or None,
        }

    def import_finished(
        self,
        *,
        model: str,
        gpu: str,
        filename: str,
        params: dict | None,
        started_at: str,
        finished_at: str,
        output_dir: Path,
        thumbnail: Path | None = None,
        job_id: str | None = None,
        error: str | None = None,
    ) -> dict:
        """Write one historical job. Returns the existing record when ``job_id`` is already stored."""

        if not MODEL_RE.fullmatch(model):
            raise ValueError(f"Invalid model id: {model!r}")
        chosen = check_job_id(job_id) if job_id else new_job_id()
        existing = self.records.get(record_key(chosen))
        if existing:
            return existing
        if self.files is None:
            raise RuntimeError("output storage is not configured")
        started = _format_utc(parse_time(started_at))
        finished = _format_utc(parse_time(finished_at))
        duration = seconds_between(started, finished)
        if duration < 0:
            raise ValueError("finished is before started")
        directory = Path(output_dir)
        if not directory.is_dir():
            raise ValueError(f"output dir not found: {directory}")
        collected = collect_output_files(directory)
        if not collected["grid"] and not collected["archive"] and not collected["videos"]:
            raise ValueError(f"No grid.mp4, output.zip, or view videos in {directory}")
        still = collected["still"]
        temp_dir: Path | None = None
        if still is None and collected["grid"] is not None:
            temp_dir = Path(tempfile.mkdtemp(prefix="models-still-"))
            candidate = temp_dir / "still.jpg"
            if write_jpeg(collected["grid"], candidate):
                still = candidate
        try:
            stored = self.attach_outputs(
                chosen,
                grid=collected["grid"],
                still=still,
                archive=collected["archive"],
                videos=collected["videos"],
                thumbnail=thumbnail if thumbnail and Path(thumbnail).is_file() else None,
            )
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
        record = {
            "job_id": chosen,
            "model": model,
            "status": "failed" if error else "succeeded",
            "created_at": started,
            "started_at": started,
            "finished_at": finished,
            "input": {
                "filename": _clip(filename or directory.name, 180) or directory.name,
                "params": coerce_params(params or {}, strict_json=True),
                "thumbnail": stored.get("thumbnail"),
            },
            "gpu": _clip(gpu or "", 64),
            "duration_seconds": duration,
            "cost_usd": estimate_cost(gpu, duration),
            "error": _clip(error, 2000) if error else None,
            "persist_error": stored.get("persist_error"),
            "summary": None,
            "outputs": stored.get("outputs") or empty_outputs(),
        }
        self.records.put(record_key(chosen), record)
        return record


def collect_output_files(output_dir: Path) -> dict[str, Any]:
    """Find the files a finished run directory should publish."""

    root = Path(output_dir)
    grid = root / "grid.mp4"
    archive = root / "output.zip"
    videos_dir = root / "videos"
    videos: list[Path] = []
    if videos_dir.is_dir():
        videos = sorted(
            path
            for path in videos_dir.glob("*.mp4")
            if path.is_file() and not path.is_symlink()
        )
    if not videos:
        videos = sorted(
            path
            for path in root.glob("*.mp4")
            if path.is_file() and not path.is_symlink() and path.name != "grid.mp4"
        )
    still = root / "still.jpg"
    if not still.is_file():
        still = root / "still.jpeg"
    return {
        "grid": grid if grid.is_file() and not grid.is_symlink() else None,
        "archive": archive if archive.is_file() and not archive.is_symlink() else None,
        "videos": videos,
        "still": still if still.is_file() and not still.is_symlink() else None,
    }


def _video_name(path: Path, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name).strip("._")
    if not cleaned.lower().endswith(".mp4"):
        cleaned = (cleaned or f"view-{index}") + ".mp4"
    return cleaned[:120]


def write_jpeg(video_path: Path, dest: Path, *, max_edge: int = 480) -> bool:
    """Save the first frame as a small JPEG. Returns false when that is not possible."""

    video = Path(video_path)
    if not video.is_file() or video.is_symlink():
        return False
    try:
        if _jpeg_with_av(video, dest, max_edge):
            return True
    except Exception:
        pass
    try:
        return _jpeg_with_ffmpeg(video, dest, max_edge)
    except Exception:
        return False


def _jpeg_with_av(video: Path, dest: Path, max_edge: int) -> bool:
    import av
    from PIL import Image

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
    image = frame.to_image()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(frame.to_ndarray(format="rgb24"))
    image.thumbnail((max_edge, max_edge))
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, "JPEG", quality=80)
    return dest.is_file() and dest.stat().st_size > 0


def _jpeg_with_ffmpeg(video: Path, dest: Path, max_edge: int) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        f"scale={max_edge}:-1",
        str(dest),
    ]
    completed = subprocess.run(command, capture_output=True, timeout=30, check=False)
    return completed.returncode == 0 and dest.is_file() and dest.stat().st_size > 0
