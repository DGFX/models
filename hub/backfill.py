"""Import a finished local run into the shared job log.

From the repository root, with Modal credentials:

    python hub/backfill.py \\
      --model 4danyone \\
      --output-dir outputs/4danyone \\
      --started 2026-10-21T12:00:00Z \\
      --finished 2026-10-21T12:08:38Z \\
      --gpu A100-80GB \\
      --filename clip.mp4 \\
      --params '{"views_per_layer": 6}' \\
      --job-id backfill-4danyone-example

Passing the same ``--job-id`` again leaves the existing job in place.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.jobs import GPU_USD_PER_SECOND, write_jpeg  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a finished local run into the shared models job log.")
    parser.add_argument("--model", required=True, help="Model id, for example 4danyone")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory with grid.mp4, videos/, and output.zip")
    parser.add_argument("--started", required=True, help="UTC start time, ISO-8601")
    parser.add_argument("--finished", required=True, help="UTC finish time, ISO-8601")
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--filename", default="", help="Input filename recorded on the job")
    parser.add_argument("--params", default="", help="JSON object of key parameters")
    parser.add_argument("--thumbnail", type=Path, default=None, help="JPEG or PNG used as the input thumbnail")
    parser.add_argument("--input-video", type=Path, default=None, help="Video used only to grab an input thumbnail")
    parser.add_argument("--error", default="", help="Record the job as failed with this message")
    parser.add_argument("--job-id", default="", help="Stable id. Reusing it does not import a second copy.")
    return parser.parse_args(argv)


def parse_params(text: str) -> dict:
    if not str(text).strip():
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("params must be a JSON object")
    return loaded


def run_import(store, args: argparse.Namespace) -> tuple[dict, bool]:
    """Return the job record and whether this call wrote it."""

    job_id = str(args.job_id).strip()
    if job_id:
        existing = store.get(job_id)
        if existing:
            return existing, False
    params = parse_params(args.params)
    filename = str(args.filename).strip() or Path(args.output_dir).name
    thumbnail = Path(args.thumbnail) if args.thumbnail else None
    temp_dir: Path | None = None
    if thumbnail is None and args.input_video:
        temp_dir = Path(tempfile.mkdtemp(prefix="models-thumb-"))
        candidate = temp_dir / "input.jpg"
        if write_jpeg(Path(args.input_video), candidate):
            thumbnail = candidate
        else:
            print(f"warning: could not read a thumbnail from {args.input_video}", file=sys.stderr)
    try:
        record = store.import_finished(
            model=args.model,
            gpu=args.gpu,
            filename=filename,
            params=params,
            started_at=args.started,
            finished_at=args.finished,
            output_dir=Path(args.output_dir),
            thumbnail=thumbnail,
            job_id=job_id or None,
            error=str(args.error).strip() or None,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return record, True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu not in GPU_USD_PER_SECOND:
        print(f"warning: no list price for gpu {args.gpu}; cost will be empty", file=sys.stderr)
    try:
        from common.jobs import open_store

        store = open_store(strict=True)
        record, created = run_import(store, args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not created:
        print(f"Job {record['job_id']} already exists; leaving it in place.")
        return 0
    print(
        f"Imported job {record['job_id']} status={record['status']} "
        f"duration_seconds={record['duration_seconds']} cost_usd={record['cost_usd']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
