"""Unit tests for the shared job log. They use a fake dict and a local directory."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from common.jobs import (
    GPU_USD_PER_SECOND,
    BatchFiles,
    DirectoryFiles,
    JobStore,
    MappingRecords,
    estimate_cost,
    safe_relative,
    write_jpeg,
)


class Boom:
    def get(self, key, default=None):
        raise RuntimeError("dict down")

    def put(self, key, value):
        raise RuntimeError("dict down")

    def items(self):
        raise RuntimeError("dict down")


def make_store(root: Path, *, strict: bool = True) -> JobStore:
    root.mkdir(parents=True, exist_ok=True)
    return JobStore(
        records=MappingRecords({}),
        files=DirectoryFiles(root),
        strict=strict,
        retry_pause=0,
        retries=1,
    )


class JobStoreTests(unittest.TestCase):
    def test_create_run_succeed_records_cost_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = make_store(root)
            source = root / "src"
            source.mkdir()
            grid = source / "grid.mp4"
            grid.write_bytes(b"grid")
            video = source / "view.mp4"
            video.write_bytes(b"view")
            archive = source / "output.zip"
            archive.write_bytes(b"zip")
            thumb = source / "input.jpg"
            thumb.write_bytes(b"thumb")
            job_id = store.create(
                model="4danyone",
                gpu="A100-80GB",
                filename="clip.mp4",
                params={"views_per_layer": 6, "layer_pitches": [15]},
            )
            self.assertIsNotNone(job_id)
            store.running(job_id)
            stored = store.attach_outputs(
                job_id,
                grid=grid,
                still=None,
                archive=archive,
                videos=[video],
                thumbnail=thumb,
            )
            store.succeed(
                job_id,
                outputs=stored["outputs"],
                thumbnail=stored["thumbnail"],
                summary={"num_target_videos": 6},
            )
            job = store.get(job_id)
            self.assertEqual(job["status"], "succeeded")
            self.assertEqual(job["model"], "4danyone")
            self.assertEqual(job["gpu"], "A100-80GB")
            self.assertEqual(job["input"]["filename"], "clip.mp4")
            self.assertEqual(job["input"]["params"]["views_per_layer"], 6)
            self.assertEqual(job["input"]["thumbnail"], f"jobs/{job_id}/input.jpg")
            self.assertEqual(job["outputs"]["grid"], f"jobs/{job_id}/grid.mp4")
            self.assertEqual(job["outputs"]["zip"], f"jobs/{job_id}/output.zip")
            self.assertEqual(job["outputs"]["videos"], [f"jobs/{job_id}/videos/view.mp4"])
            self.assertIsNotNone(job["started_at"])
            self.assertIsNotNone(job["finished_at"])
            self.assertGreaterEqual(job["duration_seconds"], 0)
            self.assertEqual(job["cost_usd"], estimate_cost("A100-80GB", job["duration_seconds"]))
            self.assertEqual((root / "jobs" / job_id / "grid.mp4").read_bytes(), b"grid")
            self.assertEqual(job["summary"]["num_target_videos"], 6)
            job["status"] = "hacked"
            self.assertEqual(store.get(job_id)["status"], "succeeded")

    def test_cost_table_matches_published_a100_rate(self) -> None:
        self.assertEqual(GPU_USD_PER_SECOND["A100-80GB"], 0.000694)
        self.assertEqual(estimate_cost("A100-80GB", 518), round(518 * 0.000694, 4))
        self.assertIsNone(estimate_cost("not-a-gpu", 10))
        self.assertIsNone(estimate_cost("A100-80GB", None))

    def test_failure_before_start_has_no_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store(Path(tmp))
            job_id = store.create(model="4danyone", gpu="A100-80GB", filename="clip.mp4", params={})
            store.fail(job_id, "weights missing")
            job = store.get(job_id)
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["error"], "weights missing")
            self.assertIsNone(job["duration_seconds"])
            self.assertIsNone(job["cost_usd"])
            self.assertIsNone(job["started_at"])

    def test_failure_after_start_keeps_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store(Path(tmp))
            job_id = store.create(model="4danyone", gpu="A100-80GB", filename="clip.mp4", params={})
            store.running(job_id)
            store.fail(job_id, "boom")
            job = store.get(job_id)
            self.assertEqual(job["status"], "failed")
            self.assertIsNotNone(job["duration_seconds"])
            self.assertIsNotNone(job["cost_usd"])

    def test_fail_if_open_does_not_clobber_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store(Path(tmp))
            job_id = store.create(model="4danyone", gpu="T4", filename="clip.mp4", params={})
            store.running(job_id)
            store.succeed(job_id, outputs={"grid": None, "still": None, "zip": None, "videos": []})
            store.fail(job_id, "late")
            store.fail_if_open(job_id, "later")
            job = store.get(job_id)
            self.assertEqual(job["status"], "succeeded")
            self.assertIsNone(job["error"])

    def test_logging_failure_is_ignored_unless_strict(self) -> None:
        soft = JobStore(records=MappingRecords(Boom()), files=None, strict=False, retry_pause=0, retries=1)
        self.assertIsNone(soft.create(model="4danyone", gpu="T4", filename="a.mp4", params={}))
        soft.running(None)
        soft.fail("not a valid id", "x")
        self.assertEqual(soft.list_jobs(), [])
        strict = JobStore(records=MappingRecords(Boom()), files=None, strict=True, retry_pause=0, retries=1)
        with self.assertRaises(RuntimeError):
            strict.create(model="4danyone", gpu="T4", filename="a.mp4", params={})

    def test_list_is_newest_first_and_filterable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = make_store(root)
            out = root / "run"
            out.mkdir()
            (out / "grid.mp4").write_bytes(b"g")
            older = store.import_finished(
                model="4danyone",
                gpu="A100-80GB",
                filename="old.mp4",
                params={},
                started_at="2026-10-21T10:00:00Z",
                finished_at="2026-10-21T10:01:00Z",
                output_dir=out,
                job_id="backfill-old",
            )
            newer = store.import_finished(
                model="other",
                gpu="T4",
                filename="new.mp4",
                params={"k": 1},
                started_at="2026-10-21T12:00:00Z",
                finished_at="2026-10-21T12:05:00Z",
                output_dir=out,
                job_id="backfill-new",
            )
            self.assertEqual([job["job_id"] for job in store.list_jobs()], [newer["job_id"], older["job_id"]])
            self.assertEqual([job["job_id"] for job in store.list_jobs(model="4danyone")], ["backfill-old"])

    def test_import_is_idempotent_and_prices_the_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = make_store(root)
            out = root / "run"
            videos = out / "videos"
            videos.mkdir(parents=True)
            (out / "grid.mp4").write_bytes(b"one")
            (videos / "a.mp4").write_bytes(b"view")
            (out / "output.zip").write_bytes(b"zip")
            record = store.import_finished(
                model="4danyone",
                gpu="A100-80GB",
                filename="10331522-uhd_2160_4096_25fps.mp4",
                params={"views_per_layer": 6, "layer_pitches": [15]},
                started_at="2026-10-21T12:00:00Z",
                finished_at="2026-10-21T12:08:38Z",
                output_dir=out,
                job_id="backfill-4danyone-example",
            )
            self.assertEqual(record["duration_seconds"], 518)
            self.assertEqual(record["cost_usd"], round(518 * 0.000694, 4))
            self.assertEqual(record["created_at"], "2026-10-21T12:00:00Z")
            self.assertEqual(record["finished_at"], "2026-10-21T12:08:38Z")
            saved = root / "jobs" / "backfill-4danyone-example" / "grid.mp4"
            self.assertEqual(saved.read_bytes(), b"one")
            (out / "grid.mp4").write_bytes(b"two")
            again = store.import_finished(
                model="4danyone",
                gpu="A100-80GB",
                filename="other.mp4",
                params={},
                started_at="2026-10-21T12:00:00Z",
                finished_at="2026-10-21T12:08:38Z",
                output_dir=out,
                job_id="backfill-4danyone-example",
            )
            self.assertEqual(again["input"]["filename"], record["input"]["filename"])
            self.assertEqual(saved.read_bytes(), b"one")
            self.assertEqual(len(store.list_jobs()), 1)

    def test_unsafe_paths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            safe_relative("../secret")
        with self.assertRaises(ValueError):
            safe_relative("jobs/../../secret")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "grid.mp4"
            source.write_bytes(b"x")
            files = DirectoryFiles(Path(tmp) / "out")
            with self.assertRaises(ValueError):
                files.put_file("../secret", source)

    def test_batch_upload_uses_volume_paths(self) -> None:
        uploaded: list[tuple[str, bytes]] = []

        class Batch:
            def put_file(self, local, remote):
                uploaded.append((remote, Path(local).read_bytes()))

        class Volume:
            def batch_upload(self, force=False):
                self.force = force
                return self

            def __enter__(self):
                return Batch()

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "grid.mp4"
            source.write_bytes(b"grid")
            files = BatchFiles(Volume())
            files.put_file("jobs/abc/grid.mp4", source)
            files.commit()
        self.assertEqual(uploaded, [("/jobs/abc/grid.mp4", b"grid")])

    def test_missing_video_preview_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "still.jpg"
            self.assertFalse(write_jpeg(Path(tmp) / "missing.mp4", dest))
            self.assertFalse(dest.exists())

    def test_empty_output_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = make_store(root)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(ValueError):
                store.import_finished(
                    model="4danyone",
                    gpu="A100-80GB",
                    filename="clip.mp4",
                    params={},
                    started_at="2026-10-21T12:00:00Z",
                    finished_at="2026-10-21T12:01:00Z",
                    output_dir=empty,
                    job_id="backfill-empty",
                )


if __name__ == "__main__":
    unittest.main()
