"""Hub page tests plus a local import check that both Modal apps define cleanly."""

from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from common.jobs import JOB_ID_RE, DICT_NAME, VOLUME_NAME, DirectoryFiles, JobStore, MappingRecords
from hub.backfill import parse_args, run_import
from hub.modal_app import (
    DICT_NAME as HUB_DICT,
    ENABLE_HUB_TOKEN,
    JOB_ID_RE as HUB_JOB_ID_RE,
    VOLUME_NAME as HUB_VOLUME,
    create_app,
    load_models,
    models_path,
)


ROOT = Path(__file__).resolve().parents[1]


def make_store(root: Path) -> JobStore:
    return JobStore(
        records=MappingRecords({}),
        files=DirectoryFiles(root),
        strict=True,
        retry_pause=0,
        retries=1,
    )


def sample_output(root: Path) -> Path:
    out = root / "run"
    (out / "videos").mkdir(parents=True)
    (out / "grid.mp4").write_bytes(b"grid-bytes")
    (out / "videos" / "view.mp4").write_bytes(b"view-bytes")
    (out / "output.zip").write_bytes(b"zip-bytes")
    return out


def closure(fn) -> dict:
    load = fn._load
    return dict(zip(load.__code__.co_freevars, (cell.cell_contents for cell in load.__closure__)))


class HubPageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = make_store(self.root / "vol")
        self.output = sample_output(self.root)
        self.record, created = run_import(
            self.store,
            parse_args(
                [
                    "--model",
                    "4danyone",
                    "--output-dir",
                    str(self.output),
                    "--started",
                    "2026-10-21T12:00:00Z",
                    "--finished",
                    "2026-10-21T12:08:38Z",
                    "--gpu",
                    "A100-80GB",
                    "--filename",
                    "10331522-uhd_2160_4096_25fps.mp4",
                    "--params",
                    '{"views_per_layer": 6, "layer_pitches": [15]}',
                    "--job-id",
                    "backfill-4danyone-example",
                ]
            ),
        )
        self.assertTrue(created)
        self.models = load_models(models_path())
        self.client = TestClient(
            create_app(store=self.store, models=self.models, output_root=self.root / "vol")
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_registry_and_names_match_the_job_log(self) -> None:
        self.assertEqual(HUB_DICT, DICT_NAME)
        self.assertEqual(HUB_VOLUME, VOLUME_NAME)
        self.assertEqual(HUB_JOB_ID_RE.pattern, JOB_ID_RE.pattern)
        self.assertFalse(ENABLE_HUB_TOKEN)
        ids = [model["id"] for model in self.models]
        self.assertIn("4danyone", ids)
        anyone = next(model for model in self.models if model["id"] == "4danyone")
        self.assertEqual(anyone["gpu"], "A100-80GB")
        self.assertEqual(anyone["verified"], "2026-10-21")
        self.assertEqual(anyone["test_url"], "https://dgfx--4danyone-ui.us-east.modal.run")

    def test_index_shows_model_job_warsaw_time_and_cost(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        text = page.text
        self.assertIn("4DAnyone", text)
        self.assertIn("https://dgfx--4danyone-ui.us-east.modal.run", text)
        self.assertIn("10331522-uhd_2160_4096_25fps.mp4", text)
        self.assertIn("2026-10-21 14:00:00 CEST", text)
        self.assertIn("8m 38s", text)
        self.assertIn("$0.36", text)
        self.assertIn("succeeded", text)
        self.assertNotIn('http-equiv="refresh"', text)

    def test_filter_and_html_escape(self) -> None:
        self.store.create(
            model="4danyone",
            gpu="A100-80GB",
            filename="<script>alert(1)</script>",
            params={},
        )
        filtered = self.client.get("/?model=other")
        self.assertEqual(filtered.status_code, 200)
        self.assertIn("No jobs for this model.", filtered.text)
        self.assertNotIn("10331522", filtered.text)
        page = self.client.get("/")
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page.text)
        self.assertNotIn("<script>alert(1)</script>", page.text)

    def test_detail_plays_grid_and_downloads_outputs(self) -> None:
        job_id = self.record["job_id"]
        page = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn(f'src="/media/{job_id}/grid.mp4"', page.text)
        self.assertIn("controls", page.text)
        self.assertIn(f"/media/{job_id}/output.zip?download=1", page.text)
        self.assertIn(f"/media/{job_id}/videos/view.mp4?download=1", page.text)
        self.assertIn("views_per_layer", page.text)
        video = self.client.get(f"/media/{job_id}/grid.mp4")
        self.assertEqual(video.status_code, 200)
        self.assertEqual(video.content, b"grid-bytes")
        self.assertIn("inline", video.headers["content-disposition"].lower())
        self.assertEqual(video.headers["content-type"], "video/mp4")
        download = self.client.get(f"/media/{job_id}/output.zip?download=1")
        self.assertEqual(download.content, b"zip-bytes")
        self.assertIn("attachment", download.headers["content-disposition"].lower())

    def test_media_rejects_path_escape(self) -> None:
        secret = self.root / "secret.txt"
        secret.write_text("nope")
        job_id = self.record["job_id"]
        response = self.client.get(f"/media/{job_id}/../../secret.txt")
        self.assertNotEqual(response.status_code, 200)
        self.assertNotIn(b"nope", response.content)
        missing = self.client.get("/jobs/not-a-real-job")
        self.assertEqual(missing.status_code, 404)

    def test_running_job_refreshes(self) -> None:
        job_id = self.store.create(model="4danyone", gpu="A100-80GB", filename="live.mp4", params={})
        self.store.running(job_id)
        page = self.client.get("/")
        self.assertIn('http-equiv="refresh"', page.text)
        detail = self.client.get(f"/jobs/{job_id}")
        self.assertIn('http-equiv="refresh"', detail.text)
        self.assertIn("running", detail.text)

    def test_token_is_optional_and_enforced_when_set(self) -> None:
        open_page = self.client.get("/health")
        self.assertEqual(open_page.status_code, 200)
        locked = TestClient(
            create_app(
                store=self.store,
                models=self.models,
                output_root=self.root / "vol",
                token="s3cret",
            )
        )
        self.assertEqual(locked.get("/").status_code, 401)
        self.assertEqual(locked.get("/media/backfill-4danyone-example/grid.mp4").status_code, 401)
        opened = locked.get("/?token=s3cret")
        self.assertEqual(opened.status_code, 200)
        self.assertNotIn("s3cret", opened.text)
        video = locked.get("/media/backfill-4danyone-example/grid.mp4")
        self.assertEqual(video.status_code, 200)
        self.assertEqual(video.content, b"grid-bytes")
        bearer = TestClient(
            create_app(
                store=self.store,
                models=self.models,
                output_root=self.root / "vol",
                token="s3cret",
            )
        )
        self.assertEqual(bearer.get("/", headers={"Authorization": "Bearer s3cret"}).status_code, 200)
        self.assertEqual(bearer.get("/", headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_backfill_rejects_bad_params_and_repeats(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            run_import(
                self.store,
                parse_args(
                    [
                        "--model",
                        "4danyone",
                        "--output-dir",
                        str(self.output),
                        "--started",
                        "2026-10-21T12:00:00Z",
                        "--finished",
                        "2026-10-21T12:01:00Z",
                        "--params",
                        "[",
                    ]
                ),
            )
        again, created = run_import(
            self.store,
            parse_args(
                [
                    "--model",
                    "4danyone",
                    "--output-dir",
                    str(self.output),
                    "--started",
                    "2026-10-21T13:00:00Z",
                    "--finished",
                    "2026-10-21T13:01:00Z",
                    "--filename",
                    "replaced.mp4",
                    "--job-id",
                    "backfill-4danyone-example",
                ]
            ),
        )
        self.assertFalse(created)
        self.assertEqual(again["input"]["filename"], "10331522-uhd_2160_4096_25fps.mp4")


class AppDefinitionTests(unittest.TestCase):
    def test_apps_import_and_define_cleanly(self) -> None:
        import hub.modal_app as hub

        self.assertEqual(hub.app.name, "models-hub")
        ui = hub.app._local_state.functions["ui"]
        values = closure(ui)
        self.assertIsNone(values["gpu"])
        self.assertIsNone(values["min_containers"])
        self.assertEqual(values["scaledown_window"], 60)
        self.assertEqual(values["autoscaler_settings"].scaledown_window, 60)
        self.assertEqual(values["autoscaler_settings"].min_containers, 0)
        self.assertFalse(ui._webhook_config.requires_proxy_auth)
        self.assertEqual(ui._function_info.gpus, [])
        self.assertEqual({str(path) for path in ui._spec.volumes}, {"/outputs"})
        self.assertEqual(list(values["secrets"]), [])

        spec = importlib.util.spec_from_file_location("anyone_app", ROOT / "4danyone" / "modal_app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.app.name, "4danyone")
        self.assertEqual(module.JOBS_DICT_NAME, DICT_NAME)
        self.assertEqual(module.JOBS_VOLUME_NAME, VOLUME_NAME)
        inference = module.app._local_state.functions["Inference.*"]
        mounts = {str(path) for path in inference._spec.volumes}
        self.assertEqual(inference._spec.gpus, "A100-80GB")
        self.assertIn("/weights", mounts)
        self.assertIn("/job-outputs", mounts)
        prefetch = module.app._local_state.functions["prefetch_assets"]
        self.assertEqual({str(path) for path in prefetch._spec.volumes}, {"/weights"})
        anyone_ui = module.app._local_state.functions["ui"]
        self.assertNotIn("/job-outputs", {str(path) for path in anyone_ui._spec.volumes})
        self.assertIsNone(closure(anyone_ui)["gpu"])
        run = module.app._local_state.classes["Inference"]._method_partials["run"].raw_f
        self.assertIn("job_id", inspect.signature(run).parameters)


if __name__ == "__main__":
    unittest.main()
