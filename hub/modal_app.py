"""CPU-only models hub.

Deploy from the repository root:

    modal deploy hub/modal_app.py

The page lists models from ``hub/models.json`` and recent jobs from the shared
``models-jobs`` Dict. Output files are read from the ``models-job-outputs``
Volume. The app scales to zero. This file does not deploy anything at import time.

Token auth is off. Set ``ENABLE_HUB_TOKEN`` to true and create the Modal secret
``models-hub-auth`` with key ``HUB_TOKEN`` before deploying to require one.
A browser can open ``/?token=...`` once; later requests reuse an HttpOnly cookie.
``requires_proxy_auth`` on ``asgi_app`` is the other switch and stays false.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

import modal

# The deploy client only needs `modal`. The image installs FastAPI, and route
# annotations must resolve in this module's globals (nested imports do not).
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
except ImportError:
    FastAPI = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    FileResponse = HTMLResponse = PlainTextResponse = None  # type: ignore[misc, assignment]

APP_NAME = "models-hub"
DICT_NAME = "models-jobs"
VOLUME_NAME = "models-job-outputs"
OUTPUT_MOUNT = "/outputs"
HUB_AUTH_SECRET = "models-hub-auth"
# Off by default. Flip this and create the secret before deploying to require a token.
ENABLE_HUB_TOKEN = False
SCALEDOWN_SECONDS = 60
REFRESH_SECONDS = 15
JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

HUB_DIR = Path(__file__).resolve().parent
REPO_DIR = HUB_DIR.parent

app = modal.App(APP_NAME)
job_records = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
job_outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

hub_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates", "tzdata")
    .pip_install("fastapi==0.157.0")
    .env({"PYTHONPATH": "/opt/models-common", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        REPO_DIR / "common",
        "/opt/models-common/common",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/test_*.py"],
    )
    .add_local_file(HUB_DIR / "models.json", "/opt/models-hub/models.json", copy=True)
)

STYLE = """
:root {
  color-scheme: dark;
  --bg: #101218;
  --card: #181c25;
  --text: #e8eaef;
  --muted: #9aa3b5;
  --line: #2c3340;
  --accent: #9ec1ff;
  --ok: #8fd0a8;
  --run: #8eb6ff;
  --wait: #b7c0d0;
  --bad: #f0a3a3;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.45 ui-sans-serif, system-ui, sans-serif;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header, main { width: min(1080px, calc(100% - 32px)); margin: 0 auto; }
header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 28px 0 8px;
}
header strong { font-size: 20px; letter-spacing: 0.01em; }
header span, .muted, .foot { color: var(--muted); }
h1 { font-size: 22px; font-weight: 650; margin: 28px 0 12px; }
h2 { font-size: 16px; font-weight: 650; margin: 0 0 12px; }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px;
  margin: 12px 0 20px;
}
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; }
th { color: var(--muted); font-size: 12px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
tr:last-child td { border-bottom: 0; }
.toolbar { display: flex; gap: 8px; align-items: end; flex-wrap: wrap; margin-bottom: 8px; }
label { display: flex; flex-direction: column; gap: 4px; color: var(--muted); font-size: 12px; }
select, button, .button {
  background: #101218;
  color: var(--text);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 7px 10px;
  font: inherit;
}
button, .button { cursor: pointer; }
.pill {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  border: 1px solid var(--line);
}
.queued { color: var(--wait); }
.running { color: var(--run); }
.succeeded { color: var(--ok); }
.failed { color: var(--bad); }
.input { display: flex; gap: 10px; align-items: center; }
.thumb { width: 72px; height: 72px; object-fit: cover; border-radius: 8px; background: #0c0e13; }
.filename { word-break: break-all; }
.still { max-width: min(100%, 720px); border-radius: 10px; display: block; margin: 8px 0; }
video { width: min(100%, 960px); border-radius: 10px; background: #000; }
dl { display: grid; grid-template-columns: 140px 1fr; gap: 6px 12px; margin: 0; }
dt { color: var(--muted); }
dd { margin: 0; }
pre {
  white-space: pre-wrap;
  word-break: break-word;
  background: #101218;
  border-radius: 8px;
  padding: 10px;
  margin: 8px 0 0;
}
ul.files { padding-left: 18px; }
.note, .empty, .foot { color: var(--muted); }
.foot { margin: 8px 0 40px; font-size: 13px; }
.error { color: var(--bad); }
"""


def _bootstrap() -> None:
    for entry in (Path("/opt/models-common"), REPO_DIR):
        if (entry / "common" / "jobs.py").is_file() and str(entry) not in sys.path:
            sys.path.insert(0, str(entry))


def models_path() -> Path:
    baked = Path("/opt/models-hub/models.json")
    if baked.is_file():
        return baked
    return HUB_DIR / "models.json"


def load_models(path: Path) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["models"] if isinstance(data, dict) else data
    models: list[dict[str, str]] = []
    for row in rows:
        item = {key: str(row[key]) for key in ("id", "name", "summary", "test_url", "gpu", "verified")}
        if not MODEL_RE.fullmatch(item["id"]):
            raise ValueError(f"Invalid model id in {path}: {item['id']!r}")
        models.append(item)
    return models


def esc(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def safe_url(url: str) -> str | None:
    parsed = urlparse(str(url).strip())
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return str(url).strip()
    return None


def format_warsaw(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        zone = ZoneInfo("Europe/Warsaw")
    except Exception:
        zone = timezone.utc
    text = str(iso).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return str(iso)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(zone).strftime("%Y-%m-%d %H:%M:%S %Z")


def format_duration(seconds: object) -> str:
    if seconds is None:
        return "—"
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    if total < 0:
        return "—"
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_cost(value: object) -> str:
    if value is None:
        return "—"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "—"
    if amount >= 0.01:
        return f"${amount:.2f}"
    return f"${amount:.4f}"


def status_class(status: object) -> str:
    text = str(status or "")
    if text in {"queued", "running", "succeeded", "failed"}:
        return text
    return "unknown"


def media_url(job_id: str, relative: str | None) -> str | None:
    if not relative:
        return None
    prefix = f"jobs/{job_id}/"
    text = str(relative)
    if not text.startswith(prefix):
        return None
    return f"/media/{job_id}/{text[len(prefix):]}"


def resolve_media(root: Path, job_id: str, file_path: str) -> Path | None:
    if not JOB_ID_RE.fullmatch(job_id):
        return None
    pure = PurePosixPath(str(file_path).strip().replace("\\", "/"))
    if not pure.parts or pure.is_absolute() or ".." in pure.parts:
        return None
    base = root.resolve()
    candidate = (base / "jobs" / job_id / Path(*pure.parts)).resolve()
    if base not in candidate.parents:
        return None
    return candidate


def file_ready(root: Path, relative: str | None) -> bool:
    if not relative:
        return False
    pure = PurePosixPath(str(relative))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        return False
    path = (root / Path(*pure.parts)).resolve()
    base = root.resolve()
    if base not in path.parents and path != base:
        return False
    return path.is_file()


def page(title: str, body: str, *, refresh_url: str | None = None) -> str:
    refresh = ""
    if refresh_url:
        refresh = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS};url={esc(refresh_url)}">'
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<meta name=\"robots\" content=\"noindex\">"
        f"<title>{esc(title)}</title>{refresh}<style>{STYLE}</style></head><body>"
        "<header><a href=\"/\"><strong>Models hub</strong></a><span>DGFX</span></header>"
        f"<main>{body}</main></body></html>"
    )


def model_name(models: list[dict[str, str]], model_id: object) -> str:
    for model in models:
        if model["id"] == model_id:
            return model["name"]
    return str(model_id or "—")


def render_models(models: list[dict[str, str]]) -> str:
    rows = []
    for model in models:
        href = safe_url(model["test_url"])
        link = f'<a href="{esc(href)}" rel="noopener">Open test page</a>' if href else "—"
        rows.append(
            "<tr>"
            f"<td>{esc(model['name'])}</td>"
            f"<td>{esc(model['summary'])}</td>"
            f"<td>{esc(model['gpu'])}</td>"
            f"<td>{esc(model['verified'])}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )
    body = "".join(rows) or "<tr><td colspan=\"5\">No models in the registry.</td></tr>"
    return (
        "<section class=\"card\"><h2>Models</h2>"
        "<table><thead><tr><th>Model</th><th>Summary</th><th>GPU</th>"
        "<th>Last verified</th><th>Test</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def render_jobs(jobs: list[dict], models: list[dict[str, str]], root: Path, *, selected: str, refreshing: bool) -> str:
    options = [("", "All models")]
    seen = {model["id"] for model in models}
    for model in models:
        options.append((model["id"], model["name"]))
    for job in jobs:
        model_id = str(job.get("model") or "")
        if model_id and model_id not in seen and MODEL_RE.fullmatch(model_id):
            seen.add(model_id)
            options.append((model_id, model_id))
    if selected and selected not in seen:
        options.append((selected, selected))
    option_html = []
    for value, label in options:
        mark = " selected" if value == selected else ""
        option_html.append(f'<option value="{esc(value)}"{mark}>{esc(label)}</option>')
    refresh_href = "/?" + urlencode({"model": selected}) if selected else "/"
    rows = []
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        if not JOB_ID_RE.fullmatch(job_id):
            continue
        status = str(job.get("status") or "unknown")
        filename = (job.get("input") or {}).get("filename") or job_id
        thumb = (job.get("input") or {}).get("thumbnail")
        src = media_url(job_id, thumb) if file_ready(root, thumb) else None
        image = f'<img class="thumb" src="{esc(src)}" alt="">' if src else ""
        rows.append(
            "<tr>"
            f"<td><span class=\"pill {status_class(status)}\">{esc(status)}</span></td>"
            f"<td>{esc(model_name(models, job.get('model')))}</td>"
            f"<td>{esc(format_warsaw(job.get('created_at')))}</td>"
            f"<td>{esc(format_duration(job.get('duration_seconds')))}</td>"
            f"<td>{esc(format_cost(job.get('cost_usd')))}</td>"
            "<td><div class=\"input\">"
            f"{image}<a class=\"filename\" href=\"/jobs/{esc(job_id)}\">{esc(filename)}</a>"
            "</div></td></tr>"
        )
    if rows:
        table = (
            "<table><thead><tr><th>Status</th><th>Model</th><th>Created</th>"
            "<th>Duration</th><th>Cost</th><th>Input</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    elif selected:
        table = "<p class=\"empty\">No jobs for this model.</p>"
    else:
        table = "<p class=\"empty\">No jobs yet.</p>"
    note = "<p class=\"note\">Refreshing every 15s while a job is queued or running.</p>" if refreshing else ""
    return (
        "<section class=\"card\"><h2>Recent jobs</h2>"
        f"<form class=\"toolbar\" method=\"get\" action=\"/\">"
        f"<label>Model<select name=\"model\">{''.join(option_html)}</select></label>"
        "<button type=\"submit\">Filter</button>"
        f"<a class=\"button\" href=\"{esc(refresh_href)}\">Refresh</a></form>"
        f"{note}{table}</section>"
        "<p class=\"foot\">Times are Europe/Warsaw. Cost is GPU list price from "
        "modal.com/pricing (21 Oct 2026) times measured GPU time. CPU, memory, "
        "idle scaledown, and region multipliers are not included.</p>"
    )


def render_job(job: dict, models: list[dict[str, str]], root: Path) -> str:
    job_id = str(job["job_id"])
    status = str(job.get("status") or "unknown")
    filename = (job.get("input") or {}).get("filename") or job_id
    bits = [
        "<p><a href=\"/\">All jobs</a></p>",
        f"<h1>{esc(filename)}</h1>",
        "<section class=\"card\"><dl>",
        f"<dt>Status</dt><dd><span class=\"pill {status_class(status)}\">{esc(status)}</span></dd>",
        f"<dt>Model</dt><dd>{esc(model_name(models, job.get('model')))}</dd>",
        f"<dt>GPU</dt><dd>{esc(job.get('gpu') or '—')}</dd>",
        f"<dt>Created</dt><dd>{esc(format_warsaw(job.get('created_at')))}</dd>",
        f"<dt>Started</dt><dd>{esc(format_warsaw(job.get('started_at')))}</dd>",
        f"<dt>Finished</dt><dd>{esc(format_warsaw(job.get('finished_at')))}</dd>",
        f"<dt>Duration</dt><dd>{esc(format_duration(job.get('duration_seconds')))}</dd>",
        f"<dt>Cost</dt><dd>{esc(format_cost(job.get('cost_usd')))}</dd>",
        "</dl>",
    ]
    params = (job.get("input") or {}).get("params") or {}
    if params:
        bits.append(f"<h2>Parameters</h2><pre>{esc(json.dumps(params, indent=2, sort_keys=True))}</pre>")
    if job.get("error"):
        bits.append(f"<h2>Error</h2><pre class=\"error\">{esc(job.get('error'))}</pre>")
    if job.get("persist_error"):
        bits.append(f"<h2>Output note</h2><pre>{esc(job.get('persist_error'))}</pre>")
    bits.append("</section>")
    thumb = (job.get("input") or {}).get("thumbnail")
    if file_ready(root, thumb):
        bits.append(
            "<section class=\"card\"><h2>Input</h2>"
            f"<img class=\"still\" src=\"{esc(media_url(job_id, thumb))}\" alt=\"Input thumbnail\"></section>"
        )
    outputs = job.get("outputs") or {}
    grid = outputs.get("grid")
    still = outputs.get("still")
    if file_ready(root, grid) or file_ready(root, still):
        poster = esc(media_url(job_id, still)) if file_ready(root, still) else ""
        poster_attr = f' poster="{poster}"' if poster else ""
        player = ""
        if file_ready(root, grid):
            player = (
                f"<video controls playsinline preload=\"metadata\"{poster_attr} "
                f"src=\"{esc(media_url(job_id, grid))}\"></video>"
            )
        elif poster:
            player = f"<img class=\"still\" src=\"{poster}\" alt=\"Still frame\">"
        bits.append(f"<section class=\"card\"><h2>Preview</h2>{player}</section>")
    links = []
    labeled = [
        ("Grid", grid),
        ("Still frame", still),
        ("Output zip", outputs.get("zip")),
    ]
    for label, relative in labeled:
        if file_ready(root, relative):
            href = media_url(job_id, relative)
            links.append(f"<li><a href=\"{esc(href)}?download=1\">{esc(label)}</a></li>")
    for relative in outputs.get("videos") or []:
        if not file_ready(root, relative):
            continue
        href = media_url(job_id, relative)
        name = PurePosixPath(str(relative)).name
        links.append(f"<li><a href=\"{esc(href)}?download=1\">{esc(name)}</a></li>")
    if links:
        bits.append("<section class=\"card\"><h2>Downloads</h2><ul class=\"files\">" + "".join(links) + "</ul></section>")
    summary = job.get("summary")
    if isinstance(summary, dict) and summary:
        bits.append(
            "<section class=\"card\"><h2>Run summary</h2>"
            f"<pre>{esc(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>"
        )
    return "".join(bits)


def create_app(*, store, models: list[dict[str, str]], output_root: Path, token: str = "", reload_outputs=None):
    """Build the ASGI app. FastAPI is optional at import so `modal deploy` does not need it."""

    if FastAPI is None:
        raise ImportError("fastapi is required to serve the hub; the Modal image installs it")

    api = FastAPI(title="Models hub", docs_url=None, redoc_url=None)
    secret = token.strip()

    def authorized(request: Request) -> bool:
        if not secret:
            return True
        header = request.headers.get("authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not supplied:
            supplied = request.query_params.get("token", "")
        if not supplied:
            supplied = request.cookies.get("hub_token", "")
        if not supplied:
            return False
        import secrets

        return secrets.compare_digest(supplied, secret)

    def deny() -> HTMLResponse:
        return HTMLResponse(page("Unauthorized", "<p>This hub requires a token.</p>"), status_code=401)

    def finish(request: Request, response):
        if secret and authorized(request):
            response.set_cookie(
                "hub_token",
                secret,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https",
                max_age=7 * 24 * 3600,
                path="/",
            )
        if "text/html" in (response.media_type or ""):
            response.headers["Cache-Control"] = "no-store"
        return response

    def reload() -> None:
        if reload_outputs is None:
            return
        try:
            reload_outputs()
        except Exception as exc:
            print(f"models hub volume reload ignored: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    @api.get("/health")
    def health(request: Request):
        if not authorized(request):
            return deny()
        return finish(request, PlainTextResponse("ok"))

    @api.get("/")
    def index(request: Request):
        if not authorized(request):
            return deny()
        reload()
        raw = request.query_params.get("model", "").strip()
        selected = raw if MODEL_RE.fullmatch(raw) else ""
        try:
            jobs = store.list_jobs(model=selected or None)
        except Exception as exc:
            print(f"models hub job list ignored: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            jobs = []
        refreshing = any(job.get("status") in {"queued", "running"} for job in jobs)
        refresh_url = ("/?" + urlencode({"model": selected})) if selected else "/"
        body = render_models(models) + render_jobs(
            jobs, models, output_root, selected=selected, refreshing=refreshing
        )
        return finish(
            request,
            HTMLResponse(page("Models hub", body, refresh_url=refresh_url if refreshing else None)),
        )

    @api.get("/jobs/{job_id}")
    def job_page(job_id: str, request: Request):
        if not authorized(request):
            return deny()
        reload()
        if not JOB_ID_RE.fullmatch(job_id):
            return finish(request, HTMLResponse(page("Not found", "<p>Unknown job.</p>"), status_code=404))
        try:
            job = store.get(job_id)
        except Exception as exc:
            print(f"models hub job read ignored: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            job = None
        if not job:
            return finish(request, HTMLResponse(page("Not found", "<p>Unknown job.</p>"), status_code=404))
        refresh = f"/jobs/{job_id}" if job.get("status") in {"queued", "running"} else None
        title = (job.get("input") or {}).get("filename") or job_id
        return finish(
            request,
            HTMLResponse(page(str(title), render_job(job, models, output_root), refresh_url=refresh)),
        )

    @api.get("/media/{job_id}/{file_path:path}")
    def media(job_id: str, file_path: str, request: Request, download: int = 0):
        if not authorized(request):
            return deny()
        reload()
        path = resolve_media(output_root, job_id, file_path)
        if path is None or not path.is_file():
            return finish(request, HTMLResponse(page("Not found", "<p>File not found.</p>"), status_code=404))
        media_type = {
            ".mp4": "video/mp4",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".zip": "application/zip",
        }.get(path.suffix.lower())
        return finish(
            request,
            FileResponse(
                path,
                media_type=media_type,
                filename=path.name,
                content_disposition_type="attachment" if download else "inline",
            ),
        )

    return api


def _hub_secrets() -> list:
    if not ENABLE_HUB_TOKEN:
        return []
    return [modal.Secret.from_name(HUB_AUTH_SECRET, required_keys=["HUB_TOKEN"])]


@app.function(
    image=hub_image,
    volumes={OUTPUT_MOUNT: job_outputs},
    secrets=_hub_secrets(),
    timeout=60 * 60,
    startup_timeout=5 * 60,
    scaledown_window=SCALEDOWN_SECONDS,
    cpu=1.0,
    memory=2048,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def ui():
    """CPU web page. No keep-warm and no GPU."""

    _bootstrap()
    from common.jobs import JobStore, MappingRecords

    store = JobStore(records=MappingRecords(job_records), files=None, strict=False)
    token = os.environ.get("HUB_TOKEN", "").strip()
    return create_app(
        store=store,
        models=load_models(models_path()),
        output_root=Path(OUTPUT_MOUNT),
        token=token,
        reload_outputs=job_outputs.reload,
    )
