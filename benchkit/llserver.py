"""Start, health-check and stop one llama-server owned by a run."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .engines import env_for, ngl_args, split_args
from .llama_log import parse_print_timing
from .procs import Proc


class LlamaServer:
    def __init__(self, ctx, extra_args: list[str], ctx_size: int, mtp: bool = False,
                 record: tuple[str, str] | None = None):
        self.ctx = ctx
        self.record = record  # (module, variant): store every request's pp/tg rate as a result
        e = ctx.engine
        self.port = ctx.cfg.server_port
        self.argv = [f"{e['bin_dir']}/llama-server", "-m", str(ctx.model_path),
                     "--host", "127.0.0.1", "--port", str(self.port),
                     *ngl_args(ctx.spec), "-fa", "on", "-c", str(ctx_size), "-np", "1", "--jinja",
                     *split_args(e, ctx.gpus, ctx.spec.get("split")), *extra_args]
        if mtp:
            if not e.get("mtp_args"):
                raise RuntimeError(f"engine {e['name']} has no MTP support configured")
            # Keep the draft on the GPUs (the fork's mtp.sh passes -ngld 99; without it decode with MTP
            # was slower than without) and give its KV cache the same type as the main one.
            kv = {a: extra_args[i + 1] for i, a in enumerate(extra_args[:-1]) if a in ("-ctk", "-ctv")}
            self.argv += [*e["mtp_args"], "-ngld", "99",
                          *(["-ctkd", kv["-ctk"]] if "-ctk" in kv else []),
                          *(["-ctvd", kv["-ctv"]] if "-ctv" in kv else [])]
        self.proc: Proc | None = None
        self.base = f"http://127.0.0.1:{self.port}"

    def _on_line(self, line: str) -> None:
        self.ctx.log("server", line)
        ev = parse_print_timing(line)
        if ev:
            self.ctx.server_timing(ev)
            if self.record and ev.kind in ("prompt", "eval") and ev.tokens:
                self.ctx.result(self.record[0], 0, self.record[1], "pp" if ev.kind == "prompt" else "tg",
                                ev.tokens_per_s, "t/s", {"tokens": ev.tokens})

    def start(self, timeout: float = 1200) -> None:
        self.ctx.log("runner", "starting llama-server: " + " ".join(self.argv))
        self.proc = Proc(self.argv, env=env_for(self.ctx.engine, self.ctx.gpus), on_line=self._on_line)
        try:
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.ctx.abort.is_set():
                    raise RuntimeError("aborted")
                if not self.proc.alive():
                    tail = "\n".join(self.proc.lines[-15:])
                    raise RuntimeError(f"llama-server exited during startup (rc={self.proc.p.returncode})\n{tail}")
                try:
                    with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                        if r.status == 200:
                            self.ctx.log("runner", f"llama-server ready after {time.time() - t0:.0f}s")
                            return
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(2)
            raise RuntimeError("llama-server did not become healthy in time")
        except BaseException:
            self.stop()  # never leak the child if anything between launch and ready raises
            raise

    def chat(self, body: dict, timeout: float = 1800) -> dict:
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def stop(self) -> None:
        if self.proc:
            self.proc.kill()
            self.ctx.log("runner", "llama-server stopped")
            self.proc = None

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.stop()  # __exit__ is not called when __enter__ raises
            raise
        return self

    def __exit__(self, *exc):
        self.stop()


COUNTER_PATHS = ("/metrics/prometheus", "/metrics")  # Strata (its /metrics is JSON), then llama-server (needs --metrics)


def parse_counters(text: str) -> dict[str, float]:
    """llama.cpp-style Prometheus counters (llamacpp:*, and Strata's strata:*) from a /metrics text page."""
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#") and (line.startswith("llamacpp:") or line.startswith("strata:")):
            parts = line.split()
            try:
                out[parts[0]] = float(parts[1])
            except (IndexError, ValueError):
                pass
    return out


def fetch_counters(base: str, timeout: float = 10) -> dict[str, float] | None:
    """The server's cumulative prompt/decode counters, or None if it exposes none."""
    for path in COUNTER_PATHS:
        try:
            with urllib.request.urlopen(base + path, timeout=timeout) as r:
                c = parse_counters(r.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if "llamacpp:prompt_tokens_total" in c:
            return c
    return None


def counter_rates(before: dict, after: dict) -> dict:
    """Prefill and decode between two counter snapshots: tokens, seconds and tokens/s.

    Prompt tokens are the ones the server actually read (llama.cpp and Strata both leave out
    tokens reused from a cache); "cached" is the reused count where the server reports it."""
    d = lambda k: after.get(k, 0.0) - before.get(k, 0.0)
    pt, ps = d("llamacpp:prompt_tokens_total"), d("llamacpp:prompt_seconds_total")
    gt, gs = d("llamacpp:tokens_predicted_total"), d("llamacpp:tokens_predicted_seconds_total")
    out = {"prompt_tokens": int(pt), "prompt_s": round(ps, 3), "pp": round(pt / ps, 2) if pt > 0 and ps > 0 else None,
           "gen_tokens": int(gt), "gen_s": round(gs, 3), "tg": round(gt / gs, 2) if gt > 0 and gs > 0 else None}
    if "llamacpp:prompt_tokens_cached_total" in after:
        out["cached"] = int(d("llamacpp:prompt_tokens_cached_total"))
    if "strata:requests_total" in after:
        out["requests"] = int(d("strata:requests_total"))
    if "strata:drafts_offered_total" in after:
        off, acc = d("strata:drafts_offered_total"), d("strata:drafts_accepted_total")
        out["draft_accept"] = round(acc / off, 4) if off > 0 else None
        out["draft_n"], out["draft_accepted"] = int(off), int(acc)
    return out


def probe_external(url: str, timeout: float = 5) -> tuple[bool, str, str]:
    """(healthy, model id, message) for an already-running OpenAI-compatible server."""
    base = url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
            if r.status != 200:
                return False, "", f"{base}/health answered {r.status}"
    except (urllib.error.URLError, OSError) as e:
        return False, "", f"{base}/health did not answer: {e}"
    model = ""
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as r:
            data = (json.loads(r.read()) or {}).get("data") or []
            model = str(data[0].get("id") or "") if data else ""
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        pass
    return True, model, "ok"


class ExternalServer:
    """Same interface as LlamaServer for a server the kit did not start.

    Nothing is launched or stopped. The server's model id is sent with every
    request, because servers that host one model often still check it.
    """

    external = True

    def __init__(self, ctx, *_, **__):
        self.ctx = ctx
        self.base = str(ctx.spec["server_url"]).rstrip("/").removesuffix("/v1")
        self.model = ctx.spec.get("server_model") or ""

    def start(self) -> None:
        ok, model, msg = probe_external(self.base)
        if not ok:
            raise RuntimeError(f"external server not reachable: {msg}")
        self.model = self.model or model
        self.ctx.log("runner", f"using external server {self.base} (model {self.model or 'unknown'})")

    def chat(self, body: dict, timeout: float = 1800) -> dict:
        body = {**body, **({"model": self.model} if self.model else {})}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def counters(self) -> dict[str, float] | None:
        return fetch_counters(self.base)

    def stop(self) -> None:
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        pass


def open_server(ctx, extra_args: list[str], ctx_size: int, mtp: bool = False,
                record: tuple[str, str] | None = None):
    """The run's server: an external one if the run uses the external engine, else our llama-server."""
    if ctx.engine.get("external"):
        return ExternalServer(ctx)
    return LlamaServer(ctx, extra_args, ctx_size, mtp=mtp, record=record)
