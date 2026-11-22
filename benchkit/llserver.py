"""Start, health-check and stop one llama-server owned by a run."""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path
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


def external_key() -> str:
    """Fallback API key for remote servers, read from a file in the data dir (BENCHKIT_EXTERNAL_KEY_FILE, default
    <data dir>/external_api_key). A key typed on New Run is kept in memory for that run only. Neither is ever
    written into a run's spec, results or export."""
    path = Path(os.environ.get("BENCHKIT_EXTERNAL_KEY_FILE")
                or Path(os.environ.get("BENCHKIT_DATA_DIR", "/data")) / "external_api_key")
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def auth_headers(key: str = "") -> dict:
    key = key or external_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def api_base(url: str) -> str:
    """http://host:port for http://host:port/v1 (and https://host/api/v1 -> https://host/api)."""
    return str(url).strip().rstrip("/").removesuffix("/v1")


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


def fetch_counters(base: str, timeout: float = 10, headers: dict | None = None) -> dict[str, float] | None:
    """The server's cumulative prompt/decode counters, or None if it exposes none."""
    for path in COUNTER_PATHS:
        try:
            with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers or {}), timeout=timeout) as r:
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


def list_models(url: str, key: str = "", timeout: float = 10) -> list[str]:
    """Model ids the server offers at /v1/models. Raises on HTTP and connection errors."""
    req = urllib.request.Request(api_base(url) + "/v1/models", headers=auth_headers(key))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = (json.loads(r.read()) or {}).get("data") or []
    return [str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")]


def probe_external(url: str, timeout: float = 5, key: str = "") -> tuple[bool, str, str]:
    """(reachable, first model id, message) for an OpenAI-compatible server.

    /v1/models first (every OpenAI-compatible server has it; hosted APIs, Ollama and LM Studio have no /health),
    then /health for servers that answer it while still loading or that guard /v1/models differently."""
    base = api_base(url)
    try:
        models = list_models(base, key, timeout)
        return True, (models[0] if models else ""), "ok"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):  # definite: the server is there but refuses this key (an open /health proves nothing)
            return False, "", f"{base}/v1/models answered {e.code} (wrong or missing API key?)"
        why = f"{base}/v1/models answered {e.code}"
    except (urllib.error.URLError, OSError, ValueError, AttributeError) as e:
        why = f"{base}/v1/models did not answer: {getattr(e, 'reason', e)}"
    try:
        with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
            if r.status == 200:
                return True, "", "ok"
    except (urllib.error.URLError, OSError):
        pass
    return False, "", why


# llama.cpp request fields other OpenAI-compatible servers may reject (hosted APIs answer 400 to unknown fields)
LLAMA_ONLY = ("ignore_eos", "cache_prompt", "chat_template_kwargs", "timings_per_token")


def stream_chat(base: str, body: dict, headers: dict | None = None, timeout: float = 1800) -> dict:
    """One streamed chat request, timed on arrival: works with any OpenAI-compatible server.

    Returns ttft (s to the first generated token, answer or reasoning), gen_s (first to last token), the usage the
    server reports, its llama.cpp-style "timings" if it sends them (llama-server and Strata do, in the last chunk),
    and the text. Fields only llama.cpp knows are dropped and the request retried if the server answers 400."""
    attempts = [body, {k: v for k, v in body.items() if k not in LLAMA_ONLY},
                {k: v for k, v in body.items() if k not in LLAMA_ONLY + ("stream_options", "seed")}]
    err = None
    for i, b in enumerate(attempts):
        b = {**b, "stream": True, **({"stream_options": {"include_usage": True}} if i < 2 else {})}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        t0 = time.time()
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                err = e
                continue
            raise
        first = last = None
        n_chunks, usage, timings, text = 0, {}, None, []
        with r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                usage = ev.get("usage") or usage
                timings = ev.get("timings") or timings
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                    if piece:
                        now = time.time()
                        first = first or now
                        last = now
                        n_chunks += 1
                        if d.get("content"):
                            text.append(d["content"])
        return {"ttft": round(first - t0, 4) if first else None, "gen_s": round(last - first, 4) if first else None,
                "wall_s": round(time.time() - t0, 3), "chunks": n_chunks, "usage": usage, "timings": timings,
                "text": "".join(text), "sent": b}
    raise err  # type: ignore[misc]


def stream_rates(m: dict) -> dict:
    """Prefill/decode from stream_chat(): the server's own timings when it sends them, else arrival timing.

    Arrival timing: prefill = prompt tokens / time to first token (includes network and queueing, so it reads
    low on short prompts and remote links); decode = (tokens - 1) / (last - first token)."""
    t, u = m.get("timings") or {}, m.get("usage") or {}
    if t.get("prompt_per_second") or t.get("predicted_per_second"):
        out = {"method": "server", "prompt_tokens": t.get("prompt_n"), "pp": t.get("prompt_per_second"),
               "gen_tokens": t.get("predicted_n"), "tg": t.get("predicted_per_second")}
        if t.get("cache_n") is not None:
            out["cached"] = t["cache_n"]
        if t.get("draft_n"):
            out["draft_n"], out["draft_accepted"] = t["draft_n"], t.get("draft_n_accepted", 0)
        return out
    pt, gt = u.get("prompt_tokens"), u.get("completion_tokens") or m.get("chunks")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    new = pt - cached if pt and cached else pt
    out = {"method": "stream", "prompt_tokens": pt, "gen_tokens": gt,
           "pp": round(new / m["ttft"], 2) if new and m.get("ttft") else None,
           "tg": round((gt - 1) / m["gen_s"], 2) if gt and gt > 1 and m.get("gen_s") and m["gen_s"] > 0.01 else None}
    if cached:
        out["cached"] = cached
    return out


class ExternalServer:
    """Same interface as LlamaServer for a server the kit did not start (remote engine).

    Nothing is launched or stopped. The server's model id is sent with every request, because servers that
    host several models need it and single-model ones often still check it."""

    external = True

    def __init__(self, ctx, *_, **__):
        self.ctx = ctx
        self.base = api_base(ctx.spec["server_url"])
        self.model = ctx.spec.get("server_model") or ""
        self.key = getattr(ctx, "api_key", "") or ""

    @property
    def headers(self) -> dict:
        return auth_headers(self.key)

    def start(self) -> None:
        ok, model, msg = probe_external(self.base, key=self.key)
        if not ok:
            raise RuntimeError(f"server not reachable: {msg}")
        self.model = self.model or model
        self.ctx.log("runner", f"using server {self.base} (model {self.model or 'unknown'})")

    def chat(self, body: dict, timeout: float = 1800) -> dict:
        body = {**body, **({"model": self.model} if self.model else {})}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **self.headers})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def stream(self, body: dict, timeout: float = 1800) -> dict:
        return stream_chat(self.base, {**body, **({"model": self.model} if self.model else {})}, self.headers, timeout)

    def counters(self) -> dict[str, float] | None:
        return fetch_counters(self.base, headers=self.headers)

    def stop(self) -> None:
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        pass


def fill_command(template: str, port: int, model: str, ctx_size: int, gpus: list[int]) -> list[str]:
    """The run's launch command as argv. Only the four placeholders are replaced, so braces elsewhere (a JSON
    argument, say) pass through untouched."""
    cmd = template
    for k, v in (("{port}", str(port)), ("{model}", model), ("{ctx}", str(ctx_size)),
                 ("{gpus}", ",".join(str(g) for g in gpus))):
        cmd = cmd.replace(k, shlex.quote(v) if k == "{model}" and v else v)
    return shlex.split(cmd)


class CommandServer(ExternalServer):
    """Any engine's OpenAI-compatible server, started by the kit from the run's command line (command engine).

    Ready when /v1/models (or /health) answers; stopped, with its whole process group, when the module ends."""

    external = False

    def __init__(self, ctx, ctx_size: int, model: str = ""):
        self.ctx = ctx
        self.port = ctx.cfg.server_port
        self.base = f"http://127.0.0.1:{self.port}"
        self.model = ctx.spec.get("server_model") or ""
        self.key = ""
        self.argv = fill_command(ctx.spec["launch_cmd"], self.port, model, ctx_size, ctx.gpus)
        self.proc: Proc | None = None

    def _on_line(self, line: str) -> None:
        self.ctx.log("server", line)
        ev = parse_print_timing(line)  # llama.cpp-based engines still feed the live chart
        if ev:
            self.ctx.server_timing(ev)

    def start(self, timeout: float = 1800) -> None:
        self.ctx.log("runner", "starting: " + shlex.join(self.argv))
        env = {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in self.ctx.gpus)} if self.ctx.gpus else {}
        try:
            self.proc = Proc(self.argv, env=env, on_line=self._on_line)
        except OSError as e:
            raise RuntimeError(f"could not start the command: {e}")
        try:
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.ctx.abort.is_set():
                    raise RuntimeError("aborted")
                if not self.proc.alive():
                    tail = "\n".join(self.proc.lines[-15:])
                    raise RuntimeError(f"the server command exited during startup (rc={self.proc.p.returncode})\n{tail}")
                ok, model, _ = probe_external(self.base, timeout=3)
                if ok:
                    self.model = self.model or model
                    self.ctx.log("runner", f"server ready after {time.time() - t0:.0f}s (model {self.model or 'unknown'})")
                    return
                time.sleep(2)
            raise RuntimeError(f"the server did not answer on port {self.port} within {timeout:.0f}s")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self.proc:
            self.proc.kill()
            self.ctx.log("runner", "server stopped")
            self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def open_server(ctx, extra_args: list[str], ctx_size: int, mtp: bool = False,
                record: tuple[str, str] | None = None):
    """The run's server: a remote one, one started from the run's command, or the kit's own llama-server."""
    if ctx.engine.get("external"):
        return ExternalServer(ctx)
    if ctx.engine.get("command"):
        return CommandServer(ctx, ctx_size, str(ctx.model_path) if ctx.model_path else "")
    return LlamaServer(ctx, extra_args, ctx_size, mtp=mtp, record=record)
