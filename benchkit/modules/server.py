"""Server: real llama-server requests, the way an agent or chat client uses it.

Fixed prompts at temperature 0 with a fixed seed and a fixed output length
(ignore_eos), so every machine generates the same number of tokens. MTP
draft acceptance still varies a little run to run; report medians.

Against a remote server or one started from a custom command, MTP and the
rest are the server's own business, so there is one pass labelled "external"
or "command". Those requests are streamed: prefill and decode come from the
server's own llama.cpp-style timings when it sends them, else from arrival
timing (prefill = prompt tokens / time to first token, decode = tokens after
the first / time from first to last), which works on any OpenAI-compatible
server; extra.method says which. Each prompt gets a unique prefix so a server
that ignores cache_prompt (Strata keeps its own conversation cache) cannot
reuse an earlier read of it. Such servers may also ignore ignore_eos and stop
early; the token counts are stored beside every rate, so check them before
comparing.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from ..engines import requests_only
from ..llserver import open_server, stream_rates
from .base import Module, load_s

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

CODE_PROMPT = ("Write a complete, well-commented Python module that implements an LRU cache with "
               "TTL expiry, thread safety, statistics counters and a small CLI demo. Include type "
               "hints and unit tests.")


def prompts() -> dict[str, str]:
    prose = (PROMPTS_DIR / "prose.txt").read_text(encoding="utf-8")
    return {"prose": "Summarize the following article in detail, section by section:\n\n" + prose,
            "code": CODE_PROMPT}


class ServerModule(Module):
    name = "server"
    label = "Server (real requests)"
    description = "Real chat requests: prefill and decode tokens/s on a ~2k-token prose prompt and a code prompt."
    params = {
        "reps": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "Runs"},
        "mtp": {"type": "multi", "default": ["off", "on"], "choices": ["off", "on"], "label": "MTP"},
        "kv": {"type": "choice", "default": "q4_0", "choices": ["f16", "q8_0", "q4_0"], "label": "KV cache type"},
        "max_tokens": {"type": "int", "default": 512, "min": 64, "max": 4096, "label": "Tokens generated"},
        "ctx": {"type": "int", "default": 32768, "min": 4096, "max": 262144, "label": "Context size"},
    }

    def check(self, p, model, engine, snapshot):
        if requests_only(engine):
            return []  # the server decides about MTP
        probs = []
        if "on" in p["mtp"] and model is not None and not model.get("mtp"):
            probs.append("MTP 'on' needs a model with MTP heads")
        if "on" in p["mtp"] and not engine.get("mtp_args"):
            probs.append(f"engine {engine['name']} has no MTP support")
        return probs

    def estimate_s(self, p, model_gb):
        return int(len(p["mtp"]) * (load_s(model_gb) + 30 + p["reps"] * 2 * (10 + p["max_tokens"] / 25)))

    def run(self, ctx, p):
        P = prompts()
        external = requests_only(ctx.engine)
        variants = [ctx.engine["name"]] if external else p["mtp"]
        steps = len(variants) * p["reps"]
        step = 0
        for mtp in variants:
            ctx.cool_down()
            with open_server(ctx, ["-ctk", p["kv"], "-ctv", p["kv"]], p["ctx"], mtp=(mtp == "on")) as srv:
                warm = {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "cache_prompt": False}
                srv.stream(warm) if external else srv.chat(warm)  # warmup, discarded
                for rep in range(p["reps"]):
                    step += 1
                    ctx.progress(self.name, step, steps, f"{mtp if external else 'MTP ' + mtp}, run {rep + 1}")
                    if rep:
                        ctx.cool_down()
                    for pname, text in P.items():
                        if ctx.abort.is_set():
                            return
                        body = {"messages": [{"role": "user", "content": text}],
                                "max_tokens": p["max_tokens"], "temperature": 0, "seed": 42,
                                "ignore_eos": True, "cache_prompt": False,
                                "chat_template_kwargs": {"enable_thinking": False}}
                        if external:
                            body["messages"][0]["content"] = f"[run {uuid.uuid4().hex}] {text}"
                            self._record_stream(ctx, rep, mtp, pname, stream_rates(srv.stream(body)))
                            continue
                        j = srv.chat(body)
                        t = j.get("timings") or {}
                        var = f"mtp={mtp}"
                        ctx.result(self.name, rep, var, f"{pname}.pp", t.get("prompt_per_second"), "t/s",
                                   {"tokens": t.get("prompt_n"),
                                    **({"cached": t["cache_n"]} if t.get("cache_n") is not None else {})})
                        ctx.result(self.name, rep, var, f"{pname}.tg", t.get("predicted_per_second"), "t/s",
                                   {"tokens": t.get("predicted_n")})
                        if t.get("draft_n"):
                            ctx.result(self.name, rep, var, f"{pname}.draft_accept",
                                       round(t["draft_n_accepted"] / t["draft_n"], 4), "",
                                       {"draft_n": t["draft_n"], "accepted": t["draft_n_accepted"]})

    def _record_stream(self, ctx, rep, var, pname, r):
        """One streamed request's prefill/decode (server timings or arrival timing; extra.method says which)."""
        extra = {"method": r["method"], **({"cached": r["cached"]} if r.get("cached") is not None else {})}
        ctx.result(self.name, rep, var, f"{pname}.pp", r["pp"], "t/s", {"tokens": r["prompt_tokens"], **extra})
        ctx.result(self.name, rep, var, f"{pname}.tg", r["tg"], "t/s", {"tokens": r["gen_tokens"], "method": r["method"]})
        if r.get("draft_n"):
            ctx.result(self.name, rep, var, f"{pname}.draft_accept", round(r["draft_accepted"] / r["draft_n"], 4), "",
                       {"draft_n": r["draft_n"], "accepted": r["draft_accepted"]})
