import hashlib
import json
from pathlib import Path

import pytest
import yaml

from benchkit import export, hw, telemetry
from benchkit.db import DB
from benchkit.guardrails import Guardrails
from benchkit.llama_log import parse_print_timing
from benchkit.models import ModelStore, load_manifest
from benchkit.modules.base import Module
from benchkit.modules.speed import metric_name, parse_bench_jsonl

ROOT = Path(__file__).resolve().parents[1]


# ---------------- parsers ----------------
def test_print_timing_real_kmic_lines():
    lines = [
        "1.41.518.006 I slot print_timing: id  0 | task 0 | prompt eval time =    1262.49 ms /    55 tokens (   22.95 ms per token,    43.56 tokens per second)",
        "1.41.518.014 I slot print_timing: id  0 | task 0 |        eval time =     620.85 ms /    28 tokens (   22.99 ms per token,    43.49 tokens per second)",
        "1.41.518.032 I slot print_timing: id  0 | task 0 | draft acceptance = 0.66667 (   18 accepted /    27 generated), mean len =  3.00",
        "1.49.336.621 I slot print_timing: id  0 | task 13 | n_gen =    122, tg =  39.80 t/s, tg_3s =  40.12 t/s",
    ]
    evs = [parse_print_timing(l) for l in lines]
    assert [e.kind if e else None for e in evs] == ["prompt", "eval", "draft", None]
    assert evs[0].tokens == 55 and evs[0].tokens_per_s == 43.56
    assert evs[2].acceptance == 0.66667 and evs[2].accepted == 18


def test_bench_jsonl_and_metric_names():
    lines = ["noise", json.dumps({"n_prompt": 512, "n_gen": 0, "n_depth": 16384, "avg_ts": 349.9}),
             json.dumps({"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 26.2}), "{bad json"]
    rows = parse_bench_jsonl(lines)
    assert [metric_name(r) for r in rows] == ["pp512@16384", "tg128@0"]


def test_gpu_csv_and_sample():
    g = hw.parse_gpu_csv("0, Tesla P100-PCIE-16GB, GPU-x, 00000000:03:00.0, 580.1, 86.0, 16384, 250.00, 125.00, 250.00, 250.00, 3, 3, 16, 16, 1328, 715")
    assert g[0]["name"] == "Tesla P100-PCIE-16GB" and g[0]["memory_total"] == 16384 and g[0]["power_limit"] == 250.0
    s = telemetry.parse_gpu_sample("1, 55, 1328, 715, 140.2, 98, 60, 9000, [N/A], 3, 16, P0, 0x0")
    assert s["1"]["sm_mhz"] == 1328 and s["1"]["fan_pct"] is None and s["1"]["pstate"] == "P0"


def test_dmidecode_and_cpuinfo():
    text = """Handle 0x0040, DMI type 17, 40 bytes
Memory Device
\tSize: 8 GB
\tLocator: DIMM_A1
\tType: DDR3
\tSpeed: 1600 MT/s
\tConfigured Memory Speed: 1333 MT/s

Handle 0x0041, DMI type 17, 40 bytes
Memory Device
\tSize: No Module Installed
\tLocator: DIMM_A2
"""
    d = hw.parse_dmidecode_memory(text)
    assert len(d) == 1 and d[0]["type"] == "DDR3" and d[0]["configured_speed"] == "1333 MT/s"
    c = hw.parse_cpuinfo("processor : 0\nmodel name : Xeon\nphysical id : 0\ncore id : 0\n\n"
                         "processor : 1\nmodel name : Xeon\nphysical id : 0\ncore id : 0\n")
    assert c == {"model": "Xeon", "sockets": 1, "cores": 1, "threads": 2}


# ---------------- guardrails ----------------
def test_dmi_dump_points_entry_at_offset_32():
    table = b"\x11" * 40
    sm2 = bytearray(b"_SM_" + bytes(27))
    sm2[5] = 31
    sm2[0x10:0x15] = b"_DMI_"
    sm2[0x18:0x1C] = (0xF0000).to_bytes(4, "little")
    sm3 = bytearray(b"_SM3_" + bytes(19))
    sm3[6] = 24
    sm3[0x10:0x18] = (0x7F000000).to_bytes(8, "little")
    for ep, at, width in ((sm2, 0x18, 4), (sm3, 0x10, 8)):
        d = hw.dmi_dump(bytes(ep), table)
        assert len(d) == 32 + len(table) and d[32:] == table
        assert int.from_bytes(d[at:at + width], "little") == 32
        assert sum(d[:ep[5] if ep[:4] == b"_SM_" else ep[6]]) % 256 == 0
    assert sum(hw.dmi_dump(bytes(sm2), table)[0x10:0x1F]) % 256 == 0
    assert hw.dmi_dump(b"junk", table) is None


def test_guardrails_sustain_once_and_recover():
    g = Guardrails(busy_util_pct=50, low_clock_mhz=1000, thermal_warn_c=80, sustain_s=10)
    busy_low = {"util_pct": 99, "sm_mhz": 405, "temp_c": 50}
    fired = []
    for t in range(0, 30, 2):
        fired += g.check({"t": t, "gpus": {"0": busy_low}})
    assert [a.kind for a in fired] == ["low_clock"]  # once, not every sample
    rec = g.check({"t": 31, "gpus": {"0": {"util_pct": 99, "sm_mhz": 1328, "temp_c": 50}}})
    assert [a.kind for a in rec] == ["recovered"]
    # short spike below the sustain window never fires
    g2 = Guardrails(sustain_s=10)
    assert not g2.check({"t": 0, "gpus": {"0": {"temp_c": 85}}})
    assert not g2.check({"t": 5, "gpus": {"0": {"temp_c": 85}}})
    assert not g2.check({"t": 6, "gpus": {"0": {"temp_c": 60}}})
    # idle GPU at idle clock is fine
    assert not Guardrails(sustain_s=0).check({"t": 0, "gpus": {"0": {"util_pct": 0, "sm_mhz": 405}}})


# ---------------- module params ----------------
def test_module_param_validation():
    from benchkit.modules import MODULES
    sp = MODULES["speed"]
    p = sp.normalize({"reps": 2, "kv": ["f16"], "depths": [16384, 0, 0]})
    assert p["reps"] == 2 and p["depths"] == [0, 16384] and p["pp"] == 512
    with pytest.raises(ValueError):
        sp.normalize({"reps": 0})
    with pytest.raises(ValueError):
        sp.normalize({"kv": ["q3_0"]})
    with pytest.raises(ValueError):
        sp.normalize({"bogus": 1})
    ag = MODULES["agent"].defaults()
    assert "err_big_file_read" not in ag["tasks"] and "no_tool_required" not in ag["tasks"]


# ---------------- models ----------------
def test_manifest_is_valid():
    ms = load_manifest(ROOT / "models" / "manifest.yaml")
    ids = [m["id"] for m in ms]
    assert len(ids) == len(set(ids))
    for m in ms:
        assert m["kind"] in ("dense", "moe")
        for f in m["files"]:
            assert len(f["sha256"]) == 64 and f["size"] > 0


def test_model_store_adopts_renamed_file_and_verifies(tmp_path):
    content = b"GGUF" + b"x" * 5000
    sha = hashlib.sha256(content).hexdigest()
    man = tmp_path / "m.yaml"
    man.write_text(yaml.safe_dump({"schema": 1, "models": [{
        "id": "t", "label": "T", "repo": "x/y", "kind": "dense", "mtp": False, "vram_gb": 1,
        "files": [{"name": "official.gguf", "size": len(content), "sha256": sha}]}]}))
    models = tmp_path / "models"
    models.mkdir()
    (models / "renamed-by-user.gguf").write_bytes(content)
    (models / "other.gguf").write_bytes(b"GGUF")
    st = ModelStore(models, tmp_path / "data", man)
    s = st.status(st.get("t"))
    assert s["present"] and not s["verified"]
    st.verify("t")
    import time
    for _ in range(50):
        if st.jobs["t"].status != "running":
            break
        time.sleep(0.05)
    assert st.jobs["t"].status == "done"
    assert st.status(st.get("t"))["verified"]
    lst = st.list()
    assert [c["label"] for c in lst["custom"]] == ["other.gguf"]
    p, comparable = st.model_path("custom:other.gguf")
    assert p.name == "other.gguf" and not comparable
    assert st.model_path("custom:../../etc/passwd")[0] is None


def test_partial_size_match_does_not_hide_custom_split_model(tmp_path):
    first, last = b"GGUF" + b"a" * 100, b"GGUF" + b"z" * 300
    man = tmp_path / "m.yaml"
    man.write_text(yaml.safe_dump({"schema": 1, "models": [{
        "id": "q2", "label": "Q2", "repo": "x/y", "kind": "moe", "mtp": False, "vram_gb": 1,
        "files": [{"name": f"Q2-0000{i}-of-00003.gguf", "size": n, "sha256": "0" * 64}
                  for i, n in ((1, len(first)), (2, 999), (3, len(last)))]}]}))
    models = tmp_path / "models" / "q4"
    models.mkdir(parents=True)
    # same sizes as the Q2's first and last shard, different middle shard: a different quant
    (models / "Q4-00001-of-00003.gguf").write_bytes(first)
    (models / "Q4-00002-of-00003.gguf").write_bytes(b"GGUF" + b"m" * 500)
    (models / "Q4-00003-of-00003.gguf").write_bytes(last)
    st = ModelStore(tmp_path / "models", tmp_path / "data", man)
    lst = st.list()
    assert not lst["standard"][0]["present"]
    assert [c["id"] for c in lst["custom"]] == ["custom:q4/Q4-00001-of-00003.gguf"]


# ---------------- db + export ----------------
def test_db_results_and_markdown(tmp_path):
    db = DB(tmp_path / "x.sqlite")
    db.create_run("r1", {"model": "m", "engine": "e", "gpus": [0], "modules": {}, "comparable": True}, 60)
    for rep, v in enumerate([10.0, 12.0, 11.0]):
        db.add_result("r1", "speed", rep, "kv=q4_0", "tg128@0", v, "t/s")
    db.add_telemetry("r1", [{"t": 1, "gpus": {"0": {"temp_c": 50, "sm_mhz": 1328}}},
                            {"t": 2, "gpus": {"0": {"temp_c": 60, "sm_mhz": 1300}}}])
    db.update_run("r1", status="done", snapshot_json=json.dumps(
        {"gpus": [{"index": 0, "name": "P100", "uuid": "GPU-secret", "pci_bus_id": "0000:03:00.0"}],
         "memory": {"total_gb": 16, "type": "DDR3", "dimms": [{"size": "8 GB", "part_number": "SERIAL1"}]}}))
    b = export.bundle(db, "r1", "test")
    export.validate_bundle(b)
    assert b["summary"]["speed"]["kv=q4_0"]["tg128@0"]["median"] == 11.0
    assert "GPU-secret" not in json.dumps(b) and "SERIAL1" not in json.dumps(b)
    assert b["telemetry_summary"]["gpu0.temp_c"]["max"] == 60
    md = export.markdown(b)
    assert "| kv=q4_0 | tg128@0 | 11.0 t/s |" in md and "1x P100" in md
    with pytest.raises(ValueError):
        db.update_run("r1", bogus=1)
    assert db.mark_unfinished_interrupted() == []


def test_build_tables_skips_custom_and_invalid(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import build_tables

    def b(rid, comparable=True, invalid=False, tg=26.0):
        return {"bundle_schema": 1, "run": {"id": rid, "invalid": invalid, "invalid_reason": "x" if invalid else None},
                "spec": {"model": "qwen38-27b-q6kxl", "engine": "kmic", "gpus": [0, 1], "comparable": comparable},
                "snapshot": {"gpus": [{"name": "Tesla P100"}] * 2, "memory": {"total_gb": 16, "type": "DDR3"}},
                "summary": {"speed": {"kv=q4_0": {"tg128@0": {"median": tg, "min": tg, "max": tg, "n": 3, "unit": "t/s"}}}}}
    (tmp_path / "p100x2").mkdir()
    for name, bb in {"a": b("a"), "b": b("b", tg=38.0), "c": b("c", comparable=False), "d": b("d", invalid=True)}.items():
        (tmp_path / "p100x2" / f"{name}.json").write_text(json.dumps(bb))
    (tmp_path / "junk.json").write_text("{}")
    md = build_tables.build(tmp_path)
    assert "2 runs." in md
    assert md.index("| 38.0 |") < md.index("| 26.0 |")  # best first
    assert "custom model" in md and "marked invalid" in md and "junk.json" in md


def test_split_args():
    from benchkit.engines import split_args
    e = {"multi_gpu_split": "tensor"}
    assert split_args(e, [0]) == ["-sm", "none"]
    assert split_args(e, [0, 1]) == ["-sm", "tensor", "-ts", "1/1"]
    assert split_args(e, [0, 1, 2, 3], "layer") == ["-sm", "layer", "-ts", "1/1/1/1"]
    assert split_args(e, [0, 1], "bogus") == ["-sm", "tensor", "-ts", "1/1"]


def test_throughput_weights_by_tokens():
    rows = [
        {"module": "speed", "variant": "kv=q4_0", "metric": "pp512@0", "value": 400.0, "unit": "t/s"},
        {"module": "speed", "variant": "kv=q4_0", "metric": "tg128@0", "value": 20.0, "unit": "t/s"},
        {"module": "speed", "variant": "kv=q4_0", "metric": "tg128@4096", "value": 30.0, "unit": "t/s"},
        {"module": "server", "variant": "mtp=on", "metric": "code.tg", "value": 50.0, "unit": "t/s", "extra": {"tokens": 500}},
        {"module": "server", "variant": "mtp=on", "metric": "code.draft_accept", "value": 0.7, "unit": ""},
        # a 10-token reply at 10 t/s barely moves the average: 510 tokens in 11 s
        {"module": "agent", "variant": "mtp=on", "metric": "tg", "value": 50.0, "unit": "t/s", "extra": {"tokens": 500}},
        {"module": "agent", "variant": "mtp=on", "metric": "tg", "value": 10.0, "unit": "t/s", "extra": {"tokens": 10}},
        {"module": "agent", "variant": "all", "metric": "total.wall", "value": 99.0, "unit": "s"},
    ]
    tp = export.throughput(rows)
    assert tp["speed"]["kv=q4_0"]["gen"] == {"avg": 25.0, "max": 30.0, "n": 2}
    assert tp["speed"]["kv=q4_0"]["prompt"]["avg"] == 400.0
    assert tp["server"]["mtp=on"] == {"gen": {"avg": 50.0, "max": 50.0, "n": 1}}
    assert tp["agent"]["mtp=on"]["gen"] == {"avg": round(510 / 11, 2), "max": 50.0, "n": 2}
    assert "all" not in tp.get("agent", {})


def test_power_cap_text_and_gpu_fields():
    snap = {"gpus": [{"index": 0, "power_limit": 150.0}, {"index": 1, "power_limit": 150.0},
                     {"index": 2, "power_limit": 250.0}]}
    assert export.power_cap_text(snap) == "150/250 W"
    assert export.power_cap_text(snap, [0, 1]) == "150 W"
    assert export.power_cap_text({"gpus": []}) == ""
    g = hw.parse_gpu_csv("0, Tesla P100-PCIE-16GB, GPU-x, 00000000:01:00.0, 580.1, 86.00.3B.00.01, 16384, "
                         "150.00, 125.00, 250.00, 250.00, 3, 3, 8, 16, 1328, 715")[0]
    assert (g["power_limit"], g["power_min_limit"], g["power_max_limit"], g["power_default_limit"]) == (150, 125, 250, 250)


# ---------------- ngl (GPU layers) ----------------
def test_ngl_args():
    from benchkit.engines import ngl_args
    assert ngl_args({}) == ["-ngl", "99"]
    assert ngl_args({"ngl": None}) == ["-ngl", "99"]
    assert ngl_args({"ngl": 0}) == ["-ngl", "0"]
    assert ngl_args({"ngl": 44}) == ["-ngl", "44"]


def _runner(env):
    from benchkit.config import Config
    from benchkit.db import DB
    from benchkit.models import ModelStore
    from benchkit.runner import Runner
    cfg = Config()
    return Runner(cfg, DB(cfg.db_path), ModelStore(cfg.models_dir, cfg.data_dir, cfg.manifest))


def test_validate_ngl(env):
    r = _runner(env)
    base = {"model": "custom:tiny-custom.gguf", "engine": "fake", "gpus": [0], "modules": {"speed": {}}}
    assert r.validate(base)["ngl"] is None
    assert r.validate({**base, "ngl": ""})["ngl"] is None
    assert r.validate({**base, "ngl": "44"})["ngl"] == 44
    assert r.validate({**base, "ngl": 44})["ngl"] == 44
    for bad in (-1, 1000, "abc", 44.5, "44.5", True, False):
        with pytest.raises(ValueError):
            r.validate({**base, "ngl": bad})


# ---------------- procs ----------------
def _pgid_gone(pgid: int) -> bool:
    import os
    import time
    for _ in range(50):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_proc_kill_kills_children_of_a_dead_leader():
    from benchkit.procs import Proc
    p = Proc(["sh", "-c", "sleep 300 & exit 0"])
    assert p.wait(timeout=10) == 0  # the leader exits at once; the sleep lives on in its group
    p.kill()
    assert _pgid_gone(p.pid)


def test_proc_kill_sigkills_group_when_leader_is_dead():
    from benchkit.procs import Proc
    # the leader exits at once; the background sleep inherits the ignored SIGTERM,
    # so only the SIGKILL step (which must look at the group, not the leader) ends it
    p = Proc(["sh", "-c", "trap '' TERM; sleep 300 & exit 0"])
    assert p.wait(timeout=10) == 0
    p.kill(grace=1)
    assert _pgid_gone(p.pid)


def test_proc_context_manager_kills_on_raise():
    from benchkit.procs import Proc
    with pytest.raises(RuntimeError):
        with Proc(["sleep", "300"]) as p:
            raise RuntimeError("boom")
    assert not p.alive()


def test_kill_all_kills_and_reports_tracked_procs():
    from benchkit.procs import Proc, kill_all
    p = Proc(["sleep", "300"])
    killed = kill_all()
    assert any(str(p.pid) in d and "sleep" in d for d in killed)
    assert not p.alive()
    assert kill_all() == []  # nothing left tracked and running


STRATA_PROM = """# HELP llamacpp:prompt_tokens_total prompt tokens read
llamacpp:prompt_tokens_total 1000
llamacpp:prompt_tokens_cached_total 5000
llamacpp:prompt_seconds_total 2.0
llamacpp:tokens_predicted_total 300
llamacpp:tokens_predicted_seconds_total 4.0
llamacpp:prompt_tokens_seconds 500.0
strata:requests_total 7
strata:drafts_offered_total 400
strata:drafts_accepted_total 300
"""


def test_counter_rates_from_strata_style_counters():
    from benchkit.llserver import counter_rates, parse_counters
    before = parse_counters(STRATA_PROM)
    assert before["llamacpp:prompt_tokens_total"] == 1000 and before["strata:requests_total"] == 7
    after = dict(before)
    after.update({"llamacpp:prompt_tokens_total": 21000, "llamacpp:prompt_seconds_total": 22.0,
                  "llamacpp:prompt_tokens_cached_total": 9000, "llamacpp:tokens_predicted_total": 1100,
                  "llamacpp:tokens_predicted_seconds_total": 14.0, "strata:requests_total": 10,
                  "strata:drafts_offered_total": 1400, "strata:drafts_accepted_total": 1000})
    r = counter_rates(before, after)
    assert (r["prompt_tokens"], r["pp"], r["gen_tokens"], r["tg"]) == (20000, 1000.0, 800, 80.0)
    assert (r["cached"], r["requests"], r["draft_accept"]) == (4000, 3, 0.7)


def test_counter_rates_plain_llama_server_and_idle():
    from benchkit.llserver import counter_rates
    c = {"llamacpp:prompt_tokens_total": 10.0, "llamacpp:prompt_seconds_total": 1.0,
         "llamacpp:tokens_predicted_total": 5.0, "llamacpp:tokens_predicted_seconds_total": 1.0}
    r = counter_rates(c, c)  # nothing happened between the snapshots
    assert r["pp"] is None and r["tg"] is None and "cached" not in r and "draft_accept" not in r


def test_fetch_counters_from_fake_server():
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    from benchkit.llserver import fetch_counters
    fake = Path(__file__).resolve().parent / "fakes" / "engines" / "fake" / "bin" / "llama-server"
    port = 18000 + os.getpid() % 1000
    srv = subprocess.Popen([sys.executable, str(fake), "--port", str(port)], stdout=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            c0 = fetch_counters(base)
            if c0 is not None:
                break
            time.sleep(0.1)
        assert c0 == {"llamacpp:prompt_tokens_total": 0, "llamacpp:prompt_seconds_total": 0.0,
                      "llamacpp:tokens_predicted_total": 0, "llamacpp:tokens_predicted_seconds_total": 0.0}
        assert fetch_counters("http://127.0.0.1:9") is None  # nothing listening
    finally:
        srv.kill()



def test_external_key_file(tmp_path, monkeypatch):
    from benchkit.llserver import auth_headers as external_headers, external_key
    monkeypatch.setenv("BENCHKIT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BENCHKIT_EXTERNAL_KEY_FILE", raising=False)
    assert external_key() == "" and external_headers() == {}
    (tmp_path / "external_api_key").write_text("abc123\n")
    assert external_headers() == {"Authorization": "Bearer abc123"}


def test_stream_rates_server_timings_then_arrival_timing():
    from benchkit.llserver import stream_rates
    m = {"ttft": 0.5, "gen_s": 2.0, "chunks": 101, "usage": {"prompt_tokens": 1000, "completion_tokens": 101},
         "timings": {"prompt_n": 1000, "prompt_per_second": 900.0, "predicted_n": 101, "predicted_per_second": 55.0,
                     "draft_n": 80, "draft_n_accepted": 60}}
    r = stream_rates(m)
    assert (r["method"], r["pp"], r["tg"], r["draft_n"], r["draft_accepted"]) == ("server", 900.0, 55.0, 80, 60)
    r = stream_rates({**m, "timings": None})  # vLLM, Ollama, hosted APIs: arrival timing
    assert (r["method"], r["pp"], r["tg"]) == ("stream", 2000.0, 50.0)
    r = stream_rates({**m, "timings": None, "usage": {"prompt_tokens": 1000, "completion_tokens": 101,
                                                     "prompt_tokens_details": {"cached_tokens": 800}}})
    assert (r["pp"], r["cached"]) == (400.0, 800)  # only the 200 new tokens count as read
    assert stream_rates({"ttft": None, "gen_s": None, "chunks": 0, "usage": {}, "timings": None})["tg"] is None


def test_fill_command_placeholders():
    from benchkit.llserver import fill_command
    argv = fill_command('vllm serve {model} --port {port} --max-model-len {ctx} --hf-overrides \'{"a": 1}\'',
                        8099, "/models/my model.gguf", 32768, [0, 1])
    assert argv == ["vllm", "serve", "/models/my model.gguf", "--port", "8099", "--max-model-len", "32768",
                    "--hf-overrides", '{"a": 1}']
    assert fill_command("srv --port {port} --devices {gpus} {model}", 1, "", 1, [2, 3]) == ["srv", "--port", "1", "--devices", "2,3"]


EXPORTERS = Path(__file__).resolve().parent / "fakes" / "exporters"


def test_nvidia_gpu_exporter_and_node_exporter_samples():
    from benchkit import promtel
    nv = promtel.parse_prom((EXPORTERS / "nvidia_gpu_exporter.prom").read_text())
    g = promtel.nvidia_exporter_gpus(nv)
    assert g["1"] == {"mem_mhz": 715.0, "sm_mhz": 1328.0, "power_w": 142.0, "temp_c": 71.0, "util_pct": 97.0,
                      "vram_mb": 16091.0}
    node = (EXPORTERS / "node_exporter.prom").read_text()
    cpu, mem, prev = promtel.node_cpu_mem(promtel.parse_prom(node), None)
    assert "util_pct" not in cpu and cpu["mhz"] == 1500 and cpu["temp_c"] == 55 and mem["avail_gb"] == 20.0
    later = node.replace('mode="user"} 300', 'mode="user"} 330').replace('cpu="0",mode="idle"} 1000', 'cpu="0",mode="idle"} 1030')
    cpu, _, _ = promtel.node_cpu_mem(promtel.parse_prom(later), prev)
    assert cpu["util_pct"] == round(100 * 60 / 90, 1)  # 60 s busy (user on both CPUs) of 90 s


def test_dcgm_and_drm_exporters():
    from benchkit import promtel
    d = promtel.dcgm_gpus(promtel.parse_prom((EXPORTERS / "dcgm_exporter.prom").read_text()))
    assert d == {"0": {"sm_mhz": 2040.0, "mem_mhz": 6250.0, "temp_c": 64.0, "power_w": 70.5, "util_pct": 99.0,
                       "vram_mb": 20563.0}}
    a = promtel.drm_gpus(promtel.parse_prom((EXPORTERS / "node_exporter_amd.prom").read_text()))
    assert a == {"0": {"util_pct": 87.0, "vram_mb": 8192.0}}


def _fake_amd_card(sysfs, n, slot, busy=40, vram_total=16 * 2**30, dpm_only=False):
    import os
    dev = sysfs / "devices" / f"0000:{slot}:00.0"
    hw = dev / "hwmon" / f"hwmon{n + 5}"
    hw.mkdir(parents=True)
    files = {dev / "vendor": "0x1002", dev / "device": "0x73bf", dev / "product_name": "AMD Radeon Test",
             dev / "gpu_busy_percent": busy, dev / "mem_busy_percent": 7, dev / "mem_info_vram_total": vram_total,
             dev / "mem_info_vram_used": 3 * 2**30, dev / "current_link_width": 8, dev / "max_link_width": 16,
             dev / "mem_info_gtt_total": 32 * 2**30, dev / "mem_info_gtt_used": 2**30,
             dev / "pp_dpm_sclk": "0: 500Mhz\n1: 2100Mhz *\n", dev / "pp_dpm_mclk": "0: 96Mhz\n1: 1000Mhz *\n",
             hw / "temp1_input": 65000, hw / "power1_average": 180500000, hw / "power1_cap": 250000000, hw / "pwm1": 128}
    if not dpm_only:
        files.update({hw / "freq1_input": 2050000000, hw / "freq2_input": 1000000000})
    for f, v in files.items():
        f.write_text(f"{v}\n")
    card = sysfs / "class" / "drm" / f"card{n}"
    card.mkdir(parents=True, exist_ok=True)
    os.symlink(dev, card / "device")
    (sysfs / "class" / "drm" / f"card{n}-DP-1").mkdir()  # a connector, not a GPU


def test_amdgpu_sysfs(tmp_path, monkeypatch):
    from benchkit import amdgpu
    sysfs = tmp_path / "sys"
    _fake_amd_card(sysfs, 1, "0b")
    _fake_amd_card(sysfs, 0, "0a", busy=99, dpm_only=True)
    monkeypatch.setenv("BENCHKIT_SYSFS_ROOT", str(sysfs))
    snap = amdgpu.snapshot_gpus()
    assert [(g["index"], g["pci_bus_id"], g["memory_total"], g["power_limit"]) for g in snap] == \
        [(0, "0000:0a:00.0", 16384, 250.0), (1, "0000:0b:00.0", 16384, 250.0)]
    assert snap[0]["pcie_link_width_current"] == 8 and snap[0]["name"] == "AMD Radeon Test"
    assert snap[0]["gtt_total"] == 32768
    s = amdgpu.sample_gpus()
    assert s["0"] == {"temp_c": 65.0, "power_w": 180.5, "fan_pct": 50, "sm_mhz": 2100.0, "mem_mhz": 1000.0,
                      "util_pct": 99, "mem_util_pct": 7, "vram_mb": 3072, "gtt_mb": 1024}
    assert s["1"]["sm_mhz"] == 2050 and s["1"]["util_pct"] == 40


def test_mixed_nvidia_and_amd(env):
    import os
    from pathlib import Path
    from benchkit import hw
    from benchkit.telemetry import Sampler
    _fake_amd_card(Path(os.environ["BENCHKIT_SYSFS_ROOT"]), 0, "0a")
    snap = hw.snapshot()
    assert [(g["index"], g["vendor"], g["vendor_index"]) for g in snap["gpus"]] == \
        [(0, "NVIDIA", 0), (1, "NVIDIA", 1), (2, "AMD", 0)]
    assert snap["gpu_vendor"] == "AMD+NVIDIA"
    assert hw.visible_devices_env(snap["gpus"], [1, 2]) == {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "0"}
    assert hw.visible_devices_env(snap["gpus"], [2]) == {"HIP_VISIBLE_DEVICES": "0"}
    assert set(Sampler().sample()["gpus"]) == {"0", "1", "2"}

