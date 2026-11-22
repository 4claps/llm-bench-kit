"""End to end through the HTTP API with fake GPU tools: login, system,
create (validation + preflight), start, wait, results, export, import,
compare, invalid marking, delete."""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient


def make_client():
    from benchkit.config import Config
    from benchkit.main import create_app
    return TestClient(create_app(Config()))


def wait_done(c, rid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get(f"/api/runs/{rid}").json()["run"]["status"]
        if st not in ("created", "running"):
            return st
        time.sleep(0.3)
    raise AssertionError("run did not finish")


def login(c):
    assert c.post("/api/login", json={"token": "wrong"}).status_code == 401
    assert c.get("/api/system").status_code == 401
    assert c.post("/api/login", json={"token": "test-token"}).status_code == 200


def test_auth_required_and_bearer(env):
    with make_client() as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/").status_code == 200
        assert c.get("/api/runs").status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer test-token"}).status_code == 200


def test_speed_and_server_run(env):
    with make_client() as c:
        login(c)
        sysinfo = c.get("/api/system").json()
        assert len(sysinfo["snapshot"]["gpus"]) == 2
        assert [e["name"] for e in sysinfo["engines"]] == ["fake", "external", "command"]
        models = c.get("/api/models").json()
        mid = models["custom"][0]["id"]

        bad = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [5], "modules": {"speed": {}}})
        assert bad.status_code == 400

        spec = {"model": mid, "engine": "fake", "gpus": [0, 1], "label": "e2e", "split": "layer",
                "modules": {"speed": {"reps": 2, "kv": ["q4_0", "f16"], "depths": [0, 4096]},
                            "server": {"reps": 2, "mtp": ["off", "on"]}}}
        r = c.post("/api/runs", json=spec).json()
        assert r["estimate_s"] > 0
        assert not r["preflight"]["block"]
        assert any("custom model" in w for w in r["preflight"]["warn"])
        assert c.post(f"/api/runs/{r['id']}/start").status_code == 200
        assert wait_done(c, r["id"]) == "done"

        d = c.get(f"/api/runs/{r['id']}").json()
        sp = d["summary"]["speed"]
        assert sp["kv=q4_0"]["pp512@0"]["n"] == 2 and sp["kv=f16"]["tg128@4096"]["median"] > 0
        sv = d["summary"]["server"]
        assert sv["mtp=on"]["code.tg"]["median"] == 50.0 and sv["mtp=off"]["code.tg"]["median"] == 26.0
        assert "code.draft_accept" in sv["mtp=on"] and "code.draft_accept" not in sv["mtp=off"]
        assert d["run"]["snapshot"]["gpus"][0]["name"].startswith("Tesla")
        assert len(c.get(f"/api/runs/{r['id']}/telemetry").json()) >= 1
        logs = c.get(f"/api/runs/{r['id']}/logs?tail=5000").json()
        assert any("llama-bench" in l["line"] and "-sm layer" in l["line"] for l in logs)
        assert any("llama-bench" in l["line"] and "-ngl 99" in l["line"] for l in logs)
        assert any("starting llama-server" in l["line"] and "-ngl 99" in l["line"] for l in logs)

        b = c.get(f"/api/runs/{r['id']}/export.json").json()
        assert b["bundle_schema"] == 1 and "GPU-fake" not in str(b)
        md = c.get(f"/api/runs/{r['id']}/export.md").text
        assert "## speed" in md and "2x Tesla P100-PCIE-16GB" in md

        imp = c.post("/api/import", json=b).json()["id"]
        cmp = c.get(f"/api/compare?ids={r['id']},{imp}").json()
        assert len(cmp) == 2 and cmp[0]["summary"] == cmp[1]["summary"]
        assert c.post("/api/import", json={"nope": 1}).status_code == 400

        assert c.patch(f"/api/runs/{r['id']}", json={"invalid": True}).status_code == 400
        assert c.patch(f"/api/runs/{r['id']}", json={"invalid": True, "invalid_reason": "test",
                                                     "tags": ["a", "b"], "note": "n"}).status_code == 200
        lst = c.get("/api/runs").json()
        assert {x["id"] for x in lst} == {r["id"], imp}
        assert next(x for x in lst if x["id"] == r["id"])["invalid"]
        assert c.delete(f"/api/runs/{imp}").status_code == 200

        lay = c.put("/api/layouts/default", json={"charts": ["gpu.temp_c", "nope"]}).json()
        assert lay["charts"] == ["gpu.temp_c"]


def test_abort(env):
    with make_client() as c:
        login(c)
        mid = c.get("/api/models").json()["custom"][0]["id"]
        r = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [0],
                                      "modules": {"speed": {"reps": 10, "kv": ["f16", "q8_0", "q4_0"]}}}).json()
        c.post(f"/api/runs/{r['id']}/start")
        assert c.post(f"/api/runs/{r['id']}/start").status_code == 409
        assert c.post(f"/api/runs/{r['id']}/abort").status_code == 200
        assert wait_done(c, r["id"]) == "aborted"


def test_ngl_in_launch_argv(env):
    with make_client() as c:
        login(c)
        mid = c.get("/api/models").json()["custom"][0]["id"]
        r = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [0], "ngl": "44",
                                      "modules": {"speed": {"reps": 1, "kv": ["f16"], "depths": [0]},
                                                  "offload": {"reps": 1},
                                                  "server": {"reps": 1, "mtp": ["off"]}}}).json()
        assert r["spec"]["ngl"] == 44
        assert not r["preflight"]["block"], r["preflight"]
        c.post(f"/api/runs/{r['id']}/start")
        assert wait_done(c, r["id"]) == "done"
        logs = c.get(f"/api/runs/{r['id']}/logs?tail=5000").json()
        bench = [l["line"] for l in logs if l["line"].startswith("llama-bench:")]
        srv = [l["line"] for l in logs if "starting llama-server" in l["line"]]
        assert len(bench) == 2 and all("-ngl 44" in l for l in bench)  # speed and offload
        assert srv and all("-ngl 44" in l for l in srv)
        assert not any("-ngl 99" in l["line"] for l in logs)
        md = c.get(f"/api/runs/{r['id']}/export.md").text
        assert "ngl 44" in md


def test_preflight_blocks_busy_vram(env, monkeypatch):
    with make_client() as c:
        login(c)
        mid = c.get("/api/models").json()["custom"][0]["id"]
        spec = {"model": mid, "engine": "fake", "gpus": [0], "modules": {"speed": {"reps": 1}}}
        assert not c.post("/api/runs", json=spec).json()["preflight"]["block"]  # fake uses 120 MiB
        monkeypatch.setenv("FAKE_NVIDIA_USED_MB", "8000")
        r = c.post("/api/runs", json=spec).json()
        assert any("VRAM in use" in b and "GPU 0" in b for b in r["preflight"]["block"])


def test_failing_module_leaves_no_tracked_process(env, monkeypatch):
    from benchkit.modules import MODULES
    from benchkit.procs import Proc, kill_all

    def boom(ctx, p):
        Proc(["sleep", "300"])  # launched and never waited on: the leak the sweep must catch
        monkeypatch.setenv("FAKE_NVIDIA_USED_MB", "8000")  # so the post-run VRAM check fires too
        raise RuntimeError("boom")

    monkeypatch.setattr(MODULES["speed"], "run", boom)
    with make_client() as c:
        login(c)
        mid = c.get("/api/models").json()["custom"][0]["id"]
        r = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [0],
                                      "modules": {"speed": {"reps": 1}}}).json()
        assert not r["preflight"]["block"]
        c.post(f"/api/runs/{r['id']}/start")
        assert wait_done(c, r["id"]) == "failed"
        d = c.get(f"/api/runs/{r['id']}").json()
        assert any(e["kind"] == "stray_process" and "sleep" in e["message"] for e in d["events"])
        assert any(e["kind"] == "vram_left" and "GPU 0" in e["message"] for e in d["events"])
    assert kill_all() == []  # nothing survived the run


@pytest.mark.skipif(not os.environ.get("BENCHKIT_TEST_HERMES_SOURCE"), reason="needs a hermes-agent checkout")
def test_agent_module_flow(env):
    with make_client() as c:
        login(c)
        mid = c.get("/api/models").json()["custom"][0]["id"]
        r = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [0],
                                      "modules": {"agent": {"tasks": ["no_tool_required", "err_case_search"],
                                                            "reps": 2, "mtp": False}}}).json()
        assert not r["preflight"]["block"], r["preflight"]
        c.post(f"/api/runs/{r['id']}/start")
        assert wait_done(c, r["id"]) == "done"
        d = c.get(f"/api/runs/{r['id']}").json()
        tasks = {(t["task"], t["rep"]): t["success"] for t in d["agent_tasks"]}
        assert tasks == {("no_tool_required", 0): True, ("no_tool_required", 1): True,
                         ("err_case_search", 0): False, ("err_case_search", 1): False}
        assert d["summary"]["agent"]["all"]["total.pass_rate"]["median"] == 0.5


def test_external_server_run(env):
    import subprocess
    import sys
    from pathlib import Path
    fake = Path(__file__).resolve().parent / "fakes" / "engines" / "fake" / "bin" / "llama-server"
    port = 19000 + os.getpid() % 1000
    srv = subprocess.Popen([sys.executable, str(fake), "--port", str(port)], stdout=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}"
        with make_client() as c:
            login(c)
            assert any(e["name"] == "external" and e["external"] for e in c.get("/api/system").json()["engines"])
            for _ in range(50):  # wait for the fake to listen
                if c.post("/api/runs", json={"engine": "external", "server_url": url, "gpus": [0],
                                             "modules": {"server": {}}}).status_code == 200:
                    break
                time.sleep(0.1)
            assert c.post("/api/runs", json={"engine": "external", "gpus": [0],
                                             "modules": {"server": {}}}).status_code == 400
            assert c.post("/api/runs", json={"engine": "external", "server_url": "http://127.0.0.1:1", "gpus": [0],
                                             "modules": {"server": {}}}).status_code == 400
            r = c.post("/api/runs", json={"engine": "external", "server_url": url + "/v1/", "gpus": [0, 1],
                                          "modules": {"speed": {}, "server": {}}}).json()
            assert any(b.startswith("speed:") for b in r["preflight"]["block"])
            r = c.post("/api/runs", json={"engine": "external", "server_url": url, "gpus": [0, 1], "label": "ext",
                                          "modules": {"server": {"reps": 2}}}).json()
            assert r["spec"]["model"] == "external: fake" and r["spec"]["comparable"] is False
            assert r["spec"]["server_url"] == url and not r["preflight"]["block"], r["preflight"]
            assert c.post(f"/api/runs/{r['id']}/start").status_code == 200
            assert wait_done(c, r["id"]) == "done"
            sv = c.get(f"/api/runs/{r['id']}").json()["summary"]["server"]
            assert list(sv) == ["external"]
            assert sv["external"]["code.tg"]["n"] == 2 and sv["external"]["prose.pp"]["median"] == 380.0
    finally:
        srv.kill()


@pytest.mark.skipif(not os.environ.get("BENCHKIT_TEST_HERMES_SOURCE"), reason="needs a hermes-agent checkout")
def test_agent_external_server_records_prefill_decode(env):
    import subprocess
    import sys
    from pathlib import Path
    fake = Path(__file__).resolve().parent / "fakes" / "engines" / "fake" / "bin" / "llama-server"
    port = 19500 + os.getpid() % 400
    srv = subprocess.Popen([sys.executable, str(fake), "--port", str(port)], stdout=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}"
        with make_client() as c:
            login(c)
            for _ in range(50):  # wait for the fake to listen
                r = c.post("/api/runs", json={"engine": "external", "server_url": url, "gpus": [0],
                                              "modules": {"agent": {"tasks": ["no_tool_required", "err_case_search"]}}})
                if r.status_code == 200:
                    break
                time.sleep(0.1)
            run = r.json()
            assert not run["preflight"]["block"], run["preflight"]
            c.post(f"/api/runs/{run['id']}/start")
            assert wait_done(c, run["id"]) == "done"
            d = c.get(f"/api/runs/{run['id']}").json()
            got = {(x["metric"], x["variant"]): x for x in d["results"] if x["module"] == "agent"}
            for task in ("no_tool_required", "err_case_search"):
                assert got[(f"{task}.pp", "external")]["value"] == pytest.approx(380.0, rel=0.01)
                assert got[(f"{task}.tg", "external")]["value"] == pytest.approx(26.0, rel=0.01)
            tp = d["throughput"]["agent"]["external"]
            assert tp["prompt"]["n"] == 2 and tp["gen"]["avg"] == pytest.approx(26.0, rel=0.01)
    finally:
        srv.kill()


def _fake_server(*args):
    import subprocess
    import sys
    from pathlib import Path
    fake = Path(__file__).resolve().parent / "fakes" / "engines" / "fake" / "bin" / "llama-server"
    port = 19100 + (os.getpid() + len(args) * 7) % 300
    return subprocess.Popen([sys.executable, str(fake), "--port", str(port), *args], stdout=subprocess.DEVNULL), port


def test_remote_server_without_gpus_and_with_key(env):
    srv, port = _fake_server("--no-timings", "--api-key", "s3cret-key")
    try:
        url = f"http://127.0.0.1:{port}/v1"
        with make_client() as c:
            login(c)
            for _ in range(50):  # wait for the fake; without the key it is refused
                r = c.post("/api/remote/models", json={"url": url, "key": "s3cret-key"})
                if r.status_code == 200:
                    break
                time.sleep(0.1)
            assert r.json() == {"models": ["fake"]}
            bad = c.post("/api/remote/models", json={"url": url, "key": "wrong"})
            assert bad.status_code == 400 and "API key" in bad.json()["detail"]
            assert c.post("/api/runs", json={"engine": "external", "server_url": url, "gpus": [],
                                             "modules": {"server": {"reps": 1}}}).status_code == 400  # no key
            r = c.post("/api/runs", json={"engine": "external", "server_url": url, "server_key": "s3cret-key", "gpus": [],
                                          "modules": {"server": {"reps": 1, "max_tokens": 64}}}).json()
            assert not r["preflight"]["block"], r["preflight"]
            assert any("no local GPUs" in w for w in r["preflight"]["warn"])
            assert c.post(f"/api/runs/{r['id']}/start").status_code == 200
            assert wait_done(c, r["id"]) == "done"
            d = c.get(f"/api/runs/{r['id']}").json()
            res = {x["metric"]: x for x in d["results"] if x["module"] == "server"}
            assert res["prose.pp"]["value"] > 0 and res["code.tg"]["value"] > 0
            assert res["prose.tg"]["extra"]["method"] == "stream"
            everything = json.dumps(d) + c.get(f"/api/runs/{r['id']}/export.json").text
            assert "s3cret-key" not in everything  # the key is never stored or exported
    finally:
        srv.kill()


def test_custom_command_engine_starts_and_stops_server(env):
    import sys
    from benchkit.procs import kill_all
    from pathlib import Path
    fake = Path(__file__).resolve().parent / "fakes" / "engines" / "fake" / "bin" / "llama-server"
    with make_client() as c:
        login(c)
        bad = c.post("/api/runs", json={"engine": "command", "launch_cmd": "srv --no-port", "gpus": [0],
                                        "modules": {"server": {}}})
        assert bad.status_code == 400 and "{port}" in bad.json()["detail"]
        r = c.post("/api/runs", json={"engine": "command", "launch_cmd": f"{sys.executable} {fake} --port {{port}}",
                                      "gpus": [0], "modules": {"server": {"reps": 1, "max_tokens": 64}}}).json()
        assert r["spec"]["model"] == "command: named by the command" and r["spec"]["comparable"] is False
        assert not r["preflight"]["block"], r["preflight"]
        assert c.post(f"/api/runs/{r['id']}/start").status_code == 200
        assert wait_done(c, r["id"]) == "done"
        d = c.get(f"/api/runs/{r['id']}").json()
        res = {(x["variant"], x["metric"]): x for x in d["results"] if x["module"] == "server"}
        assert res[("command", "prose.pp")]["value"] == 380.0 and res[("command", "prose.pp")]["extra"]["method"] == "server"
        assert res[("command", "code.tg")]["value"] == 26.0
    assert kill_all() == []  # the server the command started is gone


def _serve_exporters(*names):
    """A tiny HTTP server answering /metrics with the given exporter fixtures concatenated."""
    import http.server
    import threading
    from pathlib import Path
    text = "".join((Path(__file__).resolve().parent / "fakes" / "exporters" / n).read_text() for n in names).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(text)
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/metrics"


def test_remote_server_with_exporter_telemetry(env):
    srv, port = _fake_server("--no-timings")
    exp, exp_url = _serve_exporters("node_exporter.prom", "nvidia_gpu_exporter.prom")
    try:
        url = f"http://127.0.0.1:{port}/v1"
        with make_client() as c:
            login(c)
            for _ in range(50):
                r = c.post("/api/runs", json={"engine": "external", "server_url": url, "gpus": [],
                                              "telemetry_urls": f"{exp_url} ", "abort_on": [],
                                              "modules": {"server": {"reps": 1, "max_tokens": 64}}})
                if r.status_code == 200:
                    break
                time.sleep(0.1)
            run = r.json()
            assert run["spec"]["telemetry_urls"] == [exp_url] and run["spec"]["remote_gpus"] == [0, 1]
            assert not run["preflight"]["block"], run["preflight"]
            assert c.post(f"/api/runs/{run['id']}/start").status_code == 200
            assert wait_done(c, run["id"]) == "done"
            d = c.get(f"/api/runs/{run['id']}").json()
            assert d["run"]["snapshot"]["remote"]["gpus"][1]["name"] == "Tesla P100-PCIE-16GB"
            assert "serial" not in json.dumps(d["run"]["snapshot"]["remote"])
            from benchkit import export
            assert export.hardware_line(d["run"]["snapshot"]).startswith("remote: 2x Tesla P100-PCIE-16GB")
            tel = c.get(f"/api/runs/{run['id']}/telemetry").json()
            assert tel and tel[-1]["gpus"]["1"]["temp_c"] == 71.0 and tel[-1]["mem"]["avail_gb"] == 20.0
        bad = None
        with make_client() as c:
            login(c)
            bad = c.post("/api/runs", json={"engine": "external", "server_url": url, "gpus": [],
                                            "telemetry_urls": ["ftp://x"], "modules": {"server": {}}})
        assert bad.status_code == 400 and "telemetry URLs" in bad.json()["detail"]
    finally:
        srv.kill()
        exp.shutdown()


def test_bundled_engine_blocked_on_amd_gpu(env):
    import os
    from pathlib import Path
    from test_units import _fake_amd_card
    _fake_amd_card(Path(os.environ["BENCHKIT_SYSFS_ROOT"]), 0, "0a")
    with make_client() as c:
        login(c)
        gpus = c.get("/api/system?refresh=1").json()["snapshot"]["gpus"]
        assert [g["vendor"] for g in gpus] == ["NVIDIA", "NVIDIA", "AMD"]
        mid = c.get("/api/models").json()["custom"][0]["id"]
        r = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [2], "modules": {"server": {"reps": 1}}}).json()
        assert any("CUDA only" in b for b in r["preflight"]["block"])
        bad = c.post("/api/runs", json={"model": mid, "engine": "fake", "gpus": [2], "power_cap": 150,
                                        "modules": {"server": {}}})
        assert bad.status_code == 400


def test_remote_exporters_api(env, monkeypatch):
    from benchkit import promtel
    exp, exp_url = _serve_exporters("node_exporter.prom")
    try:
        monkeypatch.setattr(promtel, "EXPORTER_PORTS", {int(exp_url.split(":")[2].split("/")[0]): "node_exporter"})
        with make_client() as c:
            login(c)
            r = c.post("/api/remote/exporters", json={"url": "http://127.0.0.1:8080/v1"}).json()
            assert r == {"exporters": [{"url": exp_url, "kind": "node_exporter"}]}
            assert c.post("/api/remote/exporters", json={"url": "nope"}).status_code == 400
    finally:
        exp.shutdown()

