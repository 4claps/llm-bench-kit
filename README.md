# LLM Bench Kit

A web app that benchmarks LLM inference and produces a result file you can
share. It started on Tesla P100s (they are still the reference hardware for
the bundled builds), but it runs any engine and any model, on this machine or
on another one:

| Target | What it is | Modules | Telemetry, guardrails, cool-down |
|---|---|---|---|
| **Bundled llama.cpp** | the image's pinned llama.cpp builds, started on this machine's NVIDIA GPUs | speed, server, offload, agent | yes |
| **Custom command** | any engine with an OpenAI-compatible server (vLLM, SGLang, ik_llama.cpp, TabbyAPI, Strata, Ollama...), started here from a command line you give | server, agent | yes, for the GPUs you tick |
| **Remote server** | any OpenAI-compatible URL that is already running: another machine on your network or a hosted API, with an optional API key | server, agent | from that machine's Prometheus exporters if you give them; else only for GPUs you tick here |

The bundled builds target P100, P40/P4 and V100 by default; set `CUDA_ARCHS`
to build them for newer cards (for example `75;80;86;89`). AMD GPUs are
detected and monitored too (see [AMD GPUs](#amd-gpus)); run a ROCm or Vulkan
build of your engine on them with the Custom command engine. For a remote
server you don't need a GPU at all: see [Remote only](#remote-only-no-gpu).

You open a web page, pick a model, an engine, which GPUs and which tests,
choose how many runs each test does, and watch it live. Every run is kept in
a history you can compare, mark invalid, and export.

## What it measures

| Module | What | Typical time |
|---|---|---|
| **Speed** | `llama-bench` prompt processing and decode t/s at empty and 16k context, per KV cache type | 5–15 min |
| **Server** | real `llama-server` requests: a ~2k-token prose summary and a code prompt, fixed output length, MTP on and off | 5–10 min |
| **RAM offload** | MoE expert layers in system RAM (`-ot exps=CPU`); decode is then bound by RAM bandwidth, so this separates DDR3 from DDR4 boxes | 5–10 min |
| **Agent** | the Hermes agent tool-use battery (trap tasks from hermes-agent's `toolperf_abeval`, MIT); pass rate and wall time per task, plus prompt processing (prefill) and decode t/s | ~2 min per task per run |

What each test measures, and what every agent task is checking for: [docs/WHAT-IT-TESTS.md](docs/WHAT-IT-TESTS.md).

Every test cools the GPUs to 48 °C before it starts and records GPU
temperature, SM and memory clocks, power, utilization, VRAM, PCIe link, CPU
load, clock, temperature and package power, RAM and fan speeds once a second.
Two guardrails watch every sample: a busy GPU stuck below 1000 MHz (stuck
clocks have invalidated real results before) and a GPU above 80 °C. Each
warns once when the condition has held for 10 s and, by default, aborts the run
(untick it on New Run to only warn).

## Quick start

```bash
docker build -t llm-bench-kit .           # 30–90 min: compiles two llama.cpp builds
docker run -d --name bench --gpus all -p 8090:8090 \
  -v /path/to/your/models:/models \
  -v bench-data:/data \
  -v /sys/firmware/dmi/tables:/host-dmi:ro \
  llm-bench-kit
# optional: add --cap-add SYS_ADMIN to let runs change the GPU power cap (see Security)
docker logs bench                          # shows where the login token is
docker exec bench cat /data/token          # the token itself
```

Open `http://<this-machine>:8090`, paste the token, then:

1. **Models**: download a standard model, or verify one you already have.
   Existing files are found by size and checksum even if you renamed them.
2. **New run**: on the **This machine** tab pick engine, model and GPUs (or switch to the **Remote server**
   tab and give a URL), pick modules, set the number of runs, **Review run**, **Start run**.
3. Watch it on the live page. Close the browser if you like; the run keeps going.
4. On the finished run, **Download result file** to share it (see `results/README.md`).

Nothing else may use the selected GPUs during a run; the preflight check
blocks the run if another process holds them.

If the model is bigger than the VRAM of the GPUs you picked, set **GPU
layers (`-ngl`)** on the New Run page: only that many transformer layers go
to the GPUs and the rest stays in system RAM (for example `-ngl 44` for a
model that does not fit otherwise). Left empty, the kit offloads all layers,
which fails with an out-of-memory error for models larger than the selected
cards.

## Engines

| Engine | Source | Multi-GPU | Notes |
|---|---|---|---|
| `upstream` | ggml-org/llama.cpp, pinned commit | layer split | the fair baseline across hardware |
| `kmic` | Kmic-68/llama.cpp, pinned commit | tensor split | Pascal-tuned fork; uses its QUICKSTART env (P2P, graphs on pre-Volta). Built for P100 (sm_60) only, as the fork's own build guide does |

Commits are pinned in the `Dockerfile` (`UPSTREAM_COMMIT`, `KMIC_COMMIT`).
On a P40 or V100, use the `upstream` engine.
Builds use CUDA 12.9, the last toolkit line that still targets sm_60/sm_70.
The CPU backend is built for every x86 variant and picked at runtime, so old
Xeons without AVX2 still work.

### Remote server

The **Remote server** engine (`external` in result files) benchmarks an
OpenAI-compatible server that is already running: llama-server, vLLM,
SGLang, Ollama (`http://host:11434/v1`), LM Studio (`http://host:1234/v1`),
Strata, or a hosted API. Give its URL on the New Run page, and an API key if
it needs one; **Load models** lists what the server offers. Only the
**server** and **agent** modules run (speed and offload need llama.cpp's own
tools). Things to know:

- The key is kept in memory for that run only. It is never written into the
  run, its results or its export. For unattended use you can instead put a key
  in `<data dir>/external_api_key` (mode 600); it is used when none is typed.
- The kit doesn't start, stop or configure that server, so MTP, KV cache type
  and context size are whatever the server uses. The server module runs one
  pass, labelled `external`.
- Server-module requests are streamed. Prefill and decode come from the
  server's own llama.cpp-style timings when it sends them (llama-server,
  Strata); otherwise from arrival times: prefill = prompt tokens / time to the
  first token, decode = tokens after the first / time from the first to the
  last. Each result's `extra.method` says which (`server` or `stream`).
  Arrival timing includes the network and any queueing, so it reads low for
  short prompts and slow links. Fields only llama.cpp knows are dropped and
  the request retried if a server rejects them.
- Each server prompt gets a unique prefix, so a server that ignores
  `cache_prompt: false` can't reuse an earlier read of the same prompt.
- A server may ignore `ignore_eos` and stop early. Token counts are stored
  next to every rate; check them before you compare.
- Agent prefill and decode come from the server's llama.cpp-style Prometheus
  counters, read before and after each task (Strata: `/metrics/prometheus`;
  llama-server: `/metrics`, which needs `--metrics`). They are stored per task
  as `<task>.pp` and `<task>.tg`, plus `<task>.draft_accept` on Strata. The
  counters include everything the server did meanwhile, so keep other clients
  off it during a run; on Strata the request count is stored with each task.
- For a server on another machine, leave the GPUs unticked. To still get
  that machine's GPU, CPU and RAM charts and guardrails, list its Prometheus
  exporters under **Telemetry exporters** (space-separated `/metrics` URLs).
  The kit fills them in by itself when you enter the server URL: it checks
  the same host on ports 9100, 9835 and 9400 and keeps what answers with
  exporter metrics (only for private-network and local hosts, never a
  hosted API's); an exporter on another port can be typed in. Use
  node_exporter (`:9100`) for CPU and RAM (and AMD GPUs through its drm
  collector), plus nvidia_gpu_exporter (`:9835`) or NVIDIA's DCGM exporter
  (`:9400`) for NVIDIA GPUs. The result file then records that machine's
  GPUs, CPU threads, RAM and kernel (no serials or UUIDs), and its hardware
  line reads `remote: ...`. Without exporters there is no GPU telemetry,
  guardrails or cool-down. For one on this machine, tick the
  GPUs it uses so telemetry and guardrails watch them (cool-down is still
  skipped, since a running server keeps its cards warm). If the server runs
  in another container, the host's LAN IP reaches it.
- Results are marked not comparable with other machines.

### Custom command

The **Custom command** engine starts any engine's OpenAI-compatible server on
this machine, benchmarks it with the same telemetry, guardrails and cleanup
as the bundled builds, and stops it. Give the launch command on the New Run
page, for example:

```
vllm serve {model} --port {port} --max-model-len {ctx}
/opt/ik_llama/bin/llama-server -m {model} --port {port} -c {ctx} -ngl 99 --metrics
```

`{port}` is required (the kit's server port). `{model}` is the model file
picked on the page, or pick *none* when the command names its own model (a
Hugging Face id for vLLM, say). `{ctx}` is the module's context size and
`{gpus}` the ticked GPUs, which are also set as `CUDA_VISIBLE_DEVICES`. The
server counts as ready when `/v1/models` (or `/health`) answers, up to 30
minutes. It runs inside the kit's container, so the engine has to be
installed there, or run the kit [without Docker](#remote-only-no-gpu) next to
your engines. Server and agent modules only; results are marked not
comparable. The last command you used is remembered in your browser.

### AMD GPUs

AMD cards are read from the `amdgpu` driver's own sysfs files, so neither
`rocm-smi` nor `amd-smi` is needed: name, VRAM, power cap and PCIe link for
the run record, and temperature, clocks, power, utilization, VRAM and fan
for the charts and guardrails. In a container, pass the cards through with
`--device /dev/kfd --device /dev/dri`. On a machine with both kinds, the
NVIDIA GPUs come first and the AMD ones are numbered on after them; a Custom
command run gets `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` for the
cards you tick. The bundled llama.cpp builds and the power cap setting are
NVIDIA only. Cards on the older `radeon` driver are not supported. On APUs
and boards that share system RAM with the GPU (the BC-250, for example) the
model sits in GTT, so the kit shows and charts that next to VRAM. A card
that lacks a file simply goes without that reading: the BC-250 has no
`gpu_busy_percent`, so it gets no utilization chart and no low-clock
guardrail (that one needs utilization).

### Remote only (no GPU)

To benchmark remote servers from a laptop or any box without a GPU, skip the
CUDA build:

```bash
docker build -f Dockerfile.lite -t llm-bench-kit:lite .   # ~2 min: no llama.cpp builds
docker run -d --name bench -p 8090:8090 -v bench-data:/data llm-bench-kit:lite
```

or without Docker (Python 3.10+; the agent module also needs a hermes-agent
checkout and its venv, see `Dockerfile.lite` for the pinned commit):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
BENCHKIT_DATA_DIR=./data BENCHKIT_ENGINES_DIR=./no-engines .venv/bin/python -m benchkit.main
```

Only the Remote server and Custom command engines are offered then.

## Making results comparable

Numbers from two machines can only be compared when the model file, engine
commit and settings match. The kit helps:

- Standard models are pinned by file size and SHA256 in `models/manifest.yaml`.
  Any other `.gguf` in your models folder works but is marked **not comparable**.
- The result file records the engine, its commit, every module parameter
  and a hardware summary.
- Server requests run at temperature 0 with a fixed seed and a fixed number
  of generated tokens (`ignore_eos`). MTP acceptance still varies a little
  run to run, which is why each test repeats and reports the median.
- Agent pass rates mostly measure the model, not the hardware; wall time is
  the hardware number.

## Configuration

All optional, as environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BENCHKIT_TOKEN` | random, saved in `/data/token` | login token |
| `BENCHKIT_PORT` | `8090` | web port |
| `BENCHKIT_COOLDOWN_C` | `48` | cool-down target before each test |
| `BENCHKIT_LOW_CLOCK_MHZ` | `1000` | low-clock guardrail |
| `BENCHKIT_THERMAL_WARN_C` | `80` | thermal guardrail |
| `BENCHKIT_GUARDRAIL_SUSTAIN_S` | `10` | how long a condition must hold before it fires |
| `BENCHKIT_SAMPLE_INTERVAL_S` | `1` | telemetry interval |
| `BENCHKIT_VRAM_BUSY_MIB` | `1024` | used VRAM on a selected GPU that blocks a run (and warns after one) |

## Security

- The app always requires the token. Log in once per browser; the session
  cookie is HTTP-only and same-site. API clients can send
  `Authorization: Bearer <token>`.
- Keep it on your own network. For access from outside, put it behind a
  reverse proxy with HTTPS (Caddy: `reverse_proxy 127.0.0.1:8090`) or a VPN.
- Agent tasks run shell commands chosen by the model. They run as an
  unprivileged user inside the container, in a scratch folder under `/data/agent`.
- A Custom command run executes the launch command exactly as typed, with the
  kit's own rights inside the container. Anyone with the token can do that, so
  treat the token like a shell login.
- A remote server's API key is held in memory for its run only and never
  stored. While an agent run is going, the agent's config file holds it (the
  agent needs it to talk to the server), so the model's shell commands could
  read it; it is rewritten without the key when the run ends.
- RAM type and speed come from `dmidecode`, which reads the firmware tables.
  Docker hides those from containers, so the run command mounts the host's
  copy read-only at `/host-dmi`. Leave that line out and the RAM fields show
  as unknown; everything else still works.
- New Run shows each GPU's current power cap and records it with the run.
  Changing it for a run (125-250 W, within what the cards accept) needs
  `--cap-add SYS_ADMIN` on `docker run`, because `nvidia-smi -pl` is refused
  without it. The kit sets the cap on the selected GPUs when the run starts and
  puts the old one back when it ends, even if the run fails or is aborted. If
  the container is killed mid-run, the cap stays until you reset it
  (`nvidia-smi -pl <watts>`) or reboot. Without the flag the option is greyed out.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/pytest -q                        # fake nvidia-smi/llama tools, no GPU needed
BENCHKIT_TEST_HERMES_SOURCE=/path/to/hermes-agent .venv/bin/pytest -q   # also the agent flow
```

`tests/fakes/` has stand-ins for `nvidia-smi`, `llama-bench` and
`llama-server`, so the whole run flow works on a machine without a GPU.

## Credits

- llama.cpp (MIT), the Kmic-68 fork, hermes-agent and its toolperf eval (MIT, Nous Research), uPlot (MIT).
- The guardrails and the cool-down and interleaving rules come from many P100 test runs.
