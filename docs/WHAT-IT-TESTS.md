# What it tests

LLM Bench Kit has four tests (modules). You pick which ones a run does on the New Run page. This page explains what
each one measures, and what every agent task is checking for.

| Test | What it does | What it tells you | Needs the engine on this machine |
|---|---|---|---|
| [Speed](#speed) | `llama-bench` reading and writing speed at empty and filled context | the raw speed of the hardware and engine | yes |
| [Server](#server) | real chat requests through the server's API | the speed a chat client or agent actually gets | no |
| [RAM offload](#ram-offload) | an MoE model with its experts in system RAM | system RAM bandwidth (DDR3 vs DDR4) | yes |
| [Agent](#agent) | the Hermes agent tool-use tasks | whether the model uses tools correctly, and how long it takes | no |

On a **remote server** (or a custom-command engine) only Server and Agent run; Speed and RAM offload need the bundled
llama.cpp on the machine running the kit.

## Recorded for every test

- **Cool-down:** the GPUs cool to 48 °C before each test starts, so every test starts from the same state.
- **Telemetry, once a second:** GPU temperature, SM and memory clocks, power, utilization, VRAM, PCIe link; CPU load,
  clock, temperature and package power; RAM; fan speeds. For a remote server, the same comes from its Prometheus
  exporters (node_exporter, nvidia_gpu_exporter, DCGM) when you give their URLs.
- **Guardrails:** a busy GPU stuck below 1000 MHz (stuck clocks have spoiled real results before) and a GPU above
  80 °C. Each warns once after 10 s and, by default, stops the run (untick it on New Run to only warn).
- **Hardware snapshot:** GPUs, driver, CPU, RAM type and speed, so results from different machines can be compared.

## Speed

`llama-bench` with a fixed prompt length (512 tokens by default) and a fixed number of generated tokens (128), at
each context depth you choose (empty and 16k by default) and each KV cache type you tick (`q4_0` by default).

- **pp t/s** (prompt processing, "prefill", reading): how fast the model reads a prompt.
- **tg t/s** (token generation, "decode", writing): how fast it writes the answer.
- The filled-context numbers show how much a long conversation slows it down.

## Server

Real requests to `llama-server` (or your remote server) the way a chat app or agent sends them: a ~2k-token prose
summary and a code prompt, at temperature 0 with a fixed seed and a fixed output length, so every machine generates
the same number of tokens. With the bundled engine it runs with MTP (draft-model speculative decoding) off and on, so
you see how much the drafts help.

- **Prefill and decode t/s** for each prompt. They come from the server's own timings when it sends them, otherwise
  from when the streamed tokens arrive (prefill = prompt tokens / time to the first token; decode = the rest of the
  tokens / the time they took). `extra.method` in the results says which.
- **Draft acceptance** where the server reports it.
- Each prompt gets a unique prefix, so a server with its own prompt cache can't reuse an earlier read.

## RAM offload

`llama-bench` with an MoE model's expert layers kept in system RAM (`-ot exps=CPU`) and attention on the GPU. Writing
speed is then bound by RAM bandwidth, which makes this the test that separates DDR3 boxes from DDR4 (the snapshot
records RAM type, speed and populated slots beside the numbers). Needs an MoE model.

## Agent

The real Hermes agent (`hermes chat`) works through small tool-use tasks in a throwaway project folder. Each task is
a **trap for one specific tool-use mistake** agents make, and a fixed check grades it. The tasks and checks come
unchanged from hermes-agent's `scripts/toolperf_abeval/ab_eval.py` (Nous Research, MIT), so the grading matches the
published harness.

- **Pass rate** mostly measures the **model** (and how well the engine handles tool calls).
- **Wall time** per task mostly measures the **hardware and engine**.
- **Prefill and decode** per task, when the server reports them: from the server's timing lines when the kit starts
  `llama-server` itself, or from its llama.cpp-style Prometheus counters before and after each task on a remote
  server (`llama-server --metrics`, Strata `/metrics/prometheus`). Counters include anything else using that server
  during the run.

| Task | What the agent is asked | The trap | Passes when |
|---|---|---|---|
| `err_python_env` | Follow the README: run `python consume.py` and report what it prints. | The package it imports is only installed inside the project's venv, so plain `python` fails. | It reports the value printed by the venv's package. |
| `err_replay_patch` | Make sure `RETRY_LIMIT = 30` in `config.py`, using the patch tool for any change, then run the check. | The line is **already correct**. A careless agent patches anyway and duplicates or breaks it. | The check passes and the line appears exactly once. |
| `err_ambiguous_edit` | Change `timeout = 10` to `60`, but only inside `slow_handler`. | The file has three identical `timeout = 10` lines; a naive patch hits the wrong one. | Only `slow_handler` changed. |
| `err_case_search` | Find the files that contain the setting `primary_endpoint`. | The files spell it `PRIMARY_ENDPOINT` and `PrimaryEndpoint`; a case-sensitive search finds nothing. | It names both `settings.ini` and `client.go`. |
| `err_hidden_search` | Find every file that mentions `SECRET_ROTATION_KEY`. | One of them is in a hidden `.secrets/` folder that default searches skip. | It finds `.secrets/rotation.cfg` and `docs/ops.md`. |
| `err_big_output` | Run a build script and report the `UNIQUE_TOKEN=` line. | The script prints 4,000 lines and the token is in the middle, where truncated output cuts it off. | It reports the token. |
| `err_multi_dir` | Collect the versions from three package folders into `versions.txt`. | The agent must keep track of its working directory across several steps. | `versions.txt` holds `1.4.2,0.9.7,3.2.1`. |
| `err_inline_script` | Compute the sum of the squares of 1-4000 with one `python3 -c` one-liner. | The tooling may refuse a long inline command; the agent has to recover the way the tool suggests. | It answers 21,341,334,000. |
| `err_big_file_read` | Find the one `ANOMALY:` line in a log, using `read_file` (not the terminal). | The log has 6,000 lines with the anomaly at line 4,200, so the agent must page through the file instead of grepping. | It reports the line (code `X99Q`). |
| `no_tool_required` | Work out 17 × 19 without any tool; reply exactly `RESULT=323`. | A sanity check: do tools get used when none are needed, and is the format followed? | The whole reply is exactly `RESULT=323`. |

`err_big_file_read` and `no_tool_required` are **off by default**: the big file read hit the 600 s cap on about a
third of past runs, and `no_tool_required` is only a sanity check. Tick them on New Run to include them. A thinking
model can fail `no_tool_required` while getting the sum right, because Hermes prints its reasoning into the reply the
check reads.

The per-task cap is 600 s by default (a task that hits it counts as a fail, `MODEL_TIMEOUT_600S`). Runs per task
default to 1; use 3 for numbers you want to compare.
