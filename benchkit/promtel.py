"""Telemetry from another machine, read from its Prometheus exporters.

For a remote server the kit can't run nvidia-smi or read /proc on the box that
does the work, but most GPU boxes already run exporters. Give their /metrics
URLs with the run and each sample is built from them, in the same shape as the
local Sampler's, so charts, guardrails and the export work unchanged:

- node_exporter (:9100): CPU utilization and clock, CPU temperature, RAM, and
  AMD GPUs through its drm collector when there is no GPU exporter.
- nvidia_gpu_exporter (utkuozdemir, :9835): nvidia_smi_* per GPU.
- NVIDIA DCGM exporter (:9400): DCGM_FI_DEV_* per GPU.

Missing metrics are simply left out of the sample.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
CPU_CHIPS = ("coretemp", "k10temp", "zenpower")


def parse_prom(text: str) -> list[tuple[str, dict, float]]:
    """(name, labels, value) for every sample line of a Prometheus text page."""
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(4))
        except ValueError:
            continue
        out.append((m.group(1), dict(_LABEL.findall(m.group(3) or "")), v))
    return out


def fetch(url: str, timeout: float = 5) -> list[tuple[str, dict, float]]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return parse_prom(r.read().decode("utf-8", "replace"))


def _by(metrics, name):
    return [(lab, v) for n, lab, v in metrics if n == name]


def nvidia_exporter_gpus(metrics) -> dict[str, dict]:
    """nvidia_gpu_exporter: GPUs keyed by index (via nvidia_smi_index / gpu_info, which map uuid -> index)."""
    idx = {lab.get("uuid"): str(int(v)) for lab, v in _by(metrics, "nvidia_smi_index")}
    info = {lab.get("uuid"): lab for lab, _ in _by(metrics, "nvidia_smi_gpu_info")}
    for u, lab in info.items():
        idx.setdefault(u, lab.get("index", ""))
    gpus: dict[str, dict] = {}
    conv = {"nvidia_smi_temperature_gpu": ("temp_c", 1), "nvidia_smi_clocks_current_sm_clock_hz": ("sm_mhz", 1e-6),
            "nvidia_smi_clocks_current_memory_clock_hz": ("mem_mhz", 1e-6),
            "nvidia_smi_power_draw_watts": ("power_w", 1), "nvidia_smi_power_draw_instant_watts": ("power_w", 1),
            "nvidia_smi_utilization_gpu_ratio": ("util_pct", 100), "nvidia_smi_utilization_memory_ratio": ("mem_util_pct", 100),
            "nvidia_smi_memory_used_bytes": ("vram_mb", 1 / 2**20), "nvidia_smi_fan_speed_ratio": ("fan_pct", 100),
            "nvidia_smi_pcie_link_gen_current": ("pcie_gen", 1), "nvidia_smi_pcie_link_width_current": ("pcie_width", 1)}
    for name, lab, v in metrics:
        if name in conv and lab.get("uuid") in idx and idx[lab["uuid"]] != "":
            key, f = conv[name]
            gpus.setdefault(idx[lab["uuid"]], {})[key] = round(v * f, 1)
    return gpus


def dcgm_gpus(metrics) -> dict[str, dict]:
    conv = {"DCGM_FI_DEV_GPU_TEMP": "temp_c", "DCGM_FI_DEV_SM_CLOCK": "sm_mhz", "DCGM_FI_DEV_MEM_CLOCK": "mem_mhz",
            "DCGM_FI_DEV_POWER_USAGE": "power_w", "DCGM_FI_DEV_GPU_UTIL": "util_pct",
            "DCGM_FI_DEV_MEM_COPY_UTIL": "mem_util_pct", "DCGM_FI_DEV_FB_USED": "vram_mb"}
    gpus: dict[str, dict] = {}
    for name, lab, v in metrics:
        if name in conv and lab.get("gpu", "") != "":
            gpus.setdefault(lab["gpu"], {})[conv[name]] = round(v, 1)
    return gpus


def drm_gpus(metrics) -> dict[str, dict]:
    """node_exporter's drm collector (amdgpu): busy %, VRAM. Keyed by card number."""
    gpus: dict[str, dict] = {}
    for name, key, f in (("node_drm_gpu_busy_percent", "util_pct", 1), ("node_drm_memory_vram_used_bytes", "vram_mb", 1 / 2**20)):
        for lab, v in _by(metrics, name):
            card = re.sub(r"\D", "", lab.get("card", ""))
            if card:
                gpus.setdefault(card, {})[key] = round(v * f, 1)
    return gpus


def node_cpu_mem(metrics, prev: dict | None) -> tuple[dict, dict, dict]:
    """cpu and mem sample parts from node_exporter; prev carries the CPU-time counters between samples."""
    busy = total = 0.0
    for lab, v in _by(metrics, "node_cpu_seconds_total"):
        total += v
        if lab.get("mode") not in ("idle", "iowait"):
            busy += v
    cpu, mem = {}, {}
    if total and prev and total > prev.get("total", 0):
        cpu["util_pct"] = round(100 * (busy - prev["busy"]) / (total - prev["total"]), 1)
    freqs = [v for _, v in _by(metrics, "node_cpu_scaling_frequency_hertz")]
    if freqs:
        cpu["mhz"] = round(sum(freqs) / len(freqs) / 1e6)
    chips = {lab.get("chip"): lab.get("chip_name") for lab, _ in _by(metrics, "node_hwmon_chip_names")}
    temps = [(lab.get("chip", ""), lab.get("sensor"), v) for lab, v in _by(metrics, "node_hwmon_temp_celsius")]
    for chip, sensor, v in sorted(temps, key=lambda t: (t[0], t[1] or "")):
        if (chips.get(chip) in CPU_CHIPS or any(c in chip for c in CPU_CHIPS)) and sensor == "temp1":
            cpu["temp_c"] = v
            break
    for lab, v in _by(metrics, "node_memory_MemAvailable_bytes"):
        mem["avail_gb"] = round(v / 2**30, 2)
    return cpu, mem, ({"busy": busy, "total": total} if total else (prev or {}))


class RemoteSampler:
    """Samples a remote machine through its exporters; same sample() shape as telemetry.Sampler."""

    def __init__(self, urls: list[str]):
        self.urls = list(urls)
        self._cpu_prev: dict | None = None
        self.errors: dict[str, str] = {}

    def _all(self) -> list:
        metrics = []
        for u in self.urls:
            try:
                metrics += fetch(u)
                self.errors.pop(u, None)
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.errors[u] = str(getattr(e, "reason", e))[:200]
        return metrics

    @staticmethod
    def gpus_from(metrics) -> dict[str, dict]:
        return nvidia_exporter_gpus(metrics) or dcgm_gpus(metrics) or drm_gpus(metrics)

    def sample(self) -> dict:
        m = self._all()
        cpu, mem, self._cpu_prev = node_cpu_mem(m, self._cpu_prev)
        return {"t": time.time(), "gpus": self.gpus_from(m), "cpu": cpu, "mem": mem, "fans": {}}

    def max_gpu_temp(self, gpus: list[int] | None = None) -> float | None:
        temps = [g["temp_c"] for g in self.sample()["gpus"].values() if isinstance(g.get("temp_c"), (int, float))]
        return max(temps) if temps else None

    def snapshot(self) -> dict:
        """The remote box's hardware for the run record: GPU names and VRAM, CPU, RAM, kernel. No serials or UUIDs."""
        m = self._all()
        gpus = []
        for lab, _ in _by(m, "nvidia_smi_gpu_info"):
            total = next((v for l2, v in _by(m, "nvidia_smi_memory_total_bytes") if l2.get("uuid") == lab.get("uuid")), None)
            gpus.append({"index": int(lab.get("index", len(gpus))), "name": lab.get("name"),
                         "memory_total": round(total / 2**20) if total else None,
                         "driver_version": lab.get("driver_version"), "compute_cap": lab.get("compute_cap")})
        if not gpus:
            seen = {}
            for lab, _ in _by(m, "DCGM_FI_DEV_GPU_TEMP") + _by(m, "DCGM_FI_DEV_FB_USED"):
                if lab.get("gpu", "") != "":
                    seen.setdefault(lab["gpu"], {"index": int(lab["gpu"]), "name": lab.get("modelName")})
            for lab, v in _by(m, "DCGM_FI_DEV_FB_FREE"):
                used = next((u for l2, u in _by(m, "DCGM_FI_DEV_FB_USED") if l2.get("gpu") == lab.get("gpu")), 0)
                if lab.get("gpu") in seen:
                    seen[lab["gpu"]]["memory_total"] = round(v + used)
            gpus = list(seen.values())
        if not gpus:
            for lab, v in _by(m, "node_drm_memory_vram_size_bytes"):
                card = re.sub(r"\D", "", lab.get("card", ""))
                if card:
                    gpus.append({"index": int(card), "name": "AMD GPU (" + lab.get("card", "") + ")",
                                 "memory_total": round(v / 2**20)})
        gpus.sort(key=lambda g: g["index"])
        cpus = {lab.get("cpu") for lab, _ in _by(m, "node_cpu_seconds_total")}
        model = next((lab.get("model_name") for lab, _ in _by(m, "node_cpu_info") if lab.get("model_name")), None)
        mem_total = next((v for _, v in _by(m, "node_memory_MemTotal_bytes")), None)
        uname = next((lab for lab, _ in _by(m, "node_uname_info")), {})
        return {"gpus": gpus, "cpu": {"model": model, "threads": len(cpus) or None},
                "memory": {"total_gb": round(mem_total / 2**30, 1) if mem_total else None},
                "kernel": uname.get("release"), "exporters": len(self.urls), "errors": dict(self.errors)}
