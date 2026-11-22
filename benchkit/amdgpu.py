"""AMD GPUs through the amdgpu driver's sysfs files: no rocm-smi or amd-smi needed.

Every file read here is documented in the kernel's amdgpu docs
(Documentation/gpu/amdgpu/thermal.rst and driver-misc.rst) and is readable by
any user, also inside a container (pass --device /dev/kfd --device /dev/dri so
engines can use the cards; /sys is visible by default). Cards are numbered in
/sys/class/drm order (card0, card1, ...), counting AMD cards only, which is the
order ROCm's HIP_VISIBLE_DEVICES uses on machines with only AMD GPUs. Some
chips don't support every file (the BC-250's has no gpu_busy_percent, so no
utilization); what a card lacks is left out of its samples.

BENCHKIT_SYSFS_ROOT points the reader at a copy of /sys (tests use a fake tree).
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

AMD_VENDOR = "0x1002"


def _root() -> str:
    return os.environ.get("BENCHKIT_SYSFS_ROOT", "/sys")


def _read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return default


def _int(path):
    v = _read(path)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def cards() -> list[Path]:
    """The device dirs of AMD GPUs, in card order (connectors like card0-DP-1 are skipped)."""
    out = []
    for c in glob.glob(os.path.join(_root(), "class", "drm", "card*")):
        name = os.path.basename(c)
        if re.fullmatch(r"card\d+", name) and _read(os.path.join(c, "device", "vendor")) == AMD_VENDOR:
            out.append((int(name[4:]), Path(c, "device")))
    return [d for _, d in sorted(out)]


def _hwmon(dev: Path) -> Path | None:
    hw = sorted(glob.glob(str(dev / "hwmon" / "hwmon*")))
    return Path(hw[0]) if hw else None


def _dpm_mhz(path) -> float | None:
    """The active level of a pp_dpm_* table ("1: 1000Mhz *")."""
    for line in (_read(path) or "").splitlines():
        if line.rstrip().endswith("*"):
            m = re.search(r"(\d+)\s*Mhz", line, re.I)
            if m:
                return float(m.group(1))
    return None


def gpu_name(dev: Path) -> str:
    """product_name (newer dGPUs), else lspci's name, else the PCI ID database, else the device id."""
    name = _read(dev / "product_name")
    if name:
        return name
    bus = os.path.basename(os.path.realpath(dev))
    if shutil.which("lspci"):
        try:
            out = subprocess.run(["lspci", "-mm", "-s", bus], capture_output=True, text=True, timeout=5).stdout
            fields = re.findall(r'"([^"]*)"', out)
            if len(fields) >= 3 and fields[2]:
                return fields[2] if fields[2].startswith(("AMD", "Radeon")) else f"AMD {fields[2]}"
        except (OSError, subprocess.SubprocessError):
            pass
    dev_id = (_read(dev / "device") or "").lower().removeprefix("0x")
    for ids in ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids"):
        try:
            in_amd = False
            for line in Path(ids).read_text(errors="replace").splitlines():
                if line.startswith("1002 "):
                    in_amd = True
                elif in_amd and line and not line.startswith("\t"):
                    break
                elif in_amd and line.startswith(f"\t{dev_id} "):
                    return "AMD " + line.split(None, 1)[1].strip()
        except OSError:
            continue
    return f"AMD GPU 0x{dev_id or '?'}"


def snapshot_gpus() -> list[dict]:
    """Same keys as the NVIDIA snapshot where AMD has an equivalent."""
    gpus = []
    for i, dev in enumerate(cards()):
        hw = _hwmon(dev)
        cap = lambda f: (round(_int(hw / f) / 1e6, 1) if hw and _int(hw / f) else None)
        vram = _int(dev / "mem_info_vram_total")
        g = {"index": i, "vendor": "AMD",
             "name": gpu_name(dev),
             "uuid": _read(dev / "unique_id"),
             "pci_bus_id": os.path.basename(os.path.realpath(dev)),
             "driver_version": _read(Path(_root(), "module", "amdgpu", "version")) or f"kernel {platform.release()}",
             "vbios_version": _read(dev / "vbios_version"),
             "memory_total": round(vram / 2**20) if vram else None,
             # GTT: system RAM the GPU maps; on APUs and boards like the BC-250 the model lives here, not in VRAM
             "gtt_total": round(_int(dev / "mem_info_gtt_total") / 2**20) if _int(dev / "mem_info_gtt_total") else None,
             "power_limit": cap("power1_cap"), "power_min_limit": cap("power1_cap_min"),
             "power_max_limit": cap("power1_cap_max"), "power_default_limit": cap("power1_cap_default"),
             "pcie_link_width_current": _int(dev / "current_link_width"),
             "pcie_link_width_max": _int(dev / "max_link_width")}
        gpus.append(g)
    return gpus


def sample_gpus() -> dict[str, dict]:
    """{index: {temp_c, sm_mhz, mem_mhz, power_w, util_pct, mem_util_pct, vram_mb, fan_pct}} like the NVIDIA sampler."""
    out = {}
    for i, dev in enumerate(cards()):
        hw = _hwmon(dev)
        g: dict = {}
        if hw:
            t = _int(hw / "temp1_input")  # edge temperature, millidegrees
            if t is not None:
                g["temp_c"] = round(t / 1000, 1)
            for f, key in (("freq1_input", "sm_mhz"), ("freq2_input", "mem_mhz")):
                v = _int(hw / f)
                if v:
                    g[key] = round(v / 1e6)
            p = _int(hw / "power1_average") or _int(hw / "power1_input")  # microwatts
            if p is not None:
                g["power_w"] = round(p / 1e6, 1)
            pwm = _int(hw / "pwm1")
            if pwm is not None:
                g["fan_pct"] = round(100 * pwm / 255)
        g.setdefault("sm_mhz", _dpm_mhz(dev / "pp_dpm_sclk"))
        g.setdefault("mem_mhz", _dpm_mhz(dev / "pp_dpm_mclk"))
        for f, key in (("gpu_busy_percent", "util_pct"), ("mem_busy_percent", "mem_util_pct")):
            v = _int(dev / f)
            if v is not None:
                g[key] = v
        used = _int(dev / "mem_info_vram_used")
        if used is not None:
            g["vram_mb"] = round(used / 2**20)
        gtt = _int(dev / "mem_info_gtt_used")
        if gtt is not None:
            g["gtt_mb"] = round(gtt / 2**20)
        out[str(i)] = {k: v for k, v in g.items() if v is not None}
    return out
