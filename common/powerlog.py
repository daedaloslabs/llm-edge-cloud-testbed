"""
Power logger. Runs ON the serving node (edge or cloud) for the whole sweep.

GPU level (NVML, every --interval-ms):
  The primary quantity is NVML's cumulative energy counter (mJ since driver load).
  Differencing a counter cannot miss prefill spikes, unlike sampling power and
  integrating. Instantaneous power is logged too, as a fallback and for plots.

Wall level (optional, --plug IP): Shelly Gen2/Gen3 plug via its local RPC API, ~1 s.

    pip install nvidia-ml-py
    python powerlog.py --out results              # Ctrl+C to stop
    python powerlog.py --out results --plug 192.168.1.50

Outputs: OUT/power.csv, OUT/plug.csv (if --plug), OUT/node.json (hardware/driver info).
"""

import argparse
import csv
import json
import platform
import signal
import threading
import time
import urllib.request
from pathlib import Path

import pynvml as nv


def nodeInfo(handle):
    def safe(fn, *a):
        try:
            value = fn(*a)
            return value.decode() if isinstance(value, bytes) else value
        except nv.NVMLError:
            return None

    return {
        "gpuModel": safe(nv.nvmlDeviceGetName, handle),
        "driverVersion": safe(nv.nvmlSystemGetDriverVersion),
        "cudaDriverVersion": safe(nv.nvmlSystemGetCudaDriverVersion_v2),
        "powerLimitW": (safe(nv.nvmlDeviceGetEnforcedPowerLimit, handle) or 0) / 1000,
        "memoryTotalMiB": nv.nvmlDeviceGetMemoryInfo(handle).total // 2**20,
        "host": platform.node(),
        "os": platform.platform(),
    }


def plugLoop(ip, path, interval, stop):
    # Shelly Gen2+/Gen3: apower = W, aenergy.total = Wh (cumulative).
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(["tNs", "wallPowerW", "wallEnergyWh"])
        while not stop.is_set():
            try:
                with urllib.request.urlopen(f"http://{ip}/rpc/Switch.GetStatus?id=0", timeout=2) as r:
                    status = json.load(r)
                writer.writerow([time.time_ns(), status["apower"], status["aenergy"]["total"]])
                handle.flush()
            except Exception as exc:
                print("plug read failed:", exc, flush=True)
            stop.wait(interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--interval-ms", type=float, default=20.0)
    p.add_argument("--plug", help="Shelly plug IP for wall-level power")
    p.add_argument("--plug-interval", type=float, default=1.0)
    args = p.parse_args()

    def interrupt(*_):
        raise KeyboardInterrupt
    # background jobs ignore SIGINT by default; kill (SIGTERM) must stop it cleanly too
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)

    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(args.gpu)
    (outDir / "node.json").write_text(json.dumps(nodeInfo(h) | {"samplingIntervalMs": args.interval_ms}, indent=2))

    try:
        nv.nvmlDeviceGetTotalEnergyConsumption(h)
        hasCounter = True
    except nv.NVMLError:
        hasCounter = False
        print("WARNING: no NVML energy counter on this GPU, energy.py will integrate power instead")

    stop = threading.Event()
    if args.plug:
        threading.Thread(target=plugLoop, args=(args.plug, outDir / "plug.csv", args.plug_interval, stop), daemon=True).start()

    step = args.interval_ms / 1000.0
    with open(outDir / "power.csv", "a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(["tNs", "energyMj", "powerW", "smClockMHz", "memClockMHz", "tempC", "gpuUtil", "memUsedMiB"])
        nextTick = time.perf_counter()
        rows = 0
        try:
            while True:
                util = nv.nvmlDeviceGetUtilizationRates(h)
                writer.writerow([
                    time.time_ns(),
                    nv.nvmlDeviceGetTotalEnergyConsumption(h) if hasCounter else "",
                    nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                    nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                    util.gpu,
                    nv.nvmlDeviceGetMemoryInfo(h).used // 2**20,
                ])
                rows += 1
                if rows % 250 == 0:
                    handle.flush()
                nextTick += step  # drift-free schedule
                time.sleep(max(0.0, nextTick - time.perf_counter()))
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            nv.nvmlShutdown()


if __name__ == "__main__":
    main()
