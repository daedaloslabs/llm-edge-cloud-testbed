"""
Measured compute model for the simulator (main.py hooks michaelComputeEstimate /
michaelServerCapacity).

Reads summary.csv files (energy.py output) of edge and cloud, averages repetitions,
and interpolates (multilinear, no extrapolation) over the measured grid
inputTokens x outputTokens x concurrentRequests, per (serverType, quantization).

What it returns, and why:
  prefillSeconds = TTFT - one TPOT - network RTT
      main.py adds its own radio/transport delays and its own queue beyond the slot
      cap, and adds one TPOT to the TTFT itself. Engine-internal batching interference
      at the given concurrency stays in, because that is measured compute behaviour.
  tpotSeconds    = measured mean time per output token
  energyJoules   = attributed incremental energy + idle share (energyPerRequestWithIdleJ),
                   times a facility factor: cloud (1 + alphaNode) * PUE, edge wall/GPU ratio
  powerWatts     = average GPU power of the run (informational)

Knobs (calibration, must be cited / sensitivity-tested in the paper): ALPHA_NODE, PUE,
EDGE_WALL_RATIO (None = take the measured wallToGpuRatio, else 1.0).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ALPHA_NODE = float(os.environ.get("ALPHA_NODE", 0.4))  # literature 0.3-0.5, check against edge measurement
PUE = float(os.environ.get("PUE", 1.2))  # replace with the cloud provider's published value
EDGE_WALL_RATIO = os.environ.get("EDGE_WALL_RATIO")  # None -> measured
MIN_REPETITIONS = 3
AXES = ["inputTokens", "outputTokens", "concurrentRequests"]
METRICS = ["meanTtftMs", "meanTpotMs", "energyPerRequestWithIdleJ", "averageGpuPowerW", "networkRttMs"]


class MeasuredCompute:
    def __init__(self, summaryFiles, modelName=None):
        frame = pd.concat([pd.read_csv(f) for f in summaryFiles], ignore_index=True)
        frame = frame[frame["requestRate"] == 0]  # closed-loop sweeps define the grid
        if modelName:
            frame = frame[frame["modelName"] == modelName]
        if frame.empty:
            raise ValueError("no measurements left after filtering")

        self.models, self.capacity, self.wallRatio = {}, {}, {}
        for (server, quant), group in frame.groupby(["serverType", "quantization"]):
            reps = group.groupby(AXES).size()
            if (reps < MIN_REPETITIONS).any():
                raise ValueError(f"{server}/{quant}: configs with < {MIN_REPETITIONS} repetitions:\n{reps[reps < MIN_REPETITIONS]}")
            mean = group.groupby(AXES)[METRICS].mean()
            axes = [np.sort(group[a].unique()).astype(float) for a in AXES]
            full = pd.MultiIndex.from_product(axes, names=AXES)
            if len(mean) != len(full):
                raise ValueError(f"{server}/{quant}: grid is not complete ({len(mean)} of {len(full)} points)")
            mean = mean.reindex(full)
            shape = [len(a) for a in axes]
            self.models[(server, str(quant).upper())] = {
                m: RegularGridInterpolator(axes, mean[m].to_numpy().reshape(shape)) for m in METRICS
            } | {"axes": axes}
            self.capacity[server] = int(max(self.capacity.get(server, 0), axes[2].max()))
            if "wallToGpuRatio" in group and group["wallToGpuRatio"].notna().any():
                self.wallRatio[server] = float(group["wallToGpuRatio"].mean())

    def facilityFactor(self, server):
        if server == "cloud":
            return (1.0 + ALPHA_NODE) * PUE
        if EDGE_WALL_RATIO is not None:
            return float(EDGE_WALL_RATIO)
        return self.wallRatio.get("edge", 1.0)

    def estimate(self, serverType, inputTokens, outputTokens, concurrency, quantization):
        key = (serverType, str(quantization).upper())
        if key not in self.models:
            raise KeyError(f"no measurements for {key}; measured: {sorted(self.models)}")
        model = self.models[key]
        # Clamped to the measured grid: widen the sweep instead of extrapolating.
        point = [np.clip(v, a[0], a[-1]) for v, a in zip((inputTokens, outputTokens, concurrency), model["axes"])]
        value = {m: float(model[m](point)[0]) for m in METRICS}

        tpot = value["meanTpotMs"] / 1000.0
        prefill = max(value["meanTtftMs"] / 1000.0 - tpot - value["networkRttMs"] / 1000.0, 1e-3)
        return {
            "prefillSeconds": prefill,
            "tpotSeconds": tpot,
            "powerWatts": value["averageGpuPowerW"],
            "energyJoules": value["energyPerRequestWithIdleJ"] * self.facilityFactor(serverType),
        }


_instance = None


def load():
    """Singleton from env MEASUREMENTS (comma-separated summary.csv paths) and MODEL_NAME. None if unset."""
    global _instance
    if _instance is None and os.environ.get("MEASUREMENTS"):
        files = [Path(f) for f in os.environ["MEASUREMENTS"].split(",")]
        _instance = MeasuredCompute(files, os.environ.get("MODEL_NAME"))
    return _instance
