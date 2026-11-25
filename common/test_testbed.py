"""python test_testbed.py  -> energy attribution conserves energy, interpolation hits grid points."""

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from computeModel import MeasuredCompute
from energy import processRun

S = 1_000_000_000  # ns


def syntheticRun():
    # 10 s idle at 50 W, then 10 s at 150 W; 20 ms samples; energy counter in mJ.
    t = np.arange(0, 20 * S, 20_000_000)
    power = np.where(t < 10 * S, 50.0, 150.0)
    energy = np.concatenate([[0.0], np.cumsum(power[:-1] * 0.02 * 1000)])
    frame = pd.DataFrame({"tNs": t, "energyMj": energy, "powerW": power, "smClockMHz": 1800, "tempC": 60})
    run = {
        "runId": "r", "serverType": "edge", "gpuModel": "x", "modelName": "m", "quantization": "INT4",
        "engine": "vllm", "engineVersion": "0", "concurrentRequests": 2, "requestRate": 0, "repetitionId": 0,
        "randomSeed": 1, "ignoreEos": True, "cacheEnabled": False, "networkProfile": "none", "networkRttMs": 0,
        "targetInputTokens": 512, "targetOutputTokens": 128,
        "tIdleStartNs": 0, "tStartNs": 10 * S, "tEndNs": 20 * S - 20_000_000,
    }
    # request a streams 3x as many tokens as b over the same span -> gets ~3x the energy
    req = [
        {"requestIndex": 0, "status": "ok", "inputTokens": 512, "outputTokens": 300, "tSendNs": 10 * S, "tEndNs": 19 * S,
         "tFirstNs": 11 * S, "chunkNs": list(np.linspace(11 * S, 19 * S, 300).astype(int))},
        {"requestIndex": 1, "status": "ok", "inputTokens": 512, "outputTokens": 100, "tSendNs": 10 * S, "tEndNs": 19 * S,
         "tFirstNs": 11 * S, "chunkNs": list(np.linspace(11 * S, 19 * S, 100).astype(int))},
    ]
    return run, req, frame


def testAttribution():
    run, req, power = syntheticRun()
    requestLevel, summary = processRun(run, req, power, None, 0)
    assert abs(summary["idleLoadedPowerW"] - 50.0) < 1e-6
    assert abs(summary["incrementalGpuEnergyJ"] - 100.0 * 9.98) < 0.5  # 100 W extra for ~10 s
    attributed = requestLevel["incrementalEnergyJ"].sum() + summary["unattributedEnergyJ"]
    assert abs(attributed - summary["incrementalGpuEnergyJ"]) < 1e-6, "energy not conserved"
    a, b = requestLevel["incrementalEnergyJ"]
    assert 2.5 < a / b < 3.5, (a, b)


def testWallEnergy():
    # plug at 300 W every 1 s, counter that has not stepped yet: wall energy must come from power
    run, req, power = syntheticRun()
    t = np.arange(0, 21 * S, S)
    plug = pd.DataFrame({"tNs": t, "wallPowerW": 300.0, "wallEnergyWh": 1.0})
    _, summary = processRun(run, req, power, plug, 0)
    assert abs(summary["wallEnergyJ"] - 300.0 * 9) < 1e-6, summary["wallEnergyJ"]  # samples 10..19 s
    assert summary["wallEnergyCounterJ"] == 0


def testInterpolation():
    rows = []
    for server, quant, i, o, c, rep in itertools.product(
        ["edge", "cloud"], ["INT4"], [128, 1024], [64, 512], [1, 20], range(3)
    ):
        rows.append({
            "serverType": server, "quantization": quant, "modelName": "m", "requestRate": 0,
            "inputTokens": i, "outputTokens": o, "concurrentRequests": c,
            "meanTtftMs": 100 + i * 0.1 + c, "meanTpotMs": 10.0 + c * 0.5,
            "energyPerRequestWithIdleJ": o * 0.5 + c, "averageGpuPowerW": 200.0, "networkRttMs": 0.0,
        })
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "summary.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        model = MeasuredCompute([path])
    edge = model.estimate("edge", 1024, 512, 20, "int4")
    assert abs(edge["tpotSeconds"] - 0.020) < 1e-9
    assert abs(edge["energyJoules"] - (256 + 20)) < 1e-6  # edge factor 1.0 without plug data
    mid = model.estimate("edge", 1024, 288, 20, "INT4")
    assert abs(mid["energyJoules"] - (144 + 20)) < 1e-6  # linear between output grid points
    assert model.capacity["edge"] == 20


if __name__ == "__main__":
    testAttribution()
    testWallEnergy()
    testInterpolation()
    print("ok")
