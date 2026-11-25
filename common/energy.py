"""
Turns raw logs into the two datasets of the paper.

    python energy.py --results results [--shift-ms 0]

Inputs  (in --results): runs.jsonl, requests.jsonl (from bench.py), power.csv [plug.csv] (from powerlog.py)
Outputs (in --results): requestLevel.csv  one row per request (with attributed energy)
                        summary.csv       one row per configuration run (the agreed schema)

Energy rule (written down before the first experiment, as agreed):
  1. idle-loaded baseline = mean GPU power during the cool-down right before the run
     (model loaded, no requests).
  2. per power-sampling window: incremental energy = measured energy - baseline * dt.
  3. that incremental energy is split over requests in proportion to the tokens each
     one streamed in that window (time-weighted token share). A window with no streamed
     tokens (pure prefill/queueing) is split equally over the requests in flight.
  4. aggregate quantities (total energy, energy per output token) are always reported
     too; they do not depend on rule 3.

--shift-ms: client timestamps + shift = server timestamps. Use it when the client is a
different machine: shift = clockOffset (chronyc tracking) - oneWayDelay (RTT/2).
On a LAN with NTP both terms are ~0.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def readJsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def windowEnergy(power):
    """Per-window energy (J) between consecutive samples. Counter if present, else trapezoid."""
    t = power["tNs"].to_numpy(dtype=np.int64)
    if power["energyMj"].notna().all():
        dE = np.diff(power["energyMj"].to_numpy(dtype=float)) / 1000.0
    else:
        p = power["powerW"].to_numpy(dtype=float)
        dE = 0.5 * (p[1:] + p[:-1]) * np.diff(t) / 1e9
    return t, dE


def attribute(edges, dEinc, requests):
    """
    edges: window boundaries (ns), len W+1. dEinc: incremental J per window, len W.
    requests: list of dicts with tSendNs, tEndNs, chunkNs (already shifted).
    Returns (per-request J array, unattributed J).
    """
    tokens = np.array([np.histogram(r["chunkNs"], bins=edges)[0] for r in requests], dtype=float)
    inFlight = np.array(
        [(edges[1:] > r["tSendNs"]) & (edges[:-1] < r["tEndNs"]) for r in requests], dtype=float
    )
    tokenSum = tokens.sum(axis=0)
    flightSum = inFlight.sum(axis=0)

    share = np.where(tokenSum > 0, tokens / np.maximum(tokenSum, 1), inFlight / np.maximum(flightSum, 1))
    perRequest = share @ dEinc
    unattributed = float(dEinc[(tokenSum == 0) & (flightSum == 0)].sum())
    return perRequest, unattributed


def processRun(run, requests, power, plug, shiftNs):
    t, dE = windowEnergy(power)
    mid = (t[1:] + t[:-1]) // 2

    idle = (mid >= run["tIdleStartNs"]) & (mid < run["tStartNs"])
    if idle.sum() < 5:
        raise RuntimeError(f"run {run['runId']}: not enough idle samples, is powerlog.py running?")
    idleW = dE[idle].sum() / ((t[1:] - t[:-1])[idle].sum() / 1e9)

    active = (mid >= run["tStartNs"]) & (mid <= run["tEndNs"])
    edges = np.append(t[:-1][active], t[1:][active][-1:])
    dEact = dE[active]
    dEinc = dEact - idleW * np.diff(edges) / 1e9
    durationS = (edges[-1] - edges[0]) / 1e9

    ok = [r for r in requests if r["status"] == "ok"]
    shifted = [
        r | {"tSendNs": r["tSendNs"] + shiftNs, "tEndNs": r["tEndNs"] + shiftNs,
             "chunkNs": [c + shiftNs for c in r["chunkNs"]]}
        for r in ok
    ]
    perRequest, unattributed = attribute(edges, dEinc, shifted) if ok else (np.array([]), float(dEinc.sum()))

    rows = []
    for r, energyJ in zip(ok, perRequest):
        ttft = (r["tFirstNs"] - r["tSendNs"]) / 1e6
        completion = (r["tEndNs"] - r["tSendNs"]) / 1e6
        out = max(int(r["outputTokens"] or 0), 1)
        rows.append({
            "runId": run["runId"],
            "requestIndex": r["requestIndex"],
            "inputTokens": r["inputTokens"],
            "outputTokens": out,
            "ttftMs": ttft,
            "tpotMs": (completion - ttft) / max(out - 1, 1),
            "completionLatencyMs": completion,
            "incrementalEnergyJ": float(energyJ),
            # idle share: baseline power over the request's lifetime, split by concurrency
            "idleShareEnergyJ": idleW * completion / 1000.0 / run["concurrentRequests"],
        })
    req = pd.DataFrame(rows)

    totalOut = req["outputTokens"].sum() if len(req) else 0
    summary = {k: run[k] for k in (
        "runId", "serverType", "gpuModel", "modelName", "quantization", "engine", "engineVersion",
        "concurrentRequests", "requestRate", "repetitionId", "randomSeed", "ignoreEos",
        "cacheEnabled", "networkProfile", "networkRttMs",
    )}
    summary |= {
        "timestamp": pd.Timestamp(run["tStartNs"], unit="ns", tz="UTC").isoformat(),
        "inputTokens": run["targetInputTokens"],
        "outputTokens": run["targetOutputTokens"],
        "meanInputTokensMeasured": req["inputTokens"].mean() if len(req) else None,
        "meanOutputTokensMeasured": req["outputTokens"].mean() if len(req) else None,
        "meanTtftMs": req["ttftMs"].mean(),
        "p50TtftMs": req["ttftMs"].quantile(0.50),
        "p95TtftMs": req["ttftMs"].quantile(0.95),
        "p99TtftMs": req["ttftMs"].quantile(0.99),
        "meanTpotMs": req["tpotMs"].mean(),
        "meanCompletionLatencyMs": req["completionLatencyMs"].mean(),
        "p95CompletionLatencyMs": req["completionLatencyMs"].quantile(0.95),
        "p99CompletionLatencyMs": req["completionLatencyMs"].quantile(0.99),
        "outputTokensPerSecond": totalOut / durationS,
        "averageGpuPowerW": dEact.sum() / durationS,
        "idleLoadedPowerW": idleW,
        "totalGpuEnergyJ": dEact.sum(),
        "incrementalGpuEnergyJ": dEinc.sum(),
        "unattributedEnergyJ": unattributed,
        "energyPerRequestJ": req["incrementalEnergyJ"].mean(),
        "energyPerRequestWithIdleJ": (req["incrementalEnergyJ"] + req["idleShareEnergyJ"]).mean(),
        "energyPerOutputTokenJ": dEinc.sum() / max(totalOut, 1),
        "failedRequests": len(requests) - len(ok),
        "gpuClocksMHz": power.loc[(power["tNs"] >= run["tStartNs"]) & (power["tNs"] <= run["tEndNs"]), "smClockMHz"].mean(),
        "gpuTempC": power.loc[(power["tNs"] >= run["tStartNs"]) & (power["tNs"] <= run["tEndNs"]), "tempC"].max(),
        "powerMeasurementSource": "nvml-counter" if power["energyMj"].notna().all() else "nvml-power",
    }

    if plug is not None:
        inRun = plug[(plug["tNs"] >= run["tStartNs"]) & (plug["tNs"] <= run["tEndNs"])]
        if len(inRun) >= 2:
            # The plug's energy counter steps only every ~15-20 s, too coarse for runs of a few
            # seconds, so wall energy integrates the 1 s power readings. The counter is kept as a check.
            dtS = inRun["tNs"].diff().iloc[1:] / 1e9
            wallJ = float((inRun["wallPowerW"].iloc[1:] * dtS).sum())
            summary["wallEnergyJ"] = wallJ
            summary["wallEnergyCounterJ"] = (inRun["wallEnergyWh"].iloc[-1] - inRun["wallEnergyWh"].iloc[0]) * 3600.0
            summary["wallToGpuRatio"] = wallJ / summary["totalGpuEnergyJ"]
    return req, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--shift-ms", type=float, default=0.0)
    args = p.parse_args()

    folder = Path(args.results)
    runs = readJsonl(folder / "runs.jsonl")
    allRequests = readJsonl(folder / "requests.jsonl")
    power = pd.read_csv(folder / "power.csv")
    plug = pd.read_csv(folder / "plug.csv") if (folder / "plug.csv").exists() else None
    node = json.loads((folder / "node.json").read_text()) if (folder / "node.json").exists() else {}

    requestFrames, summaries = [], []
    for run in runs:
        req, summary = processRun(
            run, [r for r in allRequests if r["runId"] == run["runId"]], power, plug, int(args.shift_ms * 1e6)
        )
        summary |= {"driverVersion": node.get("driverVersion"), "samplingIntervalMs": node.get("samplingIntervalMs"),
                    "gpuModel": summary["gpuModel"] or node.get("gpuModel")}
        requestFrames.append(req)
        summaries.append(summary)

    pd.concat(requestFrames).to_csv(folder / "requestLevel.csv", index=False)
    pd.DataFrame(summaries).to_csv(folder / "summary.csv", index=False)
    print(f"{len(summaries)} runs -> {folder / 'summary.csv'}")


if __name__ == "__main__":
    main()
