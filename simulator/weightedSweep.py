"""
Latency-weight sweep for the "weighted" policy of main.py (energy vs latency trade-off).

    python weightedSweep.py [--users 30] [--seeds 7,11,19] [--quick] [--out weightedSweep]

Same compute model as main.py: set MEASUREMENTS=... for measured values, otherwise the
temporary compute numbers are used (fine for seeing the trend, not for the paper).
Outputs: sweep.csv (one row per policy/weight, mean over seeds) and sweep.png.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import main as sim

WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
REFERENCES = ["proposed", "fixedAssociation", "minimumLatency"]


def run(users, seeds, quick):
    baseStations = sim.buildTopology()
    medianModel, riskModel = sim.trainOutputPredictors(913, quick=quick)
    rows = []
    for seed in seeds:
        print(f"seed {seed}")
        positions = sim.generateUserPositions(users, seed)
        requests = sim.generateRequests(positions, seed, medianModel, riskModel)
        links = sim.calculateLinkTable(positions, baseStations, seed)
        runs = [(p, None) for p in REFERENCES] + [("weighted", w) for w in WEIGHTS]
        for policy, weight in runs:
            if weight is not None:
                sim.config.latencyWeight = weight
            result = sim.runPolicy(requests, baseStations, links, policy, seed)
            row = sim.summarizeRun(result, users, seed, policy)
            row["label"] = policy if weight is None else f"w={weight:g}"
            row["latencyWeight"] = weight
            rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = ["meanTotalEnergyJ", "p95CompletionS", "p95TtftS", "meanCompletionS",
               "slaViolationRate", "edgeRatio", "macroRatio", "microRatio", "picoRatio"]
    order = list(dict.fromkeys(frame["label"]))
    summary = frame.groupby("label", sort=False)[metrics].mean().reindex(order)
    return summary


def plot(summary, path, users):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))

    weighted = summary[summary.index.str.startswith("w=")]
    left.plot(weighted["meanCompletionS"], weighted["meanTotalEnergyJ"], "o-", label="Weighted (sweep)")
    for label, row in weighted.iterrows():
        left.annotate(label, (row["meanCompletionS"], row["meanTotalEnergyJ"]),
                      textcoords="offset points", xytext=(6, 4), fontsize=8)
    for label in REFERENCES:
        row = summary.loc[label]
        left.plot(row["meanCompletionS"], row["meanTotalEnergyJ"], "s", label=sim.policyLabels[label])
    left.set_xlabel("Mean completion latency (s)")
    left.set_ylabel("Mean total energy per request (J)")
    left.set_title("Energy vs latency")
    left.grid(alpha=0.25)
    left.legend()

    x = range(len(summary))
    macro, micro, pico = (summary[c].to_numpy() for c in ("macroRatio", "microRatio", "picoRatio"))
    right.bar(x, macro, label="Macro")
    right.bar(x, micro, bottom=macro, label="Micro")
    right.bar(x, pico, bottom=macro + micro, label="Pico")
    right.plot(x, summary["edgeRatio"], "kD--", label="Edge ratio")
    right.set_xticks(list(x))
    right.set_xticklabels(summary.index, rotation=45, ha="right")
    right.set_ylim(0, 1)
    right.set_title(f"Cell-tier association and edge ratio, {users} UEs")
    right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=30)
    p.add_argument("--seeds", default="7,11,19")
    p.add_argument("--quick", action="store_true", help="fast output-length predictor training")
    p.add_argument("--out", default="weightedSweep")
    args = p.parse_args()

    if sim.computeModel.load() is None:
        print("NOTE: temporary compute numbers (no MEASUREMENTS). Trend only, not for the paper.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = run(args.users, [int(s) for s in args.seeds.split(",")], args.quick)
    summary.to_csv(out / "sweep.csv")
    plot(summary, out / "sweep.png", args.users)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
