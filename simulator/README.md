# Simulator

`main.py` is the 5G HetNet radio simulator and joint radio-compute optimizer. For each
request it chooses the cell and the serving node (edge or cloud) under different policies
and reports energy and latency. The compute side (time and energy of a request on a node) is
read from the testbed measurements through `common/computeModel.py`, which interpolates the
`summary.csv` grid in input tokens, output tokens and concurrency.

Run from the repository root on the edge node (it has the analysis packages in `.venv`):

```bash
.venv/bin/python simulator/main.py --quick                        # installation check, fallback numbers
MEASUREMENTS=results/<edge>/summary.csv,results/<cloud>/summary.csv \
PUE=<provider value> .venv/bin/python simulator/main.py --requireMeasurements
```

| Variable | Meaning |
|---|---|
| `MEASUREMENTS` | comma-separated `summary.csv` files, one edge and one cloud (at least 3 repetitions each) |
| `MODEL_NAME` | only if a results folder mixes base models |
| `PUE` | cloud data-centre PUE, from the provider (default 1.2) |
| `ALPHA_NODE` | non-GPU share of node power where no wall measurement exists (default 0.4) |
| `EDGE_WALL_RATIO` | override the measured wall-to-GPU ratio of the edge (normally empty) |

Without `MEASUREMENTS` the simulator uses a temporary compute model, for development only;
`--requireMeasurements` refuses to run in that case.

`weightedSweep.py` sweeps the latency weight of the `weighted` policy
(score = E/Emin + w·T/Tmin) and plots energy against latency:
`.venv/bin/python simulator/weightedSweep.py --out results/weightedSweep`.
