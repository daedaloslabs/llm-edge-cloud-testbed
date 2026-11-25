# LLM inference testbed: edge vs cloud

Measurement testbed for energy-efficient LLM serving across an edge node and a cloud node.
It measures, per request, the latency (time to first token, time per output token, completion
time) and the energy (GPU and, on the edge, wall plug) of the same model served by the same
engine on two different machines. The resulting `summary.csv` files feed a 5G HetNet
simulator that decides, request by request, whether to serve at the edge or in the cloud.

## Machines

```
 client laptop  ── LAN, Ethernet ──>  edge node   (laptop GPU, Shelly wall plug)
       │
       └──────── internet ─────────>  cloud node  (rented GPU server)
```

| Folder | Runs on | What it does |
|---|---|---|
| [`client/`](client/README.md) | client laptop (no GPU, Linux, macOS or Windows) | sends the requests, timestamps every token, measures the round trip |
| [`edge/`](edge/README.md) | edge node (NVIDIA GPU, Linux) | serves the model with vLLM, logs GPU energy and wall power |
| [`cloud/`](cloud/README.md) | cloud node (NVIDIA GPU, Ubuntu) | same as the edge, GPU energy only |
| `common/` | edge and cloud | shared code: node scripts, power logger, energy attribution, pinned requirements |
| [`simulator/`](simulator/README.md) | edge node (analysis) | radio simulator and joint optimizer, reads the measurements |

The client is separate from the servers on purpose: the machine that measures latency does
not compete for CPU or memory with the machine being measured.

## What is fixed and what is a setting

Fixed by the methodology, identical on both nodes, so they are in code and not in `.env`:
vLLM version and Python packages (`common/requirements-*.txt`), the two model checkpoints
pinned to a commit, and the serving flags (`common/server.sh`).

Settings per machine (addresses, ports, keys, plug IP) live in `<folder>/.env`, created from
`<folder>/.env.example`. The `.env` files are never committed.

## A measurement campaign, end to end

1. Edge: `bash edge/run.sh setup` once, then `bash edge/run.sh check`.
2. Client: `bash client/setup.sh` once, fill in `client/.env`, then `python3 client/run.py --check`.
3. Edge, two terminals: `bash edge/run.sh serve` and `bash edge/run.sh log <name>`.
4. Client: `python3 client/run.py` (several hours for the full grid).
5. Stop the logger, copy the server's `power.csv` and `plug.csv` into the client's run folder,
   run `common/energy.py` on it. The result is `summary.csv`.
6. Repeat for INT8, then the same on the cloud node.
7. Simulator: `MEASUREMENTS=<edge summary.csv>,<cloud summary.csv> python simulator/main.py`.

Each step is described in the README of its folder.

## Results

Measurement output goes to `results/` and is not committed here. Finished campaigns are
archived with the paper material in the research repository.

## Check

`.venv/bin/python common/test_testbed.py` prints `ok` (energy attribution and the
interpolation model, on synthetic data).
