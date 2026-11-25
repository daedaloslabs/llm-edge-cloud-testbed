# Edge node

The edge node serves the model with vLLM and records its energy: the GPU energy counter
(NVML) every 20 ms and, through a Shelly Plug S Gen3, the power of the whole machine at the
wall every second. It is also the analysis machine (energy attribution, simulator).

Requirements: Linux, an NVIDIA GPU with its driver installed (`nvidia-smi` works), `curl`,
about 30 GB free disk, internet for the setup. Everything else is installed by setup.

## 1. Setup (once)

```bash
git clone <repo-url> llm-edge-cloud-testbed && cd llm-edge-cloud-testbed
bash edge/run.sh setup        # first run creates edge/.env and stops: fill it in, rerun
```

`edge/.env` (from `edge/.env.example`):

| Setting | Meaning |
|---|---|
| `MODEL` | checkpoint to serve: `Qwen/Qwen2.5-7B-Instruct-AWQ` (INT4) or `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8` (INT8) |
| `PORT` | vLLM port, default 8000 |
| `API_KEY` | optional on a private LAN; if set, the client needs the same value |
| `PLUG_IP` | local IP of the Shelly plug the edge machine is powered through |
| `LOCK_CLOCK_MHZ` | lock the GPU core clock during serving (needs sudo); empty = not locked |

Setup installs `uv`, a Python 3.12 environment in `.venv/`, the pinned vLLM and analysis
packages, the precompiled kernels, and both pinned model checkpoints into `.cache/`. It ends
with the checks below.

Shelly plug: join it to the WiFi with the Shelly app, give it a fixed IP in the router, then
`curl "http://<ip>/rpc/Switch.GetStatus?id=0"` must show `"output":true` and a non-zero `apower`.
If `output` is false: `curl "http://<ip>/rpc/Switch.Set?id=0&on=true"`.

## 2. Check (before every campaign)

```bash
bash edge/run.sh check
```

| Check | Why |
|---|---|
| GPU and NVML energy counter advancing | energy is read from the counter, not from sampled power |
| clock lock (if `LOCK_CLOCK_MHZ` set) | stable clocks between runs |
| network time synchronised | client and server timestamps are compared |
| Shelly plug reachable and on | wall-level energy |
| battery current 0 | on a laptop, a charging or discharging battery hides power from the plug |
| no USB-C power input | a second charger bypasses the plug |

Laptop: set the battery charge limit below the current level so the battery stays idle
(`echo 80 | sudo tee /sys/class/power_supply/BAT*/charge_control_end_threshold`). The
battery status may still read `Charging` with a current of 0; the current is what counts.

## 3. A campaign

Two terminals (or tmux), on the edge node:

```bash
bash edge/run.sh serve                 # vLLM with the fixed flags; ready when it prints the port
bash edge/run.sh log edge-INT4-<date>  # power logger -> results/edge-INT4-<date>/
```

Then start the client (`client/README.md`). Discard the first ~10 minutes: the laptop GPU
needs that long to reach a stable temperature. When the client finishes, stop the logger and
the server with Ctrl+C. For INT8, change `MODEL` in `edge/.env` and repeat.

## 4. After a campaign (energy attribution)

```bash
scp -r <client>:llm-edge-cloud-testbed/results/<run> results/
cp results/edge-INT4-<date>/{power,plug}.csv results/<run>/
.venv/bin/python common/energy.py --results results/<run>   # -> requestLevel.csv, summary.csv
```

`summary.csv` holds one row per configuration (latency percentiles, GPU and wall energy, energy
per request and per token, wall-to-GPU ratio). It is what the simulator reads.

## Troubleshooting

| Message | Fix |
|---|---|
| `Created edge/.env from the example` | fill it in, rerun |
| `NVML energy counter unusable` | driver too old or GPU without the counter |
| `GPU clock cannot be locked` | empty `LOCK_CLOCK_MHZ`; clocks are logged anyway |
| `battery ... current` | set the charge limit below the battery level, wait until the current is 0 |
| `Shelly plug relay is off` | run the `Switch.Set` command it prints |
| `MODEL ... is not one of the pinned models` | use one of the two checkpoints above |
