# Client

The client generates the load and the timestamps: it sends the requests, records when each
token arrives and measures the network round trip. It needs no GPU, only Python 3.10+ and
git (setup installs them). Energy is recorded on the server, not here.

| File | Purpose |
|---|---|
| `setup.sh` / `setup.ps1` | one-time setup: Python, git, network time, `.env`, then a check |
| `.env.example` | settings template, copied to `client/.env` (never committed, holds the API key) |
| `run.py` | `--check`, `--quick` smoke run, or the full campaign |
| `campaign.sh` | several campaigns back to back in one command: warm-up, then `run.py`, per quantization |
| `bench.py` | the load generator `run.py` calls (standard library only) |

## Network

- Edge campaign: client and edge node on the **same local network, both on Ethernet** (same
  router, or one cable between them). WiFi adds random delay that ends up in the latency figures.
- Cloud campaign: any wired internet connection. Its public IP must be allowed in the cloud
  security group.
- Keep the client on mains power with sleep disabled while a campaign runs.

## 1. Setup (once)

```bash
git clone <repo-url> llm-edge-cloud-testbed && cd llm-edge-cloud-testbed
bash client/setup.sh                                          # Linux / macOS
powershell -ExecutionPolicy Bypass -File client\setup.ps1     # Windows
```

While the repository is private, log in to GitHub on the client first (an SSH key or
`gh auth login`), or copy the folder from the edge node.

The first run creates `client/.env`. Fill it in:

| Setting | Meaning |
|---|---|
| `SERVER_URL` | `http://<server-ip>:8000` (edge: its LAN IP, `ip -4 addr`; cloud: its public IP) |
| `SERVER_TYPE` | `edge` or `cloud` |
| `LLM_API_KEY` | same as `API_KEY` in the server's `.env` (empty if the server has none) |
| `MODEL`, `QUANTIZATION` | must match the server's `MODEL` |
| `GPU_MODEL`, `ENGINE_VERSION`, `NETWORK_PROFILE` | recorded with the results |
| `INPUTS`, `OUTPUTS`, `CONCURRENCY`, `REPS`, `REQUESTS` | the measurement grid |
| `IGNORE_EOS` | 1 = fixed output length (main sweeps), 0 = natural termination runs |
| `SERVER_SSH` | cloud only, optional: `root@<cloud-ip>`, lets `campaign.sh` switch the server's model |

Then, with the server running:

```bash
python3 client/run.py --check      # Windows: py -3 client\run.py --check
```

It must print `server OK`, the served model, the RTT and `NTP synchronised`.

## 2. A campaign

Start the server and its power logger first (`edge/README.md` or `cloud/README.md`), then:

```bash
git pull                           # same code as the server
python3 client/run.py --quick      # one-minute smoke run
python3 client/run.py              # the campaign, several hours for the full grid
```

Results go to `results/<edge|cloud>-<INT4|INT8>-<fixed|natural>-<date-time>/` with
`runs.jsonl`, `requests.jsonl` and `clientInfo.txt` (settings without the API key, RTT, clock
status, repository commit). Copy that folder to the edge node for the energy attribution.

Every point is saved as soon as it finishes. A run with failed requests is excluded and repeated
(up to 3 tries), so each point ends with `REPS` valid repetitions. After a crash or Ctrl+C,
continue the same folder (only missing points run; the server and power logger must be running):

```bash
python3 client/run.py --resume <edge|cloud>-<INT4|INT8>-<fixed|natural>-<date-time>
```

## 3. Both quantizations in one command (recommended)

`campaign.sh` runs one campaign per quantization given, with no pause in between, so the GPU
never cools down between the warm-up and the measurements (Linux / macOS; on Windows use WSL):

```bash
bash client/campaign.sh INT4 INT8      # cloud: both quantizations, one after the other
bash client/campaign.sh INT8           # e.g. only INT8 on the edge
```

Only the quantization is given on the command line. For each one the script:

1. writes `client/.env.INT4` (or `.INT8`): a copy of `client/.env` with `MODEL` and
   `QUANTIZATION` set to that quantization (git-ignored, kept as a record);
2. if `SERVER_SSH` is set (cloud), switches the server to that model over SSH: `MODEL` in the
   server's `cloud/.env`, then restarts vLLM in its tmux session `serve` (see `cloud/README.md`);
3. waits until the server serves the model (up to 15 min);
4. runs a 10 min warm-up (1024 in, 512 out, 8 concurrent; saved to `results/warmup-*`, not used);
5. starts `run.py` with that file.

Everything else (server address, key, grid, `REPS`) comes from `client/.env` unchanged. The script
stops at the first failure; continue that campaign with `run.py --resume <folder>` and, if needed,
run `campaign.sh` again for the remaining quantization.

Edge: the server runs on the edge node and is not switched over SSH, so leave `SERVER_SSH` out and
start the edge server with the right `MODEL` before (or right after) launching `campaign.sh`.

## Troubleshooting

| Message | Fix |
|---|---|
| `server not reachable` | server not running, wrong `SERVER_URL`, firewall, or (cloud) security group |
| `MODEL ... is not served there` | `MODEL` must equal the server's `MODEL` |
| `401` / `Unauthorized` | `LLM_API_KEY` missing or different from the server's `API_KEY` |
| `NOT synchronised` | enable network time (setup does it), wait a minute, rerun `--check` |
| `could not switch the server over ssh` | wrong `SERVER_SSH`, the client's SSH key is not on the server, or no tmux session `serve` there |
| `server does not serve <model>` | the server did not come up with that model within 15 min: check the `serve` session on the server |
