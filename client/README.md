# Client

The client generates the load and the timestamps: it sends the requests, records when each
token arrives and measures the network round trip. It needs no GPU, only Python 3.10+ and
git (setup installs them). Energy is recorded on the server, not here.

| File | Purpose |
|---|---|
| `setup.sh` / `setup.ps1` | one-time setup: Python, git, network time, `.env`, then a check |
| `.env.example` | settings template, copied to `client/.env` (never committed, holds the API key) |
| `run.py` | `--check`, `--quick` smoke run, or the full campaign |
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

## Troubleshooting

| Message | Fix |
|---|---|
| `server not reachable` | server not running, wrong `SERVER_URL`, firewall, or (cloud) security group |
| `MODEL ... is not served there` | `MODEL` must equal the server's `MODEL` |
| `401` / `Unauthorized` | `LLM_API_KEY` missing or different from the server's `API_KEY` |
| `NOT synchronised` | enable network time (setup does it), wait a minute, rerun `--check` |
