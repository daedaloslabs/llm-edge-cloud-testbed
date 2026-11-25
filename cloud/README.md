# Cloud node

The cloud node runs the same vLLM version, the same model checkpoints and the same serving
flags as the edge node, on a rented GPU server. Energy is the GPU energy counter (NVML); the
wall level is not accessible, so the provider's published PUE is applied in the simulator.

Used for the paper: Scaleway L40S-1-48G in Paris (PAR2), Ubuntu GPU image, billed per hour.
Any Ubuntu machine with an NVIDIA GPU and its driver works.

## 1. Rent the node

1. Create the instance with the GPU image (driver preinstalled) and an SSH key.
2. Security group: allow SSH (22) and TCP 8000 **only from the client's public IP**.
3. Billing runs while the instance exists: delete it after each block of runs.

## 2. Setup (once per instance)

The repository is private, so copy it from the edge node instead of cloning:

```bash
rsync -a --exclude .venv --exclude .cache --exclude results --exclude '.env' \
    llm-edge-cloud-testbed/ root@<cloud-ip>:llm-edge-cloud-testbed/
ssh root@<cloud-ip>
cd llm-edge-cloud-testbed
bash cloud/run.sh setup       # first run creates cloud/.env and stops: fill it in, rerun
```

Setup first checks that the GPU power can be read (a node that cannot be measured costs
minutes, not hours), then installs everything, downloads the models, generates `API_KEY`
and runs `bash cloud/run.sh check`.

Once the repository is public, `git clone <repo-url>` replaces the `rsync`.

`cloud/.env` (from `cloud/.env.example`):

| Setting | Meaning |
|---|---|
| `MODEL` | checkpoint to serve, same choice as on the edge |
| `PORT` | vLLM port, default 8000 (must match the security group) |
| `API_KEY` | left empty, setup generates a random key; the server is on a public IP, so it is required |
| `LOCK_CLOCK_MHZ` | lock the GPU core clock during serving; empty = not locked |

Copy the generated `API_KEY` into `client/.env` as `LLM_API_KEY`. It never leaves these two files.

## 3. A campaign

Two SSH sessions (or tmux, so the run survives a dropped connection):

```bash
bash cloud/run.sh serve
bash cloud/run.sh log cloud-INT4-<date>
```

Then the client, with `SERVER_TYPE=cloud`, `SERVER_URL=http://<cloud-ip>:8000` and
`GPU_MODEL=NVIDIA L40S` in `client/.env`. Stop both with Ctrl+C when the client finishes.

## 4. Bring the data home, delete the instance

```bash
scp -r root@<cloud-ip>:llm-edge-cloud-testbed/results/cloud-INT4-<date> results/   # on the edge
```

Then copy `power.csv` into the client's run folder and run `common/energy.py` as for the edge
(`edge/README.md`, section 4). No `plug.csv` on the cloud. Delete the instance afterwards.
