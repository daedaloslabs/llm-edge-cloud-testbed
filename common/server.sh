#!/usr/bin/env bash
# Serving node (edge or cloud). Called through edge/run.sh or cloud/run.sh:
#
#   bash edge/run.sh setup     install everything, download the models, then check
#   bash edge/run.sh check     can this node be measured? (GPU, energy counter, clocks, time, plug, battery)
#   bash edge/run.sh serve     start vLLM with the fixed measurement settings (Ctrl+C to stop)
#   bash edge/run.sh log NAME  start the power logger into results/NAME (Ctrl+C to stop)
#
# Settings come from <node>/.env (copied from <node>/.env.example). The values below are
# fixed by the methodology and must be identical on both nodes, so they are not settings.
set -euo pipefail

NODE="${1:?usage: server.sh edge|cloud setup|check|serve|log}"; shift
CMD="${1:?usage: server.sh $NODE setup|check|serve|log}"; shift
[[ "$NODE" =~ ^(edge|cloud)$ ]] || { echo "node must be edge or cloud"; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO/$NODE/.env"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
export HF_HOME="$REPO/.cache/huggingface"

# --- fixed by the methodology ---------------------------------------------------------
PYTHON_VERSION=3.12
# Model snapshots pinned to a commit, so both nodes load byte-identical weights.
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e9394b818fdfca67ab2a00ecc7dd641"        # INT4
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8@6711f2b6fc95545efe6018c469eca6832e28cefd"  # INT8
)
SERVE_FLAGS=(--max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 64
             --no-enable-prefix-caching --seed 0 --generation-config vllm)

say() { printf '\n== %s\n' "$*"; }
ok() { printf '  ok    %s\n' "$*"; }
bad() { printf '  FAIL  %s\n' "$*"; FAILED=1; }
note() { printf '  note  %s\n' "$*"; }
has() { command -v "$1" >/dev/null 2>&1; }
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

loadEnv() {
    [ -f "$ENV_FILE" ] || { cp "$REPO/$NODE/.env.example" "$ENV_FILE"; chmod 600 "$ENV_FILE"
        echo "Created $NODE/.env from the example. Fill it in, then rerun."; exit 1; }
    set -a; # shellcheck disable=SC1090
    source <(sed 's/[[:space:]]#.*$//' "$ENV_FILE"); set +a
    : "${MODEL:?MODEL is empty in $NODE/.env}"
    PORT="${PORT:-8000}"
}

revisionOf() {  # MODEL -> pinned commit
    local entry
    for entry in "${MODELS[@]}"; do [ "${entry%@*}" = "$1" ] && { echo "${entry#*@}"; return; }; done
    echo "MODEL $1 is not one of the pinned models: ${MODELS[*]%@*}" >&2; exit 1
}

# --- setup ----------------------------------------------------------------------------
setup() {
    loadEnv
    # Fail in seconds, before gigabytes of downloads, if this node cannot be measured.
    has nvidia-smi || { echo "nvidia-smi missing: install the NVIDIA driver (cloud: pick a GPU image)"; exit 1; }
    local power; power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    [[ "$power" =~ ^[0-9.]+$ ]] || { echo "GPU power not readable ('$power'): energy cannot be measured here"; exit 1; }
    if has apt-get; then
        say "system packages"
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq curl chrony openssl >/dev/null
        $SUDO systemctl enable --now chrony >/dev/null 2>&1 || true
    fi
    if ! has uv; then
        say "installing uv (Python and package manager)"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    say "Python $PYTHON_VERSION environment in .venv"
    [ -x "$PY" ] || uv venv --python "$PYTHON_VERSION" "$VENV"
    uv pip install --python "$PY" -r "$REPO/common/requirements-server.txt"
    [ "$NODE" = edge ] && uv pip install --python "$PY" -r "$REPO/common/requirements-analysis.txt"
    "$VENV/bin/vllm" --version
    "$VENV/bin/vllm" download-kernels   # precompiled FlashInfer kernels, no compile at first serve

    say "downloading the pinned models into .cache/"
    local entry
    for entry in "${MODELS[@]}"; do
        "$PY" -c "from huggingface_hub import snapshot_download as d; print(d('${entry%@*}', revision='${entry#*@}'))"
    done

    if [ "$NODE" = cloud ] && [ -z "${API_KEY:-}" ]; then
        local key; key=$(openssl rand -hex 24)
        sed -i "s/^API_KEY=.*/API_KEY=$key/" "$ENV_FILE"
        say "generated API_KEY in cloud/.env; put the same value in client/.env as LLM_API_KEY"
    fi
    check
}

# --- check ----------------------------------------------------------------------------
check() {
    loadEnv
    FAILED=0
    say "checks for the $NODE node"
    [ -x "$PY" ] && ok "Python environment $("$PY" --version)" || bad "no .venv: run setup first"

    if has nvidia-smi; then
        ok "GPU $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
    else
        bad "nvidia-smi missing (no NVIDIA driver; on the cloud pick a GPU image)"
    fi

    if [ -x "$PY" ] && "$PY" - <<'EOF'
import time, pynvml
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
a = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); time.sleep(2)
b = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
print(f"  ok    NVML energy counter advanced {b - a} mJ in 2 s (~{(b - a) / 2000:.1f} W)")
raise SystemExit(b <= a)
EOF
    then :; else bad "NVML energy counter unusable: energy cannot be measured on this GPU"; fi

    if [ -n "${LOCK_CLOCK_MHZ:-}" ]; then
        if $SUDO nvidia-smi -lgc "$LOCK_CLOCK_MHZ,$LOCK_CLOCK_MHZ" >/dev/null 2>&1; then
            ok "GPU clock can be locked at $LOCK_CLOCK_MHZ MHz"; $SUDO nvidia-smi -rgc >/dev/null
        else bad "GPU clock cannot be locked: empty LOCK_CLOCK_MHZ and state it in the paper"; fi
    else note "clock not locked (LOCK_CLOCK_MHZ empty); clocks are logged per sample"; fi

    if has chronyc && chronyc tracking >/dev/null 2>&1; then
        ok "time: $(chronyc tracking | grep 'System time' | sed 's/ *: */: /')"
    elif has timedatectl; then
        [ "$(timedatectl show -p NTPSynchronized --value)" = yes ] && ok "time: NTP synchronised" \
            || bad "time not synchronised: sudo timedatectl set-ntp true, wait a minute"
    else note "no time tool found; check clock sync by hand"; fi

    if [ "$NODE" = edge ]; then edgeChecks; fi
    say "$([ "$FAILED" = 0 ] && echo 'all checks passed' || echo 'some checks FAILED, fix them before measuring')"
    return "$FAILED"
}

edgeChecks() {
    if [ -n "${PLUG_IP:-}" ]; then
        local status
        if status=$(curl -sf --max-time 3 "http://$PLUG_IP/rpc/Switch.GetStatus?id=0"); then
            echo "$status" | grep -q '"output":true' \
                && ok "Shelly plug $PLUG_IP on, $(echo "$status" | grep -o '"apower":[0-9.]*' | cut -d: -f2) W" \
                || bad "Shelly plug relay is off: curl \"http://$PLUG_IP/rpc/Switch.Set?id=0&on=true\""
        else bad "Shelly plug $PLUG_IP not reachable"; fi
    else note "PLUG_IP empty: no wall-level power"; fi

    # A laptop edge node: all power must pass through the plug and the battery must be idle.
    local ps=/sys/class/power_supply bat
    for bat in "$ps"/BAT*; do
        [ -e "$bat" ] || continue
        local current; current=$(cat "$bat/current_now" 2>/dev/null || cat "$bat/power_now" 2>/dev/null || echo "?")
        [ "$current" = 0 ] && ok "battery $(basename "$bat") idle at $(cat "$bat/capacity") %" \
            || bad "battery $(basename "$bat") current $current (charging or discharging): set the charge limit below the battery level"
    done
    local usb
    for usb in "$ps"/ucsi-source-psy-*; do
        [ -e "$usb" ] || continue
        [ "$(cat "$usb/online")" = 1 ] && bad "USB-C power input $(basename "$usb") connected: unplug it (it bypasses the plug)"
    done
    return 0
}

# --- serve / log ----------------------------------------------------------------------
serve() {
    loadEnv
    local revision; revision=$(revisionOf "$MODEL")
    if [ -n "${LOCK_CLOCK_MHZ:-}" ]; then
        $SUDO nvidia-smi -lgc "$LOCK_CLOCK_MHZ,$LOCK_CLOCK_MHZ"
        trap '$SUDO nvidia-smi -rgc' EXIT
    fi
    local auth=(); [ -n "${API_KEY:-}" ] && auth=(--api-key "$API_KEY")
    HF_HUB_OFFLINE=1 "$VENV/bin/vllm" serve "$MODEL" --revision "$revision" --host 0.0.0.0 --port "$PORT" \
        "${SERVE_FLAGS[@]}" "${auth[@]}"
}

log() {
    loadEnv
    local name="${1:?usage: run.sh log NAME (folder under results/)}"
    local plug=(); [ "$NODE" = edge ] && [ -n "${PLUG_IP:-}" ] && plug=(--plug "$PLUG_IP")
    exec "$PY" "$REPO/common/powerlog.py" --out "$REPO/results/$name" "${plug[@]}"
}

case "$CMD" in
    setup) setup ;;
    check) check ;;
    serve) serve ;;
    log) log "$@" ;;
    *) echo "unknown command $CMD (setup|check|serve|log)"; exit 1 ;;
esac
