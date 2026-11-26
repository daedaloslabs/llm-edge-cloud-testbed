#!/bin/bash
# One command for several campaigns back to back, e.g. both quantizations on the cloud:
#
#   bash client/campaign.sh INT4 INT8
#
# For each quantization: client/.env with MODEL and QUANTIZATION set accordingly -> (switch the server
# to that model) -> wait until it is served -> 10 min warm-up -> run.py, with no pause in between.
# Stops at the first failure; continue that campaign with run.py --resume <folder>.
#
# Cloud: put SERVER_SSH=root@<cloud-ip> in client/.env and the server is switched over ssh (MODEL in
# cloud/.env, vLLM restarted in tmux session "serve"). The power logger keeps running throughout.
set -u
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || { echo "usage: bash client/campaign.sh INT4 [INT8]"; exit 1; }
[ -f client/.env ] || { echo "client/.env missing"; exit 1; }

for quant in "$@"; do
    case "$quant" in
        INT4) model=Qwen/Qwen2.5-7B-Instruct-AWQ ;;
        INT8) model=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 ;;
        *) echo "unknown quantization $quant (INT4 or INT8)"; exit 1 ;;
    esac
    envFile="client/.env.$quant"                 # git-ignored copy, kept for the record
    sed -e "s#^MODEL=.*#MODEL=$model#" -e "s#^QUANTIZATION=.*#QUANTIZATION=$quant#" client/.env > "$envFile"

    unset SERVER_SSH LLM_API_KEY
    while IFS='=' read -r key value; do          # values may contain spaces (GPU_MODEL); drop comments
        value="${value%% #*}"
        export "$key=${value%"${value##*[! ]}"}"
    done < <(grep -E '^[A-Z_]+=' "$envFile")
    echo "== $(date +%T) $quant: $SERVER_TYPE $MODEL"

    if [ -n "${SERVER_SSH:-}" ]; then
        ssh -o BatchMode=yes "$SERVER_SSH" "cd llm-edge-cloud-testbed && sed -i 's#^MODEL=.*#MODEL=$MODEL#' cloud/.env \
            && tmux send-keys -t serve C-c && sleep 15 && tmux send-keys -t serve 'bash cloud/run.sh serve' Enter" \
            || { echo "could not switch the server over ssh"; exit 1; }
    fi

    auth=(); [ -n "${LLM_API_KEY:-}" ] && auth=(-H "Authorization: Bearer $LLM_API_KEY")
    for i in $(seq 90); do                        # up to 15 min for the model to load
        curl -s -m 5 "${auth[@]}" "$SERVER_URL/v1/models" | grep -q "\"$MODEL\"" && break
        [ "$i" = 90 ] && { echo "server does not serve $MODEL"; exit 1; }
        sleep 10
    done

    echo "== $(date +%T) warm-up 10 min"
    timeout 600 python3 client/bench.py --url "$SERVER_URL" --model "$MODEL" --server-type "$SERVER_TYPE" \
        --gpu-model "$GPU_MODEL" --quantization "$QUANTIZATION" --inputs 1024 --outputs 512 --concurrency 8 \
        --reps 100 --requests 400 --warmup 1 --cooldown 1 --out "results/warmup-$SERVER_TYPE-$QUANTIZATION" >/dev/null 2>&1

    echo "== $(date +%T) campaign $quant"
    python3 client/run.py --env "$envFile" || { echo "run.py failed on $quant"; exit 1; }
done
echo "== $(date +%T) all campaigns done"
