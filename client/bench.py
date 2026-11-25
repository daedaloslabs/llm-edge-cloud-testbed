"""
Load generator for an OpenAI-compatible server (vLLM for the paper, Ollama for the pilot).

For every request it logs: send time, first-token time, end time, the arrival time of
every streamed chunk (needed for per-request energy attribution) and the token counts
reported by the server. Times are wall-clock nanoseconds (time.time_ns) so they can be
aligned with the power log of the server.

Single config:
    python bench.py --url http://EDGE:8000 --model M --inputs 512 --outputs 128 --concurrency 10

Sweep (cartesian product, every config repeated --reps times, order shuffled with --seed):
    python bench.py --url ... --inputs 128,512,2048 --outputs 64,256,768 --concurrency 1,10,40 --reps 5

Outputs (appended, so several sweeps can share a folder):
    OUT/runs.jsonl      one line per configuration run
    OUT/requests.jsonl  one line per request
"""

import argparse
import itertools
import json
import os
import random
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Common English words: almost always one token each (with the leading space) in
# Llama/Qwen/Mistral tokenizers, so "N words" ~= "N prompt tokens". The exact count is
# taken from the server's usage report anyway.
WORDS = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city name "
    "team minute idea body back parent face level office door health person art war "
    "history party result change morning reason research girl moment air teacher force"
).split()


def makePrompt(words, rng):
    # Random words make every prompt unique, so prefix caching cannot help even if
    # someone forgets to disable it.
    return "Continue the following text: " + " ".join(rng.choice(WORDS) for _ in range(words))


API_KEY = os.environ.get("LLM_API_KEY")  # set when the server runs with --api-key (cloud)


def sendRequest(url, model, prompt, maxTokens, ignoreEos, seed):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": maxTokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignoreEos:
        body["ignore_eos"] = True  # vLLM extension; Ollama silently ignores it

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
        | ({"Authorization": "Bearer " + API_KEY} if API_KEY else {}),
    )

    chunks, usage, error = [], None, None
    tSend = time.time_ns()
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            for raw in response:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                message = json.loads(data)
                if message.get("usage"):
                    usage = message["usage"]
                choices = message.get("choices") or []
                if choices and choices[0].get("text"):
                    chunks.append(time.time_ns())
    except Exception as exc:  # logged, never hidden: failed requests are a measured quantity
        error = repr(exc)
    tEnd = time.time_ns()

    return {
        "tSendNs": tSend,
        "tFirstNs": chunks[0] if chunks else None,
        "tEndNs": tEnd,
        "chunkNs": chunks,
        "inputTokens": (usage or {}).get("prompt_tokens"),
        "outputTokens": (usage or {}).get("completion_tokens", len(chunks)),
        "status": "ok" if error is None and chunks else "error",
        "error": error,
    }


def runConfig(args, cfg, outDir, rng):
    """Run one configuration. Closed loop if requestRate == 0, Poisson open loop otherwise."""
    runId = uuid.uuid4().hex[:12]
    lock = threading.Lock()
    seedBase = rng.randrange(1 << 30)
    prompts = [makePrompt(cfg["inputs"], random.Random(seedBase + i)) for i in range(args.requests)]

    # Warm-up: not logged (first requests after model load are always outliers).
    for i in range(args.warmup):
        sendRequest(args.url, args.model, makePrompt(cfg["inputs"], rng), cfg["outputs"], args.ignore_eos, i)

    # Cool-down doubles as the idle-loaded baseline window for energy.py.
    tIdleStart = time.time_ns()
    time.sleep(args.cooldown)
    tStart = time.time_ns()

    def one(i):
        result = sendRequest(args.url, args.model, prompts[i], cfg["outputs"], args.ignore_eos, seedBase + i)
        result.update(runId=runId, requestIndex=i)
        with lock, open(outDir / "requests.jsonl", "a") as handle:
            handle.write(json.dumps(result) + "\n")

    if cfg["rate"] > 0:
        with ThreadPoolExecutor(max_workers=256) as pool:
            for i in range(args.requests):
                pool.submit(one, i)
                time.sleep(rng.expovariate(cfg["rate"]))
    else:
        counter = itertools.count()

        def worker():
            for i in counter:
                if i >= args.requests:
                    return
                one(i)

        threads = [threading.Thread(target=worker) for _ in range(cfg["concurrency"])]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    run = {
        "runId": runId,
        "serverType": args.server_type,
        "gpuModel": args.gpu_model,
        "modelName": args.model,
        "quantization": args.quantization,
        "engine": args.engine,
        "engineVersion": args.engine_version,
        "targetInputTokens": cfg["inputs"],
        "targetOutputTokens": cfg["outputs"],
        "concurrentRequests": cfg["concurrency"],
        "requestRate": cfg["rate"],
        "repetitionId": cfg["rep"],
        "randomSeed": seedBase,
        "ignoreEos": args.ignore_eos,
        "cacheEnabled": False,
        "networkProfile": args.network_profile,
        "networkRttMs": args.rtt_ms,
        "tIdleStartNs": tIdleStart,
        "tStartNs": tStart,
        "tEndNs": time.time_ns(),
    }
    with open(outDir / "runs.jsonl", "a") as handle:
        handle.write(json.dumps(run) + "\n")
    return run


def intList(text):
    return [int(x) for x in str(text).split(",") if x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--server-type", default="edge", choices=["edge", "cloud"])
    p.add_argument("--gpu-model", default="")
    p.add_argument("--quantization", default="INT4")
    p.add_argument("--engine", default="vllm")
    p.add_argument("--engine-version", default="")
    p.add_argument("--inputs", type=intList, default=[512])
    p.add_argument("--outputs", type=intList, default=[128])
    p.add_argument("--concurrency", type=intList, default=[1])
    p.add_argument("--rates", type=lambda s: [float(x) for x in s.split(",")], default=[0.0],
                   help="Poisson arrivals per second; 0 = closed loop at --concurrency")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--requests", type=int, default=50, help="requests per configuration run")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=20.0, help="seconds idle before each run (baseline)")
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--network-profile", default="none")
    p.add_argument("--rtt-ms", type=float, default=0.0, help="measured client-server RTT (ping)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    configs = [
        {"inputs": i, "outputs": o, "concurrency": c, "rate": r, "rep": rep}
        for i, o, c, r, rep in itertools.product(
            args.inputs, args.outputs, args.concurrency, args.rates, range(args.reps)
        )
    ]
    rng.shuffle(configs)  # randomized order: thermal drift does not line up with one variable

    for n, cfg in enumerate(configs, 1):
        print(f"[{n}/{len(configs)}] {cfg}", flush=True)
        runConfig(args, cfg, outDir, rng)


if __name__ == "__main__":
    main()
