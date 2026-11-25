"""
Client runner: reads client/.env, checks the server, measures the network RTT and runs
client/bench.py with the campaign grid. Standard library only, any OS.

    python run.py --check    settings, server reachable, model served, RTT, clock sync
    python run.py --quick    tiny smoke run (results/pilot-client-..., not committed)
    python run.py            full campaign (results/<server>-<quant>-<eos>-<time>)
"""

import argparse
import datetime
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REQUIRED = ["SERVER_URL", "SERVER_TYPE", "MODEL", "QUANTIZATION", "GPU_MODEL", "ENGINE_VERSION",
            "INPUTS", "OUTPUTS", "CONCURRENCY", "REPS", "REQUESTS", "IGNORE_EOS"]


def loadEnv(path):
    if not path.exists():
        sys.exit(f"{path} missing: copy client/.env.example to client/.env and fill it in")
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #")[0].strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        sys.exit(f"client/.env: missing values for {', '.join(missing)}")
    if env["SERVER_TYPE"] not in ("edge", "cloud"):
        sys.exit("client/.env: SERVER_TYPE must be edge or cloud")
    return env


def servedModels(env):
    headers = {"Authorization": "Bearer " + env["LLM_API_KEY"]} if env.get("LLM_API_KEY") else {}
    req = urllib.request.Request(env["SERVER_URL"].rstrip("/") + "/v1/models", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return [m["id"] for m in json.load(response)["data"]]
    except Exception as exc:  # report and stop: nothing can be measured without the server
        sys.exit(f"server not reachable at {env['SERVER_URL']}: {exc}")


def tcpRttMs(url, samples=20):
    # TCP handshake time ~ one round trip; works through firewalls that drop ping.
    parts = urllib.parse.urlparse(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        with socket.create_connection((parts.hostname, port), timeout=5):
            times.append((time.perf_counter() - start) * 1000.0)
        time.sleep(0.1)
    return statistics.median(times), min(times), max(times)


def clockStatus():
    system = platform.system()
    commands = {
        "Linux": [["timedatectl", "show", "-p", "NTPSynchronized", "--value"], ["chronyc", "tracking"]],
        "Darwin": [["sntp", "time.apple.com"]],
        "Windows": [["w32tm", "/query", "/status"]],
    }.get(system, [])
    report, synced = [], None
    for command in commands:
        try:
            out = subprocess.run(command, capture_output=True, text=True, timeout=15).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            continue
        report.append(f"$ {' '.join(command)}\n{out}")
        if command[0] == "timedatectl":
            synced = out == "yes"
    return "\n".join(report) or "no clock tool found", synced


def gitCommit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", metavar="NAME", help="continue results/NAME after a crash or stop")
    p.add_argument("--env", default=str(HERE / ".env"))
    args = p.parse_args()

    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is needed")
    env = loadEnv(Path(args.env))

    models = servedModels(env)
    print(f"server OK, serving: {', '.join(models)}")
    if env["MODEL"] not in models:
        sys.exit(f"MODEL {env['MODEL']} is not served there; fix client/.env or the server")

    rtt, rttMin, rttMax = tcpRttMs(env["SERVER_URL"])
    print(f"RTT (TCP connect, 20 samples): median {rtt:.2f} ms, min {rttMin:.2f}, max {rttMax:.2f}")
    clock, synced = clockStatus()
    print("clock:", {True: "NTP synchronised", False: "NOT synchronised (fix before measuring)",
                     None: "see clientInfo.txt"}[synced])
    if args.check:
        print("check passed")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    eos = "fixed" if env["IGNORE_EOS"] == "1" else "natural"
    name = f"pilot-client-{stamp}" if args.quick else f"{env['SERVER_TYPE']}-{env['QUANTIZATION']}-{eos}-{stamp}"
    if args.resume:
        name = Path(args.resume).name
        if not (REPO / "results" / name / "runs.jsonl").exists():
            sys.exit(f"nothing to resume: results/{name}/runs.jsonl not found")
    out = REPO / "results" / name
    out.mkdir(parents=True, exist_ok=True)

    grid = ({"INPUTS": "128", "OUTPUTS": "32", "CONCURRENCY": "1,2", "REPS": "1", "REQUESTS": "4"}
            if args.quick else env)
    command = [
        sys.executable, str(HERE / "bench.py"),
        "--url", env["SERVER_URL"], "--model", env["MODEL"],
        "--server-type", env["SERVER_TYPE"], "--gpu-model", env["GPU_MODEL"],
        "--quantization", env["QUANTIZATION"], "--engine", env.get("ENGINE", "vllm"),
        "--engine-version", env["ENGINE_VERSION"], "--network-profile", env.get("NETWORK_PROFILE", "none"),
        "--inputs", grid["INPUTS"], "--outputs", grid["OUTPUTS"], "--concurrency", grid["CONCURRENCY"],
        "--reps", grid["REPS"], "--requests", grid["REQUESTS"], "--rtt-ms", f"{rtt:.2f}", "--out", str(out),
    ] + (["--ignore-eos"] if env["IGNORE_EOS"] == "1" else []) + (
        ["--warmup", "1", "--cooldown", "2"] if args.quick else [])

    info = {k: v for k, v in env.items() if k != "LLM_API_KEY"}
    with open(out / "clientInfo.txt", "a", encoding="utf-8") as handle:
        handle.write(
        f"{'resumed' if args.resume else 'started'}: {datetime.datetime.now().astimezone().isoformat()}\n"
        f"repo commit: {gitCommit()}\nclient: {platform.platform()} / Python {platform.python_version()}\n"
        f"rttMs: median {rtt:.3f} min {rttMin:.3f} max {rttMax:.3f}\n"
        f"settings: {json.dumps(info, indent=1)}\n\n{clock}\n\n")

    print(f"running -> {out}")
    childEnv = os.environ | ({"LLM_API_KEY": env["LLM_API_KEY"]} if env.get("LLM_API_KEY") else {})
    code = subprocess.call(command, env=childEnv)
    if code != 0:
        print(f"\nstopped early; continue with: python3 client/run.py --resume {name}")
    print(f"\nbench.py finished with code {code}. Next (see README.md, 'After a campaign'):\n"
          f"  copy power.csv (and plug.csv) from the server into {out}\n"
          f"  .venv/bin/python common/energy.py --results {out.relative_to(REPO)}")
    sys.exit(code)


if __name__ == "__main__":
    main()
