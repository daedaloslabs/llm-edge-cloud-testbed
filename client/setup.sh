#!/usr/bin/env bash
# Client setup for Linux and macOS: Python 3.10+, git, network time, then a check run.
#   bash client/setup.sh
set -euo pipefail
cd "$(dirname "$0")"

has() { command -v "$1" >/dev/null 2>&1; }
pyOk() { has python3 && python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; }
install() {  # generic names; mapped per package manager
    if has apt-get; then sudo apt-get update -qq && sudo apt-get install -y "$@"
    elif has dnf; then sudo dnf install -y "$@"
    elif has pacman; then sudo pacman -S --needed --noconfirm "${@/python3/python}"
    elif has brew; then brew install "${@/python3/python@3.12}"
    else echo "No known package manager: install $* by hand, then rerun."; exit 1; fi
}

pyOk || install python3
has git || install git
pyOk || { echo "Python 3.10+ still missing (open a new terminal and rerun)."; exit 1; }
echo "python: $(python3 --version)   git: $(git --version)"

echo "== network time"
if has timedatectl; then
    sudo timedatectl set-ntp true || true
    timedatectl show -p NTPSynchronized --value | sed 's/^/NTPSynchronized=/'
elif [ "$(uname)" = Darwin ]; then
    sudo systemsetup -setusingnetworktime on || true
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created client/.env: fill in SERVER_URL and the server details, then run:"
    echo "  python3 client/run.py --check"
    exit 0
fi
python3 run.py --check
