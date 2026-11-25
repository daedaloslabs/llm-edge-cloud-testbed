#!/usr/bin/env bash
# edge serving node: bash edge/run.sh setup | check | serve | log NAME   (see edge/README.md)
exec bash "$(dirname "$0")/../common/server.sh" edge "$@"
