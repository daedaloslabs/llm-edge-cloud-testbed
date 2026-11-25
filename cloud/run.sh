#!/usr/bin/env bash
# cloud serving node: bash cloud/run.sh setup | check | serve | log NAME   (see cloud/README.md)
exec bash "$(dirname "$0")/../common/server.sh" cloud "$@"
