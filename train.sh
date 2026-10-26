#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/cockroach ]]; then
    printf 'Önce ./setup.sh çalıştırın.\n' >&2
    exit 1
fi
exec .venv/bin/cockroach train --minutes "${1:-5}" "${@:2}"
