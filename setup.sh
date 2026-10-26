#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[viewer,dev]'
printf '\nHazır. İzle: ./view.sh    Eğit: ./train.sh 5\n'
