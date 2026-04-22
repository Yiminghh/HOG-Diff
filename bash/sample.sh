#!/bin/bash
# Usage: bash bash/sample.sh <config>
#   e.g.  bash bash/sample.sh qm9
#         bash bash/sample.sh cs
set -e

cd "$(dirname "$0")/.."
source bash/env.sh

cfg=${1:?"usage: bash bash/sample.sh <config>   (e.g. qm9, zinc250k, enzymes, sbm, guacamol, moses, cs, ego)"}

python ./main.py \
  --config "$cfg" \
  --mode sample
