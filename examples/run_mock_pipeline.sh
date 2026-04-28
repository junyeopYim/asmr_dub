#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-input.wav}"
PROJECT="${2:-demo_project}"

asmr-dub init "$PROJECT"
asmr-dub run "$INPUT" --project "$PROJECT" --confirm-rights --mock
