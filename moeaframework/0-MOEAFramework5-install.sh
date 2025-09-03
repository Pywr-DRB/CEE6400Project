#!/usr/bin/env bash
set -euo pipefail

MOEA_DIR="MOEAFramework-5.0"
CLI="$MOEA_DIR/cli"
URL="https://github.com/MOEAFramework/MOEAFramework/releases/download/v5.0/MOEAFramework-5.0.tar.gz"
TAR="MOEAFramework-5.0.tar.gz"

if [[ ! -d "$MOEA_DIR" ]]; then
  echo "MOEAFramework-5.0 not found. Downloading..."
  curl -L -o "$TAR" "$URL"
  tar -xzf "$TAR"
  rm -f "$TAR"
fi

if [[ ! -x "$CLI" ]]; then
  echo "Error: CLI at $CLI is not executable. Run:"
  echo "chmod 775 $CLI"
  exit 1
fi
echo "MOEAFramework 5 ready."
