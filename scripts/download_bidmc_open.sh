#!/usr/bin/env bash
set -euo pipefail

TARGET="${BIDMC_DIR:-/root/project/data/bidmc}"
URL="${BIDMC_URL:-https://archive.physionet.org/physiobank/database/bidmc/}"

mkdir -p "${TARGET}"

set +e
wget \
  --recursive \
  --no-parent \
  --continue \
  --timestamping \
  --no-host-directories \
  --cut-dirs=3 \
  --reject "index.html*" \
  --directory-prefix "${TARGET}" \
  "${URL}"
wget_status=$?
set -e

if [[ ${wget_status} -ne 0 && ! -f "${TARGET}/RECORDS" ]]; then
  exit "${wget_status}"
fi

echo "BIDMC data prepared at ${TARGET}"
