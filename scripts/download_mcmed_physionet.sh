#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PHYSIONET_USER:-}" || -z "${PHYSIONET_PASSWORD:-}" ]]; then
  echo "Set PHYSIONET_USER and PHYSIONET_PASSWORD before running." >&2
  exit 2
fi

VERSION="${MCMED_VERSION:-1.0.1}"
TARGET="${MCMED_DIR:-/root/project/data/mc-med}"
URL="https://physionet.org/files/mc-med/${VERSION}/"

mkdir -p "${TARGET}"

wget \
  --recursive \
  --no-parent \
  --continue \
  --timestamping \
  --no-host-directories \
  --cut-dirs=3 \
  --reject "index.html*" \
  --user "${PHYSIONET_USER}" \
  --password "${PHYSIONET_PASSWORD}" \
  --directory-prefix "${TARGET}" \
  "${URL}"

mkdir -p "${TARGET}/waveforms"
for archive in "${TARGET}"/waveforms_*.zip; do
  if [[ -f "${archive}" ]]; then
    unzip -n "${archive}" -d "${TARGET}/waveforms"
  fi
done

echo "MC-MED data prepared at ${TARGET}"

