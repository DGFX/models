#!/bin/bash
# Make third_party/GVHMR a real git checkout. fdanyone.motion.gvhmr.validate_gvhmr
# refuses to run unless `git -C <gvhmr_root> rev-parse HEAD` returns a 40-character
# revision. A copied source tree has no .git directory, so the image build checks
# out the commit 4DAnyone pins.
set -euo pipefail

DEST=/opt/4DAnyone/third_party/GVHMR
PIN=6ec3ca39336c50492c0fae65fba2fb831fc7d866
ORIGIN=https://github.com/zju3dv/GVHMR.git

if [[ ! -f "${DEST}/hmr4d/__init__.py" ]]; then
  echo "Vendored GVHMR is missing at ${DEST}" >&2
  exit 1
fi

TMP="$(mktemp -d)"
cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

fetched=0
if git clone --filter=blob:none --no-checkout "${ORIGIN}" "${TMP}/GVHMR" \
  && git -C "${TMP}/GVHMR" fetch --depth 1 origin "${PIN}" \
  && git -C "${TMP}/GVHMR" checkout --force FETCH_HEAD \
  && [[ "$(git -C "${TMP}/GVHMR" rev-parse HEAD)" == "${PIN}" ]]; then
  fetched=1
fi

if [[ "${fetched}" -eq 1 ]]; then
  rm -rf "${DEST}"
  mv "${TMP}/GVHMR" "${DEST}"
  echo "Checked out GVHMR ${PIN}"
else
  echo "Could not fetch GVHMR ${PIN}; recording a local git commit on the vendored tree." >&2
  git -C "${DEST}" init
  git -C "${DEST}" add -A
  git -C "${DEST}" -c user.email=4danyone-build@localhost -c user.name=4danyone-build \
    commit -m "Vendored GVHMR ${PIN}"
fi

git config --global --add safe.directory "${DEST}"
revision="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${#revision}" -ne 40 ]]; then
  echo "GVHMR revision is not a 40-character SHA: ${revision}" >&2
  exit 1
fi
test -f "${DEST}/hmr4d/__init__.py"
test -f "${DEST}/tools/demo/demo.py"
echo "GVHMR HEAD ${revision}"
