#!/bin/bash
# Patch the connector installed in this directory for OpenAI tool-call conformance.
# install.sh runs this for you; it is here separately so you can re-run it after
# reinstalling the connector by hand.
#
# Without it, the connector returns tool calls with no "id" and no "type", so an
# agent framework cannot tie a tool result back to the call that produced it. It
# drops the call and the model reissues it forever -- strands-agents logs
# "incomplete tool use block, skipping". Details: work/NXP-eiq-aaf-connector-issues.md
#
# Safe to re-run: it does nothing if the patch is already in place. To undo it,
# reinstall the connector into the venv.

set -e

cd "$(dirname "$0")"

PKG=$(echo venv/lib/python3.*/site-packages/eiq_aaf_connector)

[ -d "$PKG" ] || { echo "connector not installed here -- run ./install.sh first"; exit 1; }

if grep -q "uuid4().hex" "$PKG/models/models.py"; then
    echo "already patched: $PKG"
    exit 0
fi

# -F3 tolerates the small import-block differences between connector releases, so
# one patch covers both the 2.0.0 and 2.1 lines. The two files live in different
# directories, so feed each half of the patch to the directory it belongs in.
sed -n '/^--- a\/models.py/,/^--- a\/Llm.py/p' connector-openai-fix.patch | sed '$d' \
    | patch -d "$PKG/models" -p1 -F3 --quiet
sed -n '/^--- a\/Llm.py/,$p' connector-openai-fix.patch \
    | patch -d "$PKG/llm_engines" -p1 -F3 --quiet

echo "patched: $PKG"
echo "restart the connector for it to take effect: ./run.sh"
