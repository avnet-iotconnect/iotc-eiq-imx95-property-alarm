#!/bin/bash
# Apply the OpenAI-conformance fix to the eIQ AAF Connector checkout on the board.
#
# Keeps a .orig of every file it touches so you can diff, and re-running is safe:
# it restores from .orig first, so the patch is always applied to pristine sources.
#
#   ./jaguar/apply-connector-patch.sh          apply
#   ./jaguar/apply-connector-patch.sh --revert restore the originals and stop
#
# What it fixes is explained in work/NXP-eiq-aaf-connector-issues.md.

set -e

BOARD=root@192.168.38.203
SRC=/root/jaguar-aaf/src/eiq_aaf_connector
FILES="models/models.py llm_engines/Llm.py"

# Restore first, so this script is idempotent rather than stacking patches.
ssh $BOARD "for f in $FILES; do
    [ -f $SRC/\$f.orig ] && cp $SRC/\$f.orig $SRC/\$f || cp $SRC/\$f $SRC/\$f.orig
done"

if [ "$1" = "--revert" ]; then
    echo "Originals restored. Restart the connector to pick them up."
    exit 0
fi

# -p1 strips the a/ and b/ prefixes; both files live in different dirs, so feed
# the patch from the package root and let it match on basename.
ssh $BOARD "cd $SRC/models && patch -p1 --quiet" < <(sed -n '/^--- a\/models.py/,/^--- a\/Llm.py/p' \
    "$(dirname "$0")/connector-openai-fix.patch" | sed '$d')
ssh $BOARD "cd $SRC/llm_engines && patch -p1 --quiet" < <(sed -n '/^--- a\/Llm.py/,$p' \
    "$(dirname "$0")/connector-openai-fix.patch")

ssh $BOARD "cd $SRC && python3 -c 'import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]' $FILES" \
    && echo "Patched and parsing cleanly. Restart the connector to pick it up."
