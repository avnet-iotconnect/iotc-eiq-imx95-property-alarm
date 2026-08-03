#!/bin/bash
# Start the eIQ AAF Connector on the board, replacing any running instance.
# Run this ON the board, from the directory it lives in.
#
#   ./run.sh            start (or restart) on port 3000, in the background
#   ./run.sh --stop     stop it and exit
#
# Loading a model takes minutes (~225 s for the 7B) and the server does not bind
# its port until every model marked "enabled" in server_config.json is resident.
# Until then "connection refused" is expected, not a fault.

set -e

cd "$(dirname "$0")"
PORT=3000
LOG=server.log

# The pattern is written bin/conn[e]ctor so it cannot match this script's own
# command line -- plain "pkill -f connector" over ssh kills the ssh session too.
PIDS=$(pgrep -f "bin/conn[e]ctor" || true)

if [ -n "$PIDS" ]; then
    # SIGTERM does not get through: uvicorn is blocked inside the Ara model load
    # and never reaches its signal handler, so go straight to SIGKILL.
    echo "stopping connector: $PIDS"
    kill -9 $PIDS
    sleep 2
fi

[ "$1" = "--stop" ] && exit 0

# setsid + </dev/null so it survives the ssh session that launched it.
setsid nohup ./connector-venv/bin/connector --host 0.0.0.0 --port "$PORT" \
    > "$LOG" 2>&1 < /dev/null &

echo "connector starting on :$PORT, logging to $(pwd)/$LOG"
echo "watch it come up:   tail -f $LOG"
echo "ready when it says: Uvicorn running on http://0.0.0.0:$PORT"
echo "then:               curl http://localhost:$PORT/v1/models"
