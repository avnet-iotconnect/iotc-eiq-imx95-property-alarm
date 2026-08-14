#!/bin/bash
# Start the connector installed in this directory, replacing any running instance.
#
#   ./run.sh            start (or restart) on port 3000, in the background
#   ./run.sh --stop     stop it and exit
#
# Loading a model takes minutes (~225 s for a 7B) and the server does not bind its
# port until every model marked "enabled" in config/server_config.json is resident.
# Until then "connection refused" is expected, not a fault. Set every model to
# "enabled": false and the port comes up in about a minute with nothing loaded,
# ready for POST /v1/models to load one on demand.

set -e

cd "$(dirname "$0")"
PORT=3000

[ -x venv/bin/connector ] || { echo "connector not installed here -- run ./install.sh first"; exit 1; }

# The pattern is written bin/conn[e]ctor so it cannot match this script's own
# command line -- a plain "pkill -f connector" over ssh kills the ssh session too.
PIDS=$(pgrep -f "bin/conn[e]ctor" || true)

if [ -n "$PIDS" ]; then
    # SIGTERM does not get through: uvicorn is blocked inside the Ara model load and
    # never reaches its signal handler, so go straight to SIGKILL.
    echo "stopping connector: $PIDS"
    kill -9 $PIDS
    sleep 2
fi

[ "$1" = "--stop" ] && exit 0

# setsid and </dev/null so it outlives the ssh session that launched it.
setsid nohup ./venv/bin/connector --host 0.0.0.0 --port "$PORT" \
    > server.log 2>&1 < /dev/null &

echo "connector starting on :$PORT, logging to $(pwd)/server.log"
echo "watch it come up:   tail -f server.log"
echo "ready when it says: Uvicorn running on http://0.0.0.0:$PORT"
echo "then:               curl http://localhost:$PORT/v1/models"
