#!/usr/bin/env bash
# Run one browser suite with its own preview server and preserve its exit status.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
control_test_port=${CONTROL_TEST_PORT:-8080}
control_server_log=${CONTROL_SERVER_LOG:-/tmp/control-server.log}
export CONTROL_TEST_URL="http://127.0.0.1:${control_test_port}/lab/"

python3 fractal-gas-web/tools/serve-control.py --port "$control_test_port" >> "$control_server_log" 2>&1 &
control_server_pid=$!
cleanup() {
  control_test_status=$?
  trap - EXIT
  kill "$control_server_pid" 2>/dev/null || true
  wait "$control_server_pid" 2>/dev/null || true
  if (( control_test_status != 0 )); then
    tail -100 "$control_server_log" >&2
  fi
  exit "$control_test_status"
}
trap cleanup EXIT
for attempt in {1..30}; do
  kill -0 "$control_server_pid"
  if curl --fail --silent "$CONTROL_TEST_URL" > /dev/null; then
    break
  fi
  sleep 1
done
kill -0 "$control_server_pid"
curl --fail --silent --show-error "$CONTROL_TEST_URL" > /dev/null
"$@"
