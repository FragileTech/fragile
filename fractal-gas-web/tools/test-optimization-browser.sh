#!/usr/bin/env bash
# Run both published URL layouts with a private server and virtual display.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
optimization_test_port=${OPTIMIZATION_TEST_PORT:-8081}
optimization_server_log=${OPTIMIZATION_SERVER_LOG:-/tmp/optimization-server.log}
optimization_base="http://127.0.0.1:${optimization_test_port}"

python3 fractal-gas-web/tools/serve-control.py --port "$optimization_test_port" > "$optimization_server_log" 2>&1 &
optimization_server_pid=$!
cleanup() {
  optimization_test_status=$?
  trap - EXIT
  kill "$optimization_server_pid" 2>/dev/null || true
  wait "$optimization_server_pid" 2>/dev/null || true
  if (( optimization_test_status != 0 )); then
    tail -100 "$optimization_server_log" >&2
  fi
  exit "$optimization_test_status"
}
trap cleanup EXIT
for attempt in {1..30}; do
  kill -0 "$optimization_server_pid"
  if curl --fail --silent "$optimization_base/optimization/" > /dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error "$optimization_base/optimization/" > /dev/null
for optimization_path in /optimization/ /fragile/optimization/; do
  OPTIMIZATION_TEST_URL="$optimization_base$optimization_path" \
    xvfb-run -a npm --prefix fractal-gas-web run test:optimization-browser
done
