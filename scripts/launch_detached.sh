#!/usr/bin/env bash
# Launch a long-running job so it survives the Claude Code session ending.
#
# Jobs started as harness background tasks are children of the session, and
# are killed with it -- a multi-turn GRPO run died at update 10/60 when the
# session restarted after an upgrade. `caffeinate` prevents system sleep but
# does nothing against that. This double-forks and calls setsid, so the job is
# reparented to launchd in its own session and process group, out of reach of
# the session's teardown. The caller returns immediately.
#
#   usage:  scripts/launch_detached.sh <name> <command...>
#   log:    logs/runs/<name>.log      pid:  logs/runs/<name>.pid
#   check:  ps -p "$(cat logs/runs/<name>.pid)"
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
name="${1:?usage: launch_detached.sh <name> <command...>}"; shift
mkdir -p "$root/logs/runs"
log="$root/logs/runs/$name.log"
pidf="$root/logs/runs/$name.pid"
if [ -f "$pidf" ] && ps -p "$(cat "$pidf")" >/dev/null 2>&1; then
  echo "refusing: '$name' is already running as pid $(cat "$pidf")" >&2
  exit 1
fi
rm -f "$pidf"
cd "$root"
"$root/.venv-qwen/bin/python" - "$log" "$pidf" "$@" <<'PY'
import os, sys
log, pidf, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
if os.fork() > 0:
    sys.exit(0)                     # the caller gets control back at once
os.setsid()                         # new session: no longer the harness's child
if os.fork() > 0:
    sys.exit(0)                     # drop session leadership; can never reacquire a tty
with open(pidf, 'w') as f:
    f.write(str(os.getpid()))
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(cmd[0], cmd)
PY
for _ in 1 2 3 4 5; do [ -s "$pidf" ] && break; sleep 1; done
echo "launched '$name'  pid $(cat "$pidf")  log ${log#$root/}"
