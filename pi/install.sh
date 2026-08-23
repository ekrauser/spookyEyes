#!/usr/bin/env bash
# Idempotent provisioning for spookyeyes ON the Raspberry Pi.
# Run as your normal user (sudo is used where needed), from anywhere:
#   bash ~/spookyEyes/pi/install.sh
# Safe to re-run after every rsync of the repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_USER="${USER:-$(id -un)}"
VENV="$HOME/spookyeyes-venv"
UNIT_SRC="$SCRIPT_DIR/spookyeyes.service"
UNIT_DST="/etc/systemd/system/spookyeyes.service"

if [ "$(id -u)" -eq 0 ]; then
    echo "error: run as your normal user, not root (sudo is invoked as needed)" >&2
    exit 1
fi

echo "==> repo: $REPO  user: $RUN_USER  venv: $VENV"

echo "==> apt packages"
sudo apt-get update -qq
sudo apt-get install -y python3-venv git

echo "==> virtualenv"
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip

echo "==> install spookyeyes (editable, with mqtt+pir extras)"
"$VENV/bin/pip" install --quiet -e "$REPO[mqtt,pir]"

if [ ! -f "$REPO/config.toml" ]; then
    echo "==> creating config.toml from config.example.toml (edit it!)"
    cp "$REPO/config.example.toml" "$REPO/config.toml"
    sed -i 's/^output = "preview"/output = "fb"/' "$REPO/config.toml"
else
    echo "==> config.toml exists, leaving it alone"
fi

echo "==> systemd unit -> $UNIT_DST"
sed -e "s|__USER__|$RUN_USER|g" \
    -e "s|__REPO__|$REPO|g" \
    -e "s|__VENV__|$VENV|g" \
    "$UNIT_SRC" | sudo tee "$UNIT_DST" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable spookyeyes.service

# /dev/fb* access without root
if ! id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx video; then
    echo "==> adding $RUN_USER to the video group (takes effect on next login)"
    sudo usermod -aG video "$RUN_USER"
fi

cat <<EOF

Done. Next:
  1. Edit $REPO/config.toml (mqtt/pir enable, broker credentials, theme).
  2. Make sure the displays work first:  $VENV/bin/python $SCRIPT_DIR/test_pattern.py
  3. sudo systemctl start spookyeyes
  4. journalctl -u spookyeyes -f          # watch logs / measured FPS
EOF
