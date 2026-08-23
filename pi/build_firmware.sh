#!/usr/bin/env bash
# Build the GC9A01 init-sequence firmware blob for the panel-mipi-dbi driver
# and install it to /lib/firmware/gc9a01.bin. Run on the Pi (or anywhere with
# python3; copy the .bin over yourself in that case).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/gc9a01.txt"
BIN="$SCRIPT_DIR/gc9a01.bin"
TOOL="$SCRIPT_DIR/mipi-dbi-cmd"
TOOL_URL="https://raw.githubusercontent.com/notro/panel-mipi-dbi/main/mipi-dbi-cmd"
DEST="/lib/firmware/gc9a01.bin"

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required (sudo apt install python3)" >&2
    exit 1
fi

if [ ! -f "$SRC" ]; then
    echo "error: init source not found: $SRC" >&2
    exit 1
fi

# Fetch notro's mipi-dbi-cmd compiler if we don't have it yet.
if [ ! -f "$TOOL" ]; then
    echo "Downloading mipi-dbi-cmd from notro/panel-mipi-dbi ..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$TOOL_URL" -o "$TOOL"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$TOOL" "$TOOL_URL"
    else
        echo "error: need curl or wget to download $TOOL_URL" >&2
        exit 1
    fi
    chmod +x "$TOOL"
fi

echo "Compiling $SRC -> $BIN"
python3 "$TOOL" "$BIN" "$SRC"

if [ ! -s "$BIN" ]; then
    echo "error: $BIN was not produced or is empty" >&2
    exit 1
fi

echo "Round-trip decode of the blob (sanity check):"
python3 "$TOOL" "$BIN"

# Install (needs root for /lib/firmware).
if [ "$(id -u)" -eq 0 ]; then
    install -m 644 "$BIN" "$DEST"
else
    echo "Installing to $DEST (sudo) ..."
    sudo install -m 644 "$BIN" "$DEST"
fi
echo "Installed $DEST ($(stat -c %s "$DEST") bytes)"

cat <<'EOF'

Next steps:
  1. Append pi/config.txt.snippet to /boot/firmware/config.txt
     (and remove any plain `dtparam=spi=on` line — the overlays claim SPI0).
  2. sudo reboot
  3. Check the driver bound:  dmesg | grep -i mipi   → expect two panels;
     ls /dev/fb*              → expect /dev/fb1 and /dev/fb2.
  4. python3 pi/test_pattern.py   (red "L" bullseye left, blue "R" right)
EOF
