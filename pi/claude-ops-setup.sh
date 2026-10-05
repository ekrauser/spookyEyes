#!/usr/bin/env bash
# Create the scoped operator account `claude-ops` on the Pi.
#
# Run as root on the Pi, from the owner's login:
#   ssh -t ladmin@spooky-pi1 "sudo bash -s -- '$(cat ~/.ssh/<key>.pub)'" < pi/claude-ops-setup.sh
#
# The account can: read the service journal, pull the repo and reinstall the
# venv package as the run user, read/replace config.toml, and start/stop/
# restart the spookyeyes service. Nothing else. The owner's own login stays
# the primary account; this one is removable with `userdel -r claude-ops`.
set -euo pipefail

PUBKEY="${1:?usage: claude-ops-setup.sh '<ssh public key>' [run_user]}"
RUN_USER="${2:-${SUDO_USER:-ladmin}}"
OPS_USER="claude-ops"
REPO="/home/$RUN_USER/spookyEyes"
VENV="/home/$RUN_USER/spookyeyes-venv"

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)"; exit 1; }
id "$RUN_USER" >/dev/null 2>&1 || { echo "run user $RUN_USER does not exist"; exit 1; }
[ -d "$REPO" ] || { echo "repo $REPO not found"; exit 1; }

echo "==> account $OPS_USER (run user: $RUN_USER, repo: $REPO)"
if ! id "$OPS_USER" >/dev/null 2>&1; then
    useradd -m -s /bin/bash "$OPS_USER"
fi
# journal + log reading without sudo
usermod -aG systemd-journal,adm "$OPS_USER"

install -d -m 700 -o "$OPS_USER" -g "$OPS_USER" "/home/$OPS_USER/.ssh"
AUTH="/home/$OPS_USER/.ssh/authorized_keys"
touch "$AUTH"
grep -qxF "$PUBKEY" "$AUTH" || echo "$PUBKEY" >> "$AUTH"
chown "$OPS_USER:$OPS_USER" "$AUTH"
chmod 600 "$AUTH"

SUDOERS="/etc/sudoers.d/$OPS_USER"
cat > "$SUDOERS.tmp" <<SUDO
# Scoped operator access for $OPS_USER (spookyeyes). Managed by pi/claude-ops-setup.sh.
Cmnd_Alias SPOOKY_SVC = /usr/bin/systemctl start spookyeyes, \\
                        /usr/bin/systemctl stop spookyeyes, \\
                        /usr/bin/systemctl restart spookyeyes, \\
                        /usr/bin/systemctl daemon-reload
Cmnd_Alias SPOOKY_REPO = /usr/bin/git -C $REPO *, \\
                         $VENV/bin/pip install -e $REPO, \\
                         $VENV/bin/pip install -e $REPO[*], \\
                         /usr/bin/cat $REPO/config.toml, \\
                         /usr/bin/tee $REPO/config.toml, \\
                         /usr/bin/cp $REPO/config.toml $REPO/config.toml.bak
$OPS_USER ALL=(root) NOPASSWD: SPOOKY_SVC
$OPS_USER ALL=($RUN_USER) NOPASSWD: SPOOKY_REPO
SUDO
visudo -cf "$SUDOERS.tmp"
install -m 440 "$SUDOERS.tmp" "$SUDOERS"
rm -f "$SUDOERS.tmp"

echo "==> done. From the operator side:"
echo "    ssh $OPS_USER@$(hostname) 'sudo -u $RUN_USER git -C $REPO pull && sudo systemctl restart spookyeyes'"
echo "    sudo -l as $OPS_USER:"
sudo -l -U "$OPS_USER"
