#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Aligner Trading — One-command AWS EC2 Deployment                           ║
# ║                                                                              ║
# ║  Run this ONCE on a fresh Ubuntu 22.04 EC2 instance:                        ║
# ║    curl -sSL https://raw.githubusercontent.com/.../aws_deploy.sh | bash     ║
# ║  OR copy this file to the server and run: bash aws_deploy.sh                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
set -euo pipefail

echo "=================================================="
echo "  Aligner Trading — AWS Deployment Setup"
echo "=================================================="

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    docker.io docker-compose-plugin \
    git curl ufw

# ── 2. Docker setup ────────────────────────────────────────────────────────────
echo "[2/6] Configuring Docker..."
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

# ── 3. Firewall (only open needed ports) ──────────────────────────────────────
echo "[3/6] Configuring firewall..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8510/tcp  # Trading dashboard
sudo ufw --force enable

# ── 4. Clone / copy project ───────────────────────────────────────────────────
echo "[4/6] Setting up project..."
APP_DIR="/opt/aligner"
if [ ! -d "$APP_DIR" ]; then
    sudo mkdir -p "$APP_DIR"
    sudo chown "$USER:$USER" "$APP_DIR"
    echo "  → Created $APP_DIR — copy your project files here"
    echo "    scp -r /path/to/Trading/* ubuntu@<EC2-IP>:$APP_DIR/"
fi

# ── 5. Systemd service (auto-start on reboot) ─────────────────────────────────
echo "[5/6] Installing systemd service..."
sudo tee /etc/systemd/system/aligner-trading.service > /dev/null <<EOF
[Unit]
Description=Aligner Trading System
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
ExecStartPre=/usr/bin/docker compose -f deploy/docker-compose.prod.yml pull --quiet
ExecStart=/usr/bin/docker compose -f deploy/docker-compose.prod.yml up --build
ExecStop=/usr/bin/docker compose -f deploy/docker-compose.prod.yml down
Restart=on-failure
RestartSec=30
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable aligner-trading

# ── 6. Get Elastic IP for Kite whitelist ──────────────────────────────────────
echo "[6/6] Network info..."
PUBLIC_IP=$(curl -s https://api.ipify.org 2>/dev/null || curl -s https://ifconfig.me)
echo ""
echo "=================================================="
echo "  SETUP COMPLETE"
echo "=================================================="
echo ""
echo "  Public IP (whitelist this on Kite developer console):"
echo "  → $PUBLIC_IP"
echo ""
echo "  Next steps:"
echo "  1. Copy your project: scp -r Trading/* ubuntu@$PUBLIC_IP:/opt/aligner/"
echo "  2. Copy your .env:    scp Trading/.env ubuntu@$PUBLIC_IP:/opt/aligner/"
echo "  3. Whitelist IP $PUBLIC_IP at https://developers.kite.trade"
echo "  4. Start trading:     sudo systemctl start aligner-trading"
echo "  5. Dashboard:         http://$PUBLIC_IP:8510/terminal"
echo ""
echo "  Service commands:"
echo "    sudo systemctl start   aligner-trading   # Start"
echo "    sudo systemctl stop    aligner-trading   # Stop"
echo "    sudo systemctl restart aligner-trading   # Restart"
echo "    sudo journalctl -fu    aligner-trading   # Logs"
echo ""
