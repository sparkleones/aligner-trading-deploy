#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Aligner Trading — One-Command VPS Setup                                    ║
# ║                                                                              ║
# ║  Run on a fresh Ubuntu 22.04+ VPS:                                          ║
# ║    curl -sSL https://raw.githubusercontent.com/.../setup-vps.sh | bash      ║
# ║  Or copy this file to the server and run:                                    ║
# ║    chmod +x setup-vps.sh && ./setup-vps.sh                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -e

echo "╔══════════════════════════════════════════╗"
echo "║  Aligner Trading — VPS Setup             ║"
echo "╚══════════════════════════════════════════╝"

# ── 1. System Updates ──
echo "[1/6] Updating system..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# ── 2. Install Docker ──
echo "[2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in."
else
    echo "Docker already installed."
fi

# ── 3. Install Docker Compose ──
echo "[3/6] Installing Docker Compose..."
if ! command -v docker compose &> /dev/null; then
    sudo apt-get install -y docker-compose-plugin
fi

# ── 4. Set timezone to IST ──
echo "[4/6] Setting timezone to Asia/Kolkata..."
sudo timedatectl set-timezone Asia/Kolkata

# ── 5. Create project directory ──
echo "[5/6] Setting up project directory..."
PROJECT_DIR="$HOME/aligner-trading"
mkdir -p "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# ── 6. Firewall ──
echo "[6/6] Configuring firewall..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8501/tcp  # Dashboard (direct access)
sudo ufw --force enable

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Setup Complete!                          ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Copy your project to $PROJECT_DIR/"
echo "     scp -r /path/to/Trading/* $USER@this-server:$PROJECT_DIR/"
echo ""
echo "  2. Create .env file:"
echo "     cp $PROJECT_DIR/deploy/.env.example $PROJECT_DIR/.env"
echo "     nano $PROJECT_DIR/.env  # Fill in your Zerodha creds"
echo ""
echo "  3. Whitelist this server's IP in Kite Connect:"
echo "     Your IP: $(curl -s ifconfig.me)"
echo "     Go to: https://developers.kite.trade/apps"
echo ""
echo "  4. Start the system:"
echo "     cd $PROJECT_DIR"
echo "     docker compose -f deploy/docker-compose.prod.yml up -d"
echo ""
echo "  5. Access dashboard:"
echo "     http://$(curl -s ifconfig.me):8501/live"
echo ""
echo "  6. (Optional) Set up Cloudflare Tunnel for HTTPS:"
echo "     - Add CLOUDFLARE_TUNNEL_TOKEN to .env"
echo "     - docker compose -f deploy/docker-compose.prod.yml --profile tunnel up -d"
echo "     - Access via: https://trade.yourdomain.com/live"
