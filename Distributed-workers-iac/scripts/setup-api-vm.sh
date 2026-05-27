#!/bin/bash
# Run this on the API VM after provisioning
# Sets up iii engine + caller-worker

set -e

echo "=== Installing dependencies ==="
sudo apt-get update
sudo apt-get install -y jq git curl

echo "=== Installing Node.js 20 ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get remove -y libnode-dev libnode72 2>/dev/null || true
sudo apt-get install -y nodejs

echo "=== Installing iii ==="
curl -fsSL https://install.iii.dev/iii/main/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

echo "=== Cloning repo ==="
git clone https://github.com/Alchemyst-ai/hiring.git
cd hiring/may-2026/devops/quickstart

echo "=== Fixing config paths ==="
sed -i "s|/Users/anuran/Alchemyst/hiring|$HOME/hiring|g" config.yaml
sed -i 's|host: 127.0.0.1|host: 0.0.0.0|g' config.yaml
sed -i 's|default_timeout: 30000|default_timeout: 300000|g' config.yaml

echo "=== Installing caller-worker dependencies ==="
cd workers/caller-worker
npm install

echo "=== API VM setup complete ==="
echo "Start engine: cd ~/hiring/may-2026/devops/quickstart && iii --config config.yaml"
echo "Start caller-worker: cd ~/hiring/may-2026/devops/quickstart/workers/caller-worker && node --import tsx/esm src/worker.ts"
