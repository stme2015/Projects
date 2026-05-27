#!/bin/bash
# Run this on the inference VM after provisioning
# Sets up inference-worker with Python + Gemma model

set -e

# Pass the API VM internal IP as argument
# Usage: ./setup-inference-vm.sh 10.0.0.2
API_VM_IP=${1:-"10.0.0.2"}

echo "=== Installing dependencies ==="
sudo apt-get update
sudo apt-get install -y jq git curl python3 python3-venv

echo "=== Installing iii ==="
curl -fsSL https://install.iii.dev/iii/main/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

echo "=== Cloning repo ==="
git clone https://github.com/Alchemyst-ai/hiring.git
cd hiring/may-2026/devops/quickstart

echo "=== Installing Python dependencies ==="
pip3 install iii-sdk==0.11.0 watchfiles transformers torch gguf accelerate jinja2

echo "=== Fixing gguf version bug ==="
echo "__version__ = '0.19.0'" >> ~/.local/lib/python3.10/site-packages/gguf/__init__.py

echo "=== Fixing inference worker ==="
# Reduce tokens for faster response
sed -i 's|max_new_tokens=32000|max_new_tokens=50|g' workers/inference-worker/inference_worker.py
# Fix return format
sed -i 's|return result|return {"response": result}|g' workers/inference-worker/inference_worker.py

echo "=== Setting engine URL ==="
echo "export III_URL=ws://${API_VM_IP}:49134" >> ~/.bashrc

echo "=== Inference VM setup complete ==="
echo "Start worker: cd ~/hiring/may-2026/devops/quickstart && export III_URL=ws://${API_VM_IP}:49134 && python3 workers/inference-worker/inference_worker.py"
