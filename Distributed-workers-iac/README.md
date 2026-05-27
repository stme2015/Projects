# GCP Distributed Workers IaC

Infrastructure and setup notes for a distributed AI inference system on private
GCP subnets using Terraform. The stack exposes a Gemma 3 270M model through a
JSON HTTP API backed by an `iii` RPC worker mesh.

## Architecture

```
Internet
  |
  | HTTP POST :3111
  v
+-----------------------------------------------------------+
| VPC: iii-network                                          |
| Subnet: 10.0.0.0/24                                       |
|                                                           |
|  +------------------------+                               |
|  | api-vm                 |                               |
|  | 10.0.0.2               |                               |
|  | - iii engine           |                               |
|  | - caller-worker        |                               |
|  +-----------+------------+                               |
|              ^ inference-worker connects to engine        |
|              | ws://10.0.0.2:49134 (engine on api-vm)    |
|  +------------------------+                               |
|  | inference-vm           |                               |
|  | 10.0.0.3               |                               |
|  | no public IP           |                               |
|  | - inference-worker     |                               |
|  | - Gemma 3 270M model   |                               |
|  +------------------------+                               |
+-----------------------------------------------------------+
```

## Request Flow

1. A client sends an HTTP `POST` request to the public IP of `api-vm`.
2. `caller-worker` receives the request and forwards it to the `iii` engine.
3. The engine routes the job over RPC to `inference-worker` on `inference-vm`.
4. `inference-worker` runs the Gemma model and generates a response.
5. The response returns through the same chain back to the client.

## Stack

- **Cloud:** GCP (`us-central1`)
- **IaC:** Terraform
- **Engine:** `iii` `v0.13.0`
- **Workers:** Python (inference) and TypeScript (caller)
- **Model:** Gemma 3 270M (GGUF quantized)
- **Networking:** Private subnet, Cloud NAT, and no public IP on `inference-vm`

## Prerequisites

- A GCP account with billing enabled
- Terraform installed locally
- `gcloud` CLI installed and authenticated

## Included Scripts

- `scripts/setup-api-vm.sh`: installs `iii`, Node.js, and caller-worker
  dependencies on `api-vm`
- `scripts/setup-inference-vm.sh`: installs Python dependencies and configures
  the inference worker on `inference-vm`
- `scripts/caller-worker.service`: `systemd` unit for the caller worker
- `scripts/inference-worker.service`: `systemd` unit for the inference worker

## Deploy From Scratch

### 1. Authenticate with GCP

```bash
gcloud auth login
gcloud auth application-default login
```

### 2. Provision Infrastructure with Terraform

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

Save the Terraform outputs for later:

```text
api_vm_public_ip = "XX.XX.XX.XX"
inference_vm_internal_ip = "10.0.0.X"
```

### 3. Copy Setup Files to the VMs

From your local checkout:

```bash
gcloud compute scp scripts/setup-api-vm.sh scripts/caller-worker.service \
  api-vm:~/ --zone=us-central1-a

gcloud compute scp --tunnel-through-iap \
  scripts/setup-inference-vm.sh scripts/inference-worker.service \
  inference-vm:~/ --zone=us-central1-a
```

### 4. Set Up the API VM

```bash
gcloud compute ssh api-vm --zone=us-central1-a
chmod +x ~/setup-api-vm.sh
~/setup-api-vm.sh
```

### 5. Set Up the Inference VM

Replace `<api_vm_internal_ip>` with the Terraform output from step 2.

```bash
gcloud compute ssh inference-vm --zone=us-central1-a --tunnel-through-iap
chmod +x ~/setup-inference-vm.sh
~/setup-inference-vm.sh <api_vm_internal_ip>
```

### 6. Install the `systemd` Units

Before installing the services, update the service files if needed:

- Set `User=` to the Linux username on your VM
- Set `III_URL=` to the correct API VM address

On `api-vm`:

```bash
sudo cp ~/caller-worker.service /etc/systemd/system/caller-worker.service
sudo systemctl daemon-reload
sudo systemctl enable --now caller-worker
```

On `inference-vm`:

```bash
sudo cp ~/inference-worker.service /etc/systemd/system/inference-worker.service
sudo systemctl daemon-reload
sudo systemctl enable --now inference-worker
```

### 7. Start the Engine

On `api-vm`:

```bash
cd ~/hiring/may-2026/devops/quickstart
iii --config config.yaml
```

## API Usage

### Endpoint

- Method: `POST`
- URL: `http://<api_vm_public_ip>:3111/v1/chat/completions`
- Header: `Content-Type: application/json`

### Request Body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is Kubernetes?"
    }
  ]
}
```

### Example Request

```bash
curl -X POST http://34.171.205.77:3111/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is Kubernetes?"}]}'
```

### Sample Response

```json
{
  "result": {
    "response": "Kubernetes is an open-source container orchestration platform...",
    "success": "You've connected two workers and they're interoperating seamlessly"
  }
}
```

## Production Hardening

If this were being prepared for production, the next improvements would be:

1. **TLS termination:** Put Nginx or Caddy in front of the engine so traffic is
   encrypted with HTTPS.
2. **Authentication:** Add API key or JWT validation in `caller-worker` before
   forwarding requests to the engine.
3. **Secrets management:** Store credentials in GCP Secret Manager instead of
   environment variables.
4. **Health checks and monitoring:** Add a `/health` endpoint and configure
   Cloud Monitoring alerts.
5. **Firewall hardening:** Restrict SSH access to trusted IP ranges instead of
   allowing broad access.
6. **Service reliability:** Keep `Restart=on-failure` and add memory limits,
   logging, and rotation policies.

## Scaling for a 100x Larger Model

If the model were about 100x larger, such as 27B parameters instead of 270M:

1. **GPU instances:** Move `inference-vm` to a GPU-backed machine because CPU
   inference would be too slow.
2. **Model sharding:** Split the model across multiple GPUs or VMs if it no
   longer fits on a single device.
3. **Async inference:** Queue requests and let clients poll for results instead
   of waiting synchronously.
4. **Autoscaling:** Use Managed Instance Groups to scale inference capacity up
   and down with demand.
5. **Specialized serving:** Replace raw `transformers` inference with a serving
   stack such as vLLM or TGI for batching and cache efficiency.
