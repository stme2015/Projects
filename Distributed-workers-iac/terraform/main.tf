terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ── NETWORK ──────────────────────────────────────────────
resource "google_compute_network" "iii_network" {
  name                    = "iii-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "iii_subnet" {
  name          = "iii-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.iii_network.id
}

# ── FIREWALL RULES ────────────────────────────────────────
resource "google_compute_firewall" "allow_ssh" {
  name    = "iii-allow-ssh"
  network = google_compute_network.iii_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "allow_http" {
  name    = "iii-allow-http"
  network = google_compute_network.iii_network.name

  allow {
    protocol = "tcp"
    ports    = ["3111"]
  }
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "allow_internal" {
  name    = "iii-allow-internal"
  network = google_compute_network.iii_network.name

  allow {
    protocol = "all"
  }
  source_ranges = [var.subnet_cidr]
}

# ── CLOUD NAT (so inference-vm can reach internet) ────────
resource "google_compute_router" "iii_router" {
  name    = "iii-router"
  region  = var.region
  network = google_compute_network.iii_network.id
}

resource "google_compute_router_nat" "iii_nat" {
  name                               = "iii-nat"
  router                             = google_compute_router.iii_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# ── API VM (public facing) ────────────────────────────────
resource "google_compute_instance" "api_vm" {
  name         = "api-vm"
  machine_type = "e2-medium"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.iii_subnet.id
    access_config {}  # gives public IP
  }

  tags = ["iii-api"]
}

# ── INFERENCE VM (private, no public IP) ──────────────────
resource "google_compute_instance" "inference_vm" {
  name         = "inference-vm"
  machine_type = "e2-highmem-4"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.iii_subnet.id
    # no access_config = no public IP
  }

  tags = ["iii-inference"]
}