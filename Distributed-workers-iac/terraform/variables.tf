variable "project_id" {
  description = "GCP Project ID"
  default     = "devops-ai-inference"
}

variable "region" {
  description = "GCP Region"
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  default     = "us-central1-a"
}

variable "subnet_cidr" {
  description = "Private subnet CIDR"
  default     = "10.0.0.0/24"
}