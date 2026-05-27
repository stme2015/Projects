output "api_vm_public_ip" {
  description = "Public IP of the API VM - use this for curl commands"
  value       = google_compute_instance.api_vm.network_interface[0].access_config[0].nat_ip
}

output "inference_vm_internal_ip" {
  description = "Internal IP of inference VM"
  value       = google_compute_instance.inference_vm.network_interface[0].network_ip
}