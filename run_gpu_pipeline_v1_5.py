r"""
Self-Contained RunPod Orchestrator for 4-to-10 Strategy (v1.5 Series):
1. Allocates secure cloud GPU (RTX 3090 / A5000 / A4000 / RTX 4090).
2. Uploads dataset and trainer.
3. Saturates GPU compute with FP16 AMP.
4. Auto-exports best_order_net_v1_5_4to10.pt and .onnx.
5. Downloads back to W:\data\trading\.
6. Guaranteed immediate pod termination.
"""
import os
import sys
import time
import runpod
import paramiko

runpod.api_key = os.environ.get("RUNPOD_API_KEY")
LOCAL_DIR = r"W:\data\trading"
DATASET_PATH = os.path.join(LOCAL_DIR, "xauusd_m15_v1_5_4to10_dataset.npz")
TRAINER_PATH = os.path.join(LOCAL_DIR, "train_runpod_m15_v1_5_4to10.py")
ONNX_TARGET = os.path.join(LOCAL_DIR, "best_order_net_v1_5_4to10.onnx")
PT_TARGET = os.path.join(LOCAL_DIR, "best_order_net_v1_5_4to10.pt")

print("1. Generating temporary SSH Keypair for the pod...")
key = paramiko.RSAKey.generate(2048)
pub_key_str = f"ssh-rsa {key.get_base64()} runpod-auto"

gpu_candidates = ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA RTX A4000", "NVIDIA GeForce RTX 4090"]

pod_id = None
for gpu_id in gpu_candidates:
    try:
        print(f"2. Attempting to launch GPU pod ({gpu_id}, Secure/All Cloud)...")
        pod = runpod.create_pod(
            name="xauusd-4to10-strategy-train",
            image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            gpu_type_id=gpu_id,
            cloud_type="ALL",
            container_disk_in_gb=20,
            volume_in_gb=0,
            ports="22/tcp",
            env={"PUBLIC_KEY": pub_key_str}
        )
        if pod and pod.get("id"):
            pod_id = pod["id"]
            print(f"   Successfully provisioned {gpu_id} with ID: {pod_id}")
            break
    except Exception as e:
        print(f"   Candidate {gpu_id} skipped: {e}")

if not pod_id:
    print("Error: Could not allocate any candidate GPU pod.")
    sys.exit(1)

ssh = None
sftp = None

try:
    print("3. Waiting for pod to spin up and report public SSH port...")
    ssh_ip = None
    ssh_port = None

    for attempt in range(60):
        time.sleep(5)
        p_info = runpod.get_pod(pod_id) or {}
        runtime = p_info.get("runtime") or {}
        ports = runtime.get("ports") or []
        
        for p in ports:
            if p.get("privatePort") == 22 and p.get("ip") and p.get("publicPort"):
                ssh_ip = p.get("ip")
                ssh_port = p.get("publicPort")
                break
        
        if ssh_ip and ssh_port:
            print(f"   Pod running at {ssh_ip}:{ssh_port}")
            break
        print(f"   State: {p_info.get('desiredStatus', 'pending')} / runtime: {runtime.get('status', 'starting')}... ({attempt * 5}s)")

    if not ssh_ip or not ssh_port:
        raise RuntimeError("Pod timed out waiting for SSH port exposure.")

    print(f"4. Connecting via SSH to root@{ssh_ip}:{ssh_port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connected = False
    for attempt in range(20):
        try:
            time.sleep(3)
            ssh.connect(ssh_ip, port=int(ssh_port), username="root", pkey=key, timeout=15)
            connected = True
            print("   SSH connection established successfully.")
            break
        except Exception as e:
            print(f"   SSH retry {attempt+1}/20: {e}")

    if not connected:
        raise RuntimeError("Could not connect to SSH on pod.")

    sftp = ssh.open_sftp()
    print("5. Uploading dataset and GPU trainer script to pod...")
    sftp.put(DATASET_PATH, "/workspace/xauusd_m15_v1_5_4to10_dataset.npz")
    sftp.put(TRAINER_PATH, "/workspace/train.py")
    print("   Upload complete.")

    print("6. Installing required dependencies (numpy<2.0.0, onnx==1.16.1)...")
    stdin, stdout, stderr = ssh.exec_command("pip install 'numpy<2.0.0' 'onnx==1.16.1' -q")
    stdout.channel.recv_exit_status()

    print("7. Launching full-performance GPU training job (60 Epochs, FP16 AMP)...")
    stdin, stdout, stderr = ssh.exec_command("python -u /workspace/train.py")

    while True:
        line = stdout.readline()
        if not line:
            break
        safe_line = line.strip().encode('ascii', errors='replace').decode('ascii')
        print(f"   [POD] {safe_line}")
        sys.stdout.flush()

    status = stdout.channel.recv_exit_status()
    print(f"8. Remote training finished with exit code {status}.")

    print("9. Downloading trained weights (.pt) and (.onnx)...")
    sftp.get("/workspace/best_order_net_v1_5_4to10.pt", PT_TARGET)
    print(f"   Saved {PT_TARGET} ({os.path.getsize(PT_TARGET)/(1024*1024):.2f} MB)")

    sftp.get("/workspace/best_order_net_v1_5_4to10.onnx", ONNX_TARGET)
    print(f"   Saved {ONNX_TARGET} ({os.path.getsize(ONNX_TARGET)/(1024*1024):.2f} MB)")

except Exception as err:
    print(f"Pipeline error: {err}")
    sys.exit(1)

finally:
    if sftp:
        try: sftp.close()
        except: pass
    if ssh:
        try: ssh.close()
        except: pass
    if pod_id:
        print(f"10. Terminating RunPod pod {pod_id} immediately (Cost Hygiene)...")
        try:
            runpod.terminate_pod(pod_id)
            print("   Pod successfully terminated. Zero idle compute charged.")
        except Exception as e:
            print(f"   Failed to terminate pod: {e}")
