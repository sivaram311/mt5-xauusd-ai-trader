r"""
Self-Contained RunPod Orchestrator (Unicode-safe stdout handling):
1. Spins up an on-demand Secure Cloud GPU (RTX 3090 / A5000 / A4000).
2. Uploads bundle, trains model on GPU, exports ONNX.
3. Downloads models back to W:\data\trading\.
4. Guaranteed immediate pod termination.
"""
import os
import sys
import time
import runpod
import paramiko

runpod.api_key = os.environ.get("RUNPOD_API_KEY")
LOCAL_DIR = r"W:\data\trading"
DATASET_PATH = os.path.join(LOCAL_DIR, "xauusd_unified_patchtst_dataset.npz")
TRAINER_PATH = os.path.join(LOCAL_DIR, "train_runpod_patchtst.py")
ONNX_TARGET = os.path.join(LOCAL_DIR, "best_xauusd_unified_patchtst.onnx")
PT_TARGET = os.path.join(LOCAL_DIR, "best_xauusd_unified_patchtst.pt")

print("1. Generating temporary SSH Keypair for the pod...")
key = paramiko.RSAKey.generate(2048)
pub_key_str = f"ssh-rsa {key.get_base64()} runpod-auto"

gpu_candidates = ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA RTX A4000", "NVIDIA GeForce RTX 4090"]

pod_id = None
for gpu_id in gpu_candidates:
    try:
        print(f"2. Attempting to launch GPU pod ({gpu_id}, Secure/All Cloud)...")
        pod = runpod.create_pod(
            name="xauusd-mean-reversion-train",
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
        ports = runtime.get("ports", [])
        
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
        raise RuntimeError("Pod failed to expose SSH port in time.")

    print("4. Connecting via SSH...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    connected = False
    for attempt in range(25):
        try:
            time.sleep(4)
            ssh.connect(ssh_ip, port=ssh_port, username="root", pkey=key, timeout=15)
            connected = True
            print("   SSH connection established successfully.")
            break
        except Exception as e:
            print(f"   SSH retry {attempt+1}... ({e})")
    
    if not connected:
        raise RuntimeError("Could not establish SSH connection.")

    sftp = ssh.open_sftp()
    print("5. Uploading training script and dataset...")
    sftp.put(TRAINER_PATH, "/workspace/train_runpod_patchtst.py")
    sftp.put(DATASET_PATH, "/workspace/xauusd_unified_patchtst_dataset.npz")
    print("   Upload complete.")

    print("6. Executing GPU Training and ONNX Export...")
    cmd = "cd /workspace && python3 -m pip install 'numpy<2.0.0' 'onnx==1.16.1' -q && python3 train_runpod_patchtst.py"
    stdin, stdout, stderr = ssh.exec_command(cmd)

    for line in iter(stdout.readline, ""):
        clean_line = line.encode('ascii', errors='replace').decode('ascii').strip()
        if clean_line:
            print(f"   [Pod] {clean_line}")
    
    err_output = stderr.read().decode('ascii', errors='replace')
    if err_output and "Error" in err_output:
        print(f"   [Pod Stderr] {err_output}")

    print("7. Downloading trained weights & ONNX model back to local drive...")
    sftp.get("/workspace/best_xauusd_unified_patchtst.onnx", ONNX_TARGET)
    sftp.get("/workspace/best_xauusd_unified_patchtst.pt", PT_TARGET)
    print(f"   Retrieved ONNX: {ONNX_TARGET} ({os.path.getsize(ONNX_TARGET)/(1024*1024):.2f} MB)")
    print(f"   Retrieved PT:   {PT_TARGET} ({os.path.getsize(PT_TARGET)/(1024*1024):.2f} MB)")

finally:
    print("8. Terminating RunPod instance immediately to prevent idle charges...")
    if sftp:
        try: sftp.close()
        except: pass
    if ssh:
        try: ssh.close()
        except: pass
    if pod_id:
        try:
            runpod.terminate_pod(pod_id)
            print(f"   Pod {pod_id} successfully terminated. Zero lingering costs.")
        except Exception as e:
            print(f"   Error terminating pod {pod_id}: {e}")
