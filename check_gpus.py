import runpod
import os
import json

runpod.api_key = os.environ.get("RUNPOD_API_KEY")

gpus = runpod.get_gpus()
print(f"Total GPU types: {len(gpus)}")
candidates = []
for g in gpus:
    gpu_id = g.get("id")
    c_price = g.get("communityPrice")
    s_price = g.get("lowestPrice", {}).get("uninterruptablePrice")
    candidates.append({"id": gpu_id, "comm": c_price, "sec": s_price})

# Sort by community price if available
valid = [c for c in candidates if c["comm"] is not None]
valid.sort(key=lambda x: x["comm"])

for v in valid[:10]:
    print(f"{v['id']:30} | Comm: ${v['comm']}/hr | Sec: ${v['sec']}/hr")
