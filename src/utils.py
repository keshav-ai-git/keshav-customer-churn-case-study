import hashlib, json, os
from datetime import datetime

def run_hash(params: dict) -> str:
    m = hashlib.md5(json.dumps(params, sort_keys=True).encode())
    return m.hexdigest()[:10]

def now_ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
