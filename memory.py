import json
import os
from config import MEMORY_FILE

def load_memory_file() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_memory_file(mem: dict) -> None:
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MEMORY_FILE)

def append_memory(mem: dict, agent: str, role: str, text: str) -> None:
    if agent not in mem:
        mem[agent] = []
    mem[agent].append({"role": role, "text": text})
    save_memory_file(mem)
