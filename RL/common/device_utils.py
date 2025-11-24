# RL/common/device_utils.py
import torch

def cuda_free_mb(device_str="cuda"):
    try:
        if not torch.cuda.is_available():
            return 0
        free, _ = torch.cuda.mem_get_info(torch.device(device_str))
        return int(free // (1024 * 1024))
    except Exception:
        return 0
