# utils/rollout_pool.py
import os
import json
import uuid
import glob
from typing import List, Dict, Tuple

ROLL_GLOB = "roll_*.jsonl"
TMP_SUFFIX = ".tmp"

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _atomic_write(path: str, text: str) -> None:
    tmp = path + TMP_SUFFIX
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # 原子改名

def enqueue_items(dirpath: str, items: List[Dict]) -> str:
    """
    写入一批 JSONL（原子落盘）
    items: [{'prompt_ids': [...], 'full_ids': [...]}]
    """
    ensure_dir(dirpath)
    fname = f"roll_{uuid.uuid4().hex}.jsonl"
    path = os.path.join(dirpath, fname)
    buf = []
    for it in items:
        # 最小校验
        if not isinstance(it, dict) or "prompt_ids" not in it or "full_ids" not in it:
            continue
        buf.append(json.dumps(it, ensure_ascii=False))
    if not buf:
        # 空批也创建文件没有意义，直接返回路径占位（但不落盘）
        return path
    _atomic_write(path, "\n".join(buf) + "\n")
    return path

def _read_lines(fp: str) -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        return f.readlines()

def _safe_parse(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None

def _atomic_rewrite(fp: str, lines: List[str]) -> None:
    if not lines:
        # 没有剩余，直接删
        try: os.remove(fp)
        except FileNotFoundError: pass
        return
    _atomic_write(fp, "".join(lines if lines[-1].endswith("\n") else lines + ["\n"]))

def dequeue_items(dirpath: str, n: int) -> List[Dict]:
    """
    从池中“弹出”n 条样本。会在文件内就地消费：
    - 若某文件只取走前 K 行，则余下行原子写回同名文件；
    - 若全部取走，文件被删除。
    返回：list[dict]（长度 <= n）
    """
    ensure_dir(dirpath)
    want = max(int(n), 0)
    if want == 0:
        return []

    files = sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB)))
    out: List[Dict] = []
    for fp in files:
        if len(out) >= want:
            break
        try:
            lines = _read_lines(fp)
        except (FileNotFoundError, PermissionError):
            continue

        keep: List[str] = []
        for li, line in enumerate(lines):
            if len(out) >= want:
                # 后续行全部保留
                keep.extend(lines[li:])
                break
            obj = _safe_parse(line)
            if obj is None:
                # 坏行丢弃，不放回
                continue
            out.append(obj)

        # 原子回写剩余（或删除）
        try:
            _atomic_rewrite(fp, keep)
        except Exception:
            # 若写回失败，尽量不阻塞主流程：尝试删除，失败则忽略
            try: os.remove(fp)
            except Exception: pass

    return out

def estimate_size(dirpath: str, approx_per_file: int = 256, scan_cap_files: int = 100) -> int:
    """
    估算池中可用样本数量：
    - 优先按“行数”统计，最多扫描 scan_cap_files 个文件；
    - 若没有权限/IO 问题，则回退为：文件数 * approx_per_file。
    """
    try:
        files = sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB)))
        if not files:
            return 0
        # 只扫描较新的前 scan_cap_files 个文件，提高速度
        files = files[:scan_cap_files]
        total_lines = 0
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for _ in f:
                        total_lines += 1
            except Exception:
                # 出错就忽略该文件
                continue
        # 用已扫描的平均行数 * 剩余文件数 做外推
        avg = total_lines / max(len(files), 1)
        rest = max(len(glob.glob(os.path.join(dirpath, ROLL_GLOB))) - len(files), 0)
        return int(total_lines + avg * rest)
    except Exception:
        # 兜底
        return len(glob.glob(os.path.join(dirpath, ROLL_GLOB))) * max(approx_per_file, 1)
