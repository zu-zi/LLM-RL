# utils/rollout_pool.py
import os
import json
import uuid
import glob
import hashlib
from typing import List, Dict, Optional

ROLL_GLOB = "roll_*.jsonl"
TMP_SUFFIX = ".tmp"

# ---------------------------
# 基础 IO
# ---------------------------

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _atomic_write(path: str, text: str) -> None:
    tmp = path + TMP_SUFFIX
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # 原子改名

def _read_lines(fp: str) -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        return f.readlines()

def _safe_parse(line: str) -> Optional[dict]:
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
        try:
            os.remove(fp)
        except FileNotFoundError:
            pass
        return
    if lines and (not lines[-1].endswith("\n")):
        lines = lines + ["\n"]
    _atomic_write(fp, "".join(lines))

def _clean_tmp_leftovers(dirpath: str) -> None:
    """清理异常残留的 .tmp 文件，避免估算/遍历干扰。"""
    for fp in glob.glob(os.path.join(dirpath, f"*{TMP_SUFFIX}")):
        try:
            os.remove(fp)
        except Exception:
            # 忽略清理失败
            pass

# ---------------------------
# 校验与去重
# ---------------------------

def _is_int_list(x) -> bool:
    if not isinstance(x, list):
        return False
    # 允许 int/long；保证都是整数
    return all(isinstance(t, int) for t in x)

def _hash_ids(ids: List[int]) -> str:
    # 用 sha1 对 id 列表做哈希，避免大对象占内存
    h = hashlib.sha1()
    h.update((",".join(map(str, ids))).encode("utf-8"))
    return h.hexdigest()

def _sanitize_item(it: dict) -> Optional[dict]:
    """
    过滤/修正不合格样本：
    - 必须有 prompt_ids/full_ids 且为 List[int]
    - full_ids 长度 > prompt_ids（保证有 response）
    返回清洗后的字典或 None（丢弃）
    """
    if not isinstance(it, dict):
        return None
    p = it.get("prompt_ids", None)
    f = it.get("full_ids", None)
    if (not _is_int_list(p)) or (not _is_int_list(f)):
        return None
    if len(f) <= len(p):
        # 没有 response，丢弃
        return None
    return {"prompt_ids": p, "full_ids": f}

# ---------------------------
# 写入（入队）
# ---------------------------

def enqueue_items(dirpath: str, items: List[Dict]) -> str:
    """
    写入一批 JSONL（原子落盘）
    items: [{'prompt_ids': [...], 'full_ids': [...]}]
    - 仅写入通过 _sanitize_item 的样本
    - 批内去重（按 full_ids 哈希）
    """
    ensure_dir(dirpath)
    _clean_tmp_leftovers(dirpath)

    fname = f"roll_{uuid.uuid4().hex}.jsonl"
    path = os.path.join(dirpath, fname)

    seen = set()
    buf = []
    for it in items:
        clean = _sanitize_item(it)
        if clean is None:
            continue
        h = _hash_ids(clean["full_ids"])
        if h in seen:
            continue
        seen.add(h)
        buf.append(json.dumps(clean, ensure_ascii=False))

    if not buf:
        # 空批没有意义，直接返回路径占位（但不落盘）
        return path

    _atomic_write(path, "\n".join(buf) + "\n")
    return path

# ---------------------------
# 读取（出队）
# ---------------------------

def dequeue_items(dirpath: str, n: int) -> List[Dict]:
    """
    从池中“弹出”n 条样本。会在文件内就地消费：
    - 若某文件只取走前 K 行，则余下行原子写回同名文件；
    - 若全部取走，文件被删除。
    - 仅返回通过 _sanitize_item 校验的样本；
    - 出队时也做去重（跨文件/跨行重复）。
    返回：list[dict]（长度 <= n）
    """
    ensure_dir(dirpath)
    _clean_tmp_leftovers(dirpath)

    want = max(int(n), 0)
    if want == 0:
        return []

    files = sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB)))
    out: List[Dict] = []
    seen = set()  # 按 full_ids 去重

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
            clean = _sanitize_item(obj) if obj is not None else None
            if clean is None:
                # 坏行丢弃，不放回
                continue

            h = _hash_ids(clean["full_ids"])
            if h in seen:
                # 去重命中：不放回、不纳入 out
                continue

            seen.add(h)
            out.append(clean)

        # 原子回写剩余（或删除）
        try:
            _atomic_rewrite(fp, keep)
        except Exception:
            # 若写回失败，尽量不阻塞主流程：尝试删除，失败则忽略
            try:
                os.remove(fp)
            except Exception:
                pass

        if len(out) >= want:
            break

    return out

# ---------------------------
# 估算
# ---------------------------

def estimate_size(dirpath: str, approx_per_file: int = 256, scan_cap_files: int = 100) -> int:
    """
    估算池中可用样本数量：
    - 优先按“行数”统计，最多扫描 scan_cap_files 个文件；
    - 用已扫描的平均行数 * 剩余文件数 做外推；
    - 若遇到 IO/权限问题则兜底为：文件数 * approx_per_file。
    - 仅对“可能合格”的行做保守估算（粗略过滤空行）。
    """
    try:
        _clean_tmp_leftovers(dirpath)
        files_all = sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB)))
        if not files_all:
            return 0

        files_scan = files_all[:max(int(scan_cap_files), 1)]
        total_lines = 0
        for fp in files_scan:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for ln in f:
                        if ln.strip():
                            total_lines += 1
            except Exception:
                # 出错就忽略该文件
                continue

        scanned = len(files_scan)
        if scanned == 0:
            # 扫描不到，回退
            return len(files_all) * max(int(approx_per_file), 1)

        avg = total_lines / scanned
        rest_files = max(len(files_all) - scanned, 0)
        # 外推 + 已扫描
        return int(total_lines + avg * rest_files)
    except Exception:
        # 兜底
        return len(glob.glob(os.path.join(dirpath, ROLL_GLOB))) * max(int(approx_per_file), 1)
