# utils/rollout_pool.py
import os
import json
import uuid
import glob
import time
import random
import hashlib
from typing import List, Dict, Optional, Tuple

ROLL_GLOB = "roll_*.jsonl"
TMP_SUFFIX = ".tmp"

# =========================
# 环境变量 & 策略默认值
# =========================

def _get_env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, "").strip())
        return v
    except Exception:
        return default

def _get_env_float(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, "").strip())
        return v
    except Exception:
        return default

def _get_env_bool(name: str, default: bool) -> bool:
    s = os.getenv(name, "").strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default

ROLL_MAX_AGE_SEC          = _get_env_int("ROLL_MAX_AGE_SEC", 6 * 3600)  # 样本最大年龄（秒），默认 6h
ROLL_MIN_RESP_TOKENS      = _get_env_int("ROLL_MIN_RESP_TOKENS", 8)     # 回复最少 token 数（过滤太短回复）
ROLL_DEDUP_PROMPT_IN_BATCH= _get_env_bool("ROLL_DEDUP_PROMPT_IN_BATCH", True)  # 批内按 prompt 去重
ROLL_VERBOSE              = _get_env_bool("ROLL_VERBOSE", False)         # 关键日志
ROLL_FILE_ORDER           = os.getenv("ROLL_FILE_ORDER", "mtime_desc")   # mtime_desc|random|name_asc
ROLL_SCAN_CAP_FILES       = _get_env_int("ROLL_SCAN_CAP_FILES", 100)     # 估算时最多扫描多少文件
ROLL_APPROX_PER_FILE      = _get_env_int("ROLL_APPROX_PER_FILE", 256)    # 扫描失败时的兜底行数估计


# =========================
# 基础 IO
# =========================

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _atomic_write(path: str, text: str) -> None:
    tmp = path + TMP_SUFFIX
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

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
        try:
            os.remove(fp)
        except FileNotFoundError:
            pass
        return
    if lines and (not lines[-1].endswith("\n")):
        lines = lines + ["\n"]
    _atomic_write(fp, "".join(lines))

def _clean_tmp_leftovers(dirpath: str) -> None:
    for fp in glob.glob(os.path.join(dirpath, f"*{TMP_SUFFIX}")):
        try:
            os.remove(fp)
        except Exception:
            pass

def _now_ts() -> float:
    return time.time()


# =========================
# 校验与哈希
# =========================

def _is_int_list(x) -> bool:
    if not isinstance(x, list):
        return False
    return all(isinstance(t, int) for t in x)

def _hash_ids(ids: List[int]) -> str:
    h = hashlib.sha1()
    h.update((",".join(map(str, ids))).encode("utf-8"))
    return h.hexdigest()

def _sanitize_item(it: dict) -> Optional[dict]:
    """
    清洗输入样本：
    - 必须有 prompt_ids/full_ids 且为 List[int]
    - full_ids 长度 > prompt_ids（保证有 response）
    - 填充 ts/hid/pid 等元数据
    """
    if not isinstance(it, dict):
        return None
    p = it.get("prompt_ids", None)
    f = it.get("full_ids", None)
    if (not _is_int_list(p)) or (not _is_int_list(f)):
        return None
    if len(f) <= len(p):
        return None

    # 元信息
    ts = it.get("ts", None)
    if not isinstance(ts, (int, float)):
        ts = _now_ts()
    pid = it.get("pid") or _hash_ids(p)
    hid = it.get("hid") or _hash_ids(f)

    clean = {
        "prompt_ids": p,
        "full_ids": f,
        "ts": float(ts),
        "pid": pid,
        "hid": hid,
    }
    # 兼容扩展字段透传（如果你后续添加更多字段）
    for k in ("prompt_text", "response_text"):
        if k in it:
            clean[k] = it[k]
    return clean


# =========================
# 写入（入队）
# =========================

def enqueue_items(dirpath: str, items: List[Dict]) -> str:
    """
    写入一批 JSONL（原子落盘）
    items: [{'prompt_ids': [...], 'full_ids': [...], ...}]
    - 仅写入通过 _sanitize_item 的样本
    - 批内按 hid 去重
    - 自动写入 ts/pid/hid
    """
    ensure_dir(dirpath)
    _clean_tmp_leftovers(dirpath)

    fname = f"roll_{uuid.uuid4().hex}.jsonl"
    path = os.path.join(dirpath, fname)

    seen = set()
    buf = []
    kept, dropped = 0, 0

    for it in items:
        clean = _sanitize_item(it)
        if clean is None:
            dropped += 1
            continue
        h = clean["hid"]
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        buf.append(json.dumps(clean, ensure_ascii=False))
        kept += 1

    if not buf:
        return path  # 返回占位路径，但不会落盘

    _atomic_write(path, "\n".join(buf) + "\n")

    if ROLL_VERBOSE:
        print(f"[rollout_pool.enqueue] file={os.path.basename(path)} kept={kept} dropped={dropped}")

    return path


# =========================
# 读取（出队）
# =========================

def _file_order(files: List[str]) -> List[str]:
    if not files:
        return files
    mode = ROLL_FILE_ORDER
    if mode == "random":
        random.shuffle(files)
        return files
    if mode == "name_asc":
        return sorted(files)  # 字典序
    # 默认：按 mtime 逆序（新文件优先）
    files_sorted = sorted(files, key=lambda fp: os.path.getmtime(fp), reverse=True)
    return files_sorted

def _is_fresh(obj: dict, now_ts: float) -> bool:
    if ROLL_MAX_AGE_SEC <= 0:
        return True
    ts = obj.get("ts")
    if not isinstance(ts, (int, float)):
        return True  # 没有 ts 的旧数据先放过（你也可以选择丢弃）
    age = max(0.0, now_ts - float(ts))
    return age <= float(ROLL_MAX_AGE_SEC)

def _resp_len_ok(obj: dict) -> bool:
    p = obj.get("prompt_ids", [])
    f = obj.get("full_ids", [])
    if not (_is_int_list(p) and _is_int_list(f)):
        return False
    resp_len = len(f) - len(p)
    return resp_len >= max(0, int(ROLL_MIN_RESP_TOKENS))

def _stringify(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)

def dequeue_items(dirpath: str, n: int) -> List[Dict]:
    """
    从池中“弹出”n 条样本（优先新鲜）：
    - 文件顺序：默认按 mtime 逆序（新 -> 旧），可用 env 调整
    - 行过滤：TTL（max_age_sec）、最小回复长度
    - 批内去重：按 hid；可选地按 pid（去重复 prompt）
    - 文件内就地消费：取走行不再放回；淘汰行直接丢弃；剩余行原子回写
    返回：list[dict]，长度 <= n
    """
    ensure_dir(dirpath)
    _clean_tmp_leftovers(dirpath)

    want = max(int(n), 0)
    if want == 0:
        return []

    files = _file_order(sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB))))
    out: List[Dict] = []

    seen_hid = set()
    seen_pid = set()
    now_ts = _now_ts()

    # 统计
    cnt_scanned = 0
    cnt_taken = 0
    cnt_dropped_stale = 0
    cnt_dropped_short = 0
    cnt_dropped_dup_hid = 0
    cnt_dropped_dup_pid = 0

    for fp in files:
        if len(out) >= want:
            break

        try:
            lines = _read_lines(fp)
        except (FileNotFoundError, PermissionError):
            continue

        keep_lines: List[str] = []

        # 优先在文件内也“新鲜优先”：把行按 ts 逆序（无 ts 的排后）
        def _ts_from_line(ln: str) -> float:
            obj = _safe_parse(ln)
            if isinstance(obj, dict) and isinstance(obj.get("ts"), (int, float)):
                return float(obj["ts"])
            return 0.0

        # 在文件内部也做一次排序，有利于先消费新样本
        # 注意：这是在内存里重排，不改变原文件顺序；我们只在回写时去掉已消费/淘汰的行
        lines_sorted = sorted(lines, key=_ts_from_line, reverse=True)

        for ln in lines_sorted:
            obj = _safe_parse(ln)
            cnt_scanned += 1

            if obj is None:
                # 坏行：直接不放回
                continue

            clean = _sanitize_item(obj)
            if clean is None:
                # 不合格：直接不放回
                continue

            # 过滤旧样本
            if not _is_fresh(clean, now_ts):
                cnt_dropped_stale += 1
                continue

            # 回复太短过滤
            if not _resp_len_ok(clean):
                cnt_dropped_short += 1
                continue

            # 批内去重（hid）
            hid = clean["hid"]
            if hid in seen_hid:
                cnt_dropped_dup_hid += 1
                continue

            # 批内按 prompt 去重（可选）
            if ROLL_DEDUP_PROMPT_IN_BATCH:
                pid = clean["pid"]
                if pid in seen_pid:
                    cnt_dropped_dup_pid += 1
                    continue

            # 选中：加入 out
            out.append(clean)
            seen_hid.add(hid)
            if ROLL_DEDUP_PROMPT_IN_BATCH:
                seen_pid.add(clean["pid"])
            cnt_taken += 1

            if len(out) >= want:
                # 剩余行全部保留（但要过滤坏行/过期行）
                break

        # 文件回写：仅保留“未被选中且仍然合格的行”
        # 重新从原 lines 过滤（保证未排序的原始顺序被保留给后续消费）
        for ln in lines:
            obj = _safe_parse(ln)
            if obj is None:
                continue
            clean = _sanitize_item(obj)
            if clean is None:
                continue
            if not _is_fresh(clean, now_ts):
                continue
            if not _resp_len_ok(clean):
                continue
            # 若该行已被本次取走（按 hid 判断），则不保留
            if clean["hid"] in seen_hid:
                continue
            # 若批内按 prompt 去重已命中，但未来批次仍可消费，所以保留
            keep_lines.append(_stringify(clean) + "\n")

        try:
            _atomic_rewrite(fp, keep_lines)
        except Exception:
            # 写回失败，尽量不阻塞主流程：尝试删除
            try:
                os.remove(fp)
            except Exception:
                pass

    if ROLL_VERBOSE:
        # 简要观测日志
        print(
            "[rollout_pool.dequeue] "
            f"want={want} got={len(out)} "
            f"scanned={cnt_scanned} taken={cnt_taken} "
            f"drop(stale)={cnt_dropped_stale} "
            f"drop(short)={cnt_dropped_short} "
            f"drop(dup_hid)={cnt_dropped_dup_hid} "
            f"drop(dup_pid)={cnt_dropped_dup_pid} "
            f"ttl={ROLL_MAX_AGE_SEC}s min_resp={ROLL_MIN_RESP_TOKENS}"
        )

    return out


# =========================
# 估算 & 统计
# =========================

def estimate_size(dirpath: str, approx_per_file: int = None, scan_cap_files: int = None) -> int:
    """
    估算池中“可能可用”的样本数量（粗略）：
    - 优先扫描一定数量的文件（默认 ROLL_SCAN_CAP_FILES）并数非空行
    - 忽略明显坏行；对有 ts 的样本，**超过 TTL 的不计入**
    - 扫描集得到平均行数 -> 外推剩余文件
    - 兜底：文件数 * approx_per_file
    """
    try:
        _clean_tmp_leftovers(dirpath)
        files_all = sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB)))
        if not files_all:
            return 0

        cap = scan_cap_files if isinstance(scan_cap_files, int) and scan_cap_files > 0 else ROLL_SCAN_CAP_FILES
        approx = approx_per_file if isinstance(approx_per_file, int) and approx_per_file > 0 else ROLL_APPROX_PER_FILE

        files_scan = files_all[:max(int(cap), 1)]
        total_kept = 0
        now_ts = _now_ts()

        for fp in files_scan:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        obj = _safe_parse(ln)
                        if not isinstance(obj, dict):
                            continue
                        # 过滤过期/短回复行
                        clean = _sanitize_item(obj)
                        if clean is None:
                            continue
                        if not _is_fresh(clean, now_ts):
                            continue
                        if not _resp_len_ok(clean):
                            continue
                        total_kept += 1
            except Exception:
                continue

        scanned = len(files_scan)
        if scanned == 0:
            return len(files_all) * max(int(approx), 1)

        avg = total_kept / scanned
        rest_files = max(len(files_all) - scanned, 0)
        return int(total_kept + avg * rest_files)

    except Exception:
        return len(glob.glob(os.path.join(dirpath, ROLL_GLOB))) * (
            approx_per_file if approx_per_file else ROLL_APPROX_PER_FILE
        )

def get_pool_stats(dirpath: str) -> Dict:
    """
    返回更详细的池统计（用于日志/监控）：
    {
        'files': N,
        'candidates': 行计数（含坏行/过期前）,
        'usable': 可用行（不过期 & 回复长度足够）,
        'avg_age_sec': 平均年龄（可用行）,
        'p50_age_sec': 中位数年龄（可用行）,
        'p90_age_sec': 90 分位年龄（可用行）
    }
    """
    import statistics as _st

    stats = {
        "files": 0,
        "candidates": 0,
        "usable": 0,
        "avg_age_sec": None,
        "p50_age_sec": None,
        "p90_age_sec": None,
    }
    try:
        _clean_tmp_leftovers(dirpath)
        files = glob.glob(os.path.join(dirpath, ROLL_GLOB))
        stats["files"] = len(files)
        if not files:
            return stats

        now_ts = _now_ts()
        ages = []

        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        stats["candidates"] += 1
                        obj = _safe_parse(ln)
                        if not isinstance(obj, dict):
                            continue
                        clean = _sanitize_item(obj)
                        if clean is None:
                            continue
                        if not _is_fresh(clean, now_ts):
                            continue
                        if not _resp_len_ok(clean):
                            continue
                        stats["usable"] += 1
                        ts = clean.get("ts")
                        if isinstance(ts, (int, float)):
                            ages.append(max(0.0, now_ts - float(ts)))
            except Exception:
                continue

        if ages:
            ages_sorted = sorted(ages)
            stats["avg_age_sec"] = float(sum(ages_sorted) / len(ages_sorted))
            stats["p50_age_sec"] = float(ages_sorted[len(ages_sorted)//2])
            p90_idx = int(len(ages_sorted) * 0.9) - 1
            p90_idx = min(max(p90_idx, 0), len(ages_sorted) - 1)
            stats["p90_age_sec"] = float(ages_sorted[p90_idx])

        if ROLL_VERBOSE:
            print(
                "[rollout_pool.stats] "
                f"files={stats['files']} candidates={stats['candidates']} usable={stats['usable']} "
                f"avg_age={stats['avg_age_sec']}s p50={stats['p50_age_sec']}s p90={stats['p90_age_sec']}s "
                f"ttl={ROLL_MAX_AGE_SEC}s min_resp={ROLL_MIN_RESP_TOKENS}"
            )

        return stats
    except Exception:
        return stats
