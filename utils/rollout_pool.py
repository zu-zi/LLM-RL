# utils/rollout_pool.py
import os, json, uuid, glob, time, random, hashlib
from typing import List, Dict, Optional

ROLL_GLOB = "roll_*.jsonl"
TMP_SUFFIX = ".tmp"

# 固定策略
ROLL_MAX_AGE_SEC = 300
ROLL_MIN_RESP_TOKENS = 24
ROLL_DEDUP_PROMPT_IN_BATCH = True
ROLL_FILE_ORDER = "mtime_desc"
ROLL_VERBOSE = True

# 估算用
ROLL_SCAN_CAP_FILES = 12
ROLL_APPROX_PER_FILE = 64

# ---- 基础 IO ----
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _atomic_write(path: str, text: str) -> None:
    tmp = path + TMP_SUFFIX
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _read_lines(fp: str) -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        return f.readlines()

def _safe_parse(line: str) -> Optional[dict]:
    try:
        s = line.strip()
        return json.loads(s) if s else None
    except Exception:
        return None

def _atomic_rewrite(fp: str, lines: List[str]) -> None:
    if not lines:
        try: os.remove(fp)
        except FileNotFoundError: pass
        return
    if not lines[-1].endswith("\n"):
        lines = lines + ["\n"]
    _atomic_write(fp, "".join(lines))

def _clean_tmp_leftovers(dirpath: str) -> None:
    for fp in glob.glob(os.path.join(dirpath, f"*{TMP_SUFFIX}")):
        try: os.remove(fp)
        except Exception: pass

def _now_ts() -> float:
    return time.time()

# ---- 校验 / 规范化 ----
def _is_int_list(x) -> bool:
    return isinstance(x, list) and all(isinstance(t, int) for t in x)

def _hash_ids(ids: List[int]) -> str:
    h = hashlib.sha1()
    h.update((",".join(map(str, ids))).encode("utf-8"))
    return h.hexdigest()

def _sanitize_item(it: dict) -> Optional[dict]:
    if not isinstance(it, dict): return None
    p, f = it.get("prompt_ids"), it.get("full_ids")
    if not (_is_int_list(p) and _is_int_list(f)): return None
    if len(f) <= len(p): return None  # 必须含响应

    ts = it.get("ts"); ts = float(ts) if isinstance(ts, (int, float)) else _now_ts()
    pid = it.get("pid") or _hash_ids(p)
    hid = it.get("hid") or _hash_ids(f)

    out = {"prompt_ids": p, "full_ids": f, "ts": ts, "pid": pid, "hid": hid}
    for k in ("prompt_text", "response_text"):
        if k in it: out[k] = it[k]
    return out

# ---- 写入 ----
def enqueue_items(dirpath: str, items: List[Dict]) -> str:
    ensure_dir(dirpath); _clean_tmp_leftovers(dirpath)
    path = os.path.join(dirpath, f"roll_{uuid.uuid4().hex}.jsonl")

    seen, buf, kept, dropped = set(), [], 0, 0
    for it in items:
        clean = _sanitize_item(it)
        if clean is None: dropped += 1; continue
        hid = clean["hid"]
        if hid in seen: dropped += 1; continue
        seen.add(hid)
        buf.append(json.dumps(clean, ensure_ascii=False))
        kept += 1

    if not buf:
        return path  # 不落盘：用于上层拿到拟写入路径也不影响逻辑

    _atomic_write(path, "\n".join(buf) + "\n")
    if ROLL_VERBOSE:
        print(f"[rollout_pool.enqueue] file={os.path.basename(path)} kept={kept} dropped={dropped}")
    return path

# ---- 读取 ----
def _file_order(files: List[str]) -> List[str]:
    if not files: return files
    mode = ROLL_FILE_ORDER
    if mode == "random":
        random.shuffle(files); return files
    if mode == "name_asc":
        return sorted(files)
    return sorted(files, key=lambda fp: os.path.getmtime(fp), reverse=True)

def _is_fresh(obj: dict, now_ts: float) -> bool:
    if ROLL_MAX_AGE_SEC <= 0: return True
    ts = obj.get("ts")
    if not isinstance(ts, (int, float)): return True
    return (now_ts - float(ts)) <= float(ROLL_MAX_AGE_SEC)

def _resp_len_ok(obj: dict) -> bool:
    p, f = obj.get("prompt_ids", []), obj.get("full_ids", [])
    if not (_is_int_list(p) and _is_int_list(f)): return False
    return (len(f) - len(p)) >= max(0, int(ROLL_MIN_RESP_TOKENS))

def _stringify(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)

def dequeue_items(dirpath: str, n: int) -> List[Dict]:
    ensure_dir(dirpath); _clean_tmp_leftovers(dirpath)
    want = max(int(n), 0)
    if want == 0: return []

    files = _file_order(sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB))))
    out, seen_hid, seen_pid = [], set(), set()
    now_ts = _now_ts()

    cnt_scanned = cnt_taken = cnt_dropped_stale = 0
    cnt_dropped_short = cnt_dropped_dup_hid = cnt_dropped_dup_pid = 0

    for fp in files:
        if len(out) >= want: break
        try: lines = _read_lines(fp)
        except (FileNotFoundError, PermissionError): continue

        def _ts(ln: str) -> float:
            obj = _safe_parse(ln)
            return float(obj["ts"]) if isinstance(obj, dict) and isinstance(obj.get("ts"), (int, float)) else 0.0

        lines_sorted = sorted(lines, key=_ts, reverse=True)
        keep_lines: List[str] = []

        for ln in lines_sorted:
            obj = _safe_parse(ln); cnt_scanned += 1
            if obj is None: continue
            clean = _sanitize_item(obj)
            if clean is None: continue
            if not _is_fresh(clean, now_ts): cnt_dropped_stale += 1; continue
            if not _resp_len_ok(clean): cnt_dropped_short += 1; continue

            hid = clean["hid"]
            if hid in seen_hid: cnt_dropped_dup_hid += 1; continue

            if ROLL_DEDUP_PROMPT_IN_BATCH:
                pid = clean["pid"]
                if pid in seen_pid: cnt_dropped_dup_pid += 1; continue

            out.append(clean); seen_hid.add(hid)
            if ROLL_DEDUP_PROMPT_IN_BATCH: seen_pid.add(clean["pid"])
            cnt_taken += 1
            if len(out) >= want: break

        # 回写剩余合格但未被取走的
        for ln in lines:
            obj = _safe_parse(ln)
            if obj is None: continue
            c = _sanitize_item(obj)
            if c is None: continue
            if not _is_fresh(c, now_ts): continue
            if not _resp_len_ok(c): continue
            if c["hid"] in seen_hid: continue
            keep_lines.append(_stringify(c) + "\n")

        try: _atomic_rewrite(fp, keep_lines)
        except Exception:
            try: os.remove(fp)
            except Exception: pass

    if ROLL_VERBOSE:
        print(
            "[rollout_pool.dequeue] "
            f"want={want} got={len(out)} scanned={cnt_scanned} taken={cnt_taken} "
            f"drop(stale)={cnt_dropped_stale} drop(short)={cnt_dropped_short} "
            f"drop(dup_hid)={cnt_dropped_dup_hid} drop(dup_pid)={cnt_dropped_dup_pid} "
            f"ttl={ROLL_MAX_AGE_SEC}s min_resp={ROLL_MIN_RESP_TOKENS}"
        )
    return out

def dequeue_groups(dirpath: str, group_size: int, num_groups: int, allow_partial: bool = False):
    """按同一 pid 聚合成组出队；不影响现有 PPO 逻辑。"""
    ensure_dir(dirpath); _clean_tmp_leftovers(dirpath)
    G = max(int(group_size), 1); K = max(int(num_groups), 0)
    if K == 0: return []

    files = _file_order(sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB))))
    now_ts = _now_ts()

    buckets = {}          # pid -> {"items": [dict...], "hids": set(...)}
    used_hid = set()      # 扫描阶段临时防重复

    cnt_scanned = cnt_kept = cnt_drop_stale = cnt_drop_short = 0

    for fp in files:
        try:
            lines = _read_lines(fp)
        except (FileNotFoundError, PermissionError):
            continue

        def _ts(ln: str) -> float:
            obj = _safe_parse(ln)
            return float(obj["ts"]) if isinstance(obj, dict) and isinstance(obj.get("ts"), (int, float)) else 0.0

        for ln in sorted(lines, key=_ts, reverse=True):
            obj = _safe_parse(ln); cnt_scanned += 1
            if obj is None: continue
            c = _sanitize_item(obj)
            if c is None: continue
            if not _is_fresh(c, now_ts): cnt_drop_stale += 1; continue
            if not _resp_len_ok(c): cnt_drop_short += 1; continue

            pid, hid = c["pid"], c["hid"]
            if hid in used_hid: continue
            b = buckets.get(pid)
            if b is None:
                b = {"items": [], "hids": set()}
                buckets[pid] = b
            if hid in b["hids"]:  # 同一 pid 内去重
                continue
            b["items"].append(c); b["hids"].add(hid); used_hid.add(hid); cnt_kept += 1
            if not allow_partial and len(b["items"]) > G:
                # 只保留最新的前 G 条
                b["items"] = b["items"][:G]
                b["hids"]  = set(it["hid"] for it in b["items"])

    groups, chosen_pids = [], set()
    for pid, b in buckets.items():
        items = b["items"]
        if not items: continue
        if not allow_partial and len(items) < G: continue
        take = items if allow_partial else items[:G]
        groups.append(take); chosen_pids.add(pid)
        if len(groups) >= K: break

    chosen_hids = set(h for grp in groups for h in (it["hid"] for it in grp))

    for fp in files:
        keep_lines = []
        try:
            lines = _read_lines(fp)
        except (FileNotFoundError, PermissionError):
            continue
        for ln in lines:
            obj = _safe_parse(ln)
            if obj is None: continue
            c = _sanitize_item(obj)
            if c is None: continue
            if not _is_fresh(c, now_ts): continue
            if not _resp_len_ok(c): continue
            if c["hid"] in chosen_hids: continue
            keep_lines.append(_stringify(c) + "\n")
        try:
            _atomic_rewrite(fp, keep_lines)
        except Exception:
            try: os.remove(fp)
            except Exception: pass

    if ROLL_VERBOSE:
        total_items = sum(len(g) for g in groups)
        print(
            "[rollout_pool.dequeue_groups] "
            f"want_groups={K} got_groups={len(groups)} group_size={G} total_items={total_items} "
            f"scanned={cnt_scanned} kept_for_bucket={cnt_kept} "
            f"drop(stale)={cnt_drop_stale} drop(short)={cnt_drop_short} "
            f"ttl={ROLL_MAX_AGE_SEC}s min_resp={ROLL_MIN_RESP_TOKENS}"
        )
    return groups

# ---- 估算 / 统计 ----
def estimate_size(dirpath: str, approx_per_file: int = None, scan_cap_files: int = None) -> int:
    try:
        _clean_tmp_leftovers(dirpath)
        files_all = sorted(glob.glob(os.path.join(dirpath, ROLL_GLOB)))
        if not files_all: return 0

        cap = scan_cap_files if isinstance(scan_cap_files, int) and scan_cap_files > 0 else ROLL_SCAN_CAP_FILES
        approx = approx_per_file if isinstance(approx_per_file, int) and approx_per_file > 0 else ROLL_APPROX_PER_FILE

        files_scan = files_all[:max(int(cap), 1)]
        total_kept, now_ts = 0, _now_ts()

        for fp in files_scan:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for ln in f:
                        obj = _safe_parse(ln)
                        if not isinstance(obj, dict): continue
                        c = _sanitize_item(obj)
                        if c and _is_fresh(c, now_ts) and _resp_len_ok(c):
                            total_kept += 1
            except Exception:
                continue

        scanned = len(files_scan)
        if scanned == 0: return len(files_all) * max(int(approx), 1)

        avg = total_kept / scanned
        rest = max(len(files_all) - scanned, 0)
        return int(total_kept + avg * rest)
    except Exception:
        files = glob.glob(os.path.join(dirpath, ROLL_GLOB))
        base = approx_per_file if approx_per_file else ROLL_APPROX_PER_FILE
        return len(files) * base

def get_pool_stats(dirpath: str) -> Dict:
    stats = {"files": 0, "candidates": 0, "usable": 0, "avg_age_sec": None, "p50_age_sec": None, "p90_age_sec": None}
    try:
        _clean_tmp_leftovers(dirpath)
        files = glob.glob(os.path.join(dirpath, ROLL_GLOB))
        stats["files"] = len(files)
        if not files: return stats

        now_ts = _now_ts()
        ages: List[float] = []

        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for ln in f:
                        s = ln.strip()
                        if not s: continue
                        stats["candidates"] += 1
                        obj = _safe_parse(ln)
                        if not isinstance(obj, dict): continue
                        c = _sanitize_item(obj)
                        if c and _is_fresh(c, now_ts) and _resp_len_ok(c):
                            stats["usable"] += 1
                            ts = c.get("ts")
                            if isinstance(ts, (int, float)): ages.append(max(0.0, now_ts - float(ts)))
            except Exception:
                continue

        if ages:
            ages.sort()
            n = len(ages)
            stats["avg_age_sec"] = float(sum(ages) / n)
            stats["p50_age_sec"] = float(ages[n // 2])
            stats["p90_age_sec"] = float(ages[min(max(int(n * 0.9) - 1, 0), n - 1)])

        if ROLL_VERBOSE:
            print(
                "[rollout_pool.stats] "
                f"files={stats['files']} candidates={stats['candidates']} usable={stats['usable']} "
                f"avg_age={stats['avg_age_sec']}s p50={stats['p50_age_sec']}s p90={stats['p90_age_sec']}s "
                f"ttl={ROLL_MAX_AGE_SEC}s min_resp={ROLL_MIN_RESP_TOKENS}"
            )
    except Exception:
        pass
    return stats
