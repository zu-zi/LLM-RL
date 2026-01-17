#!/usr/bin/env python
"""Simple metrics visualizer for DAPO runs.

Usage examples:
  # single run
  python tools/plot_metrics.py --csv Results/<run_dir>/metrics.csv

  # overlay two runs
  python tools/plot_metrics.py --csv Results/<runA>/metrics.csv --csv2 Results/<runB>/metrics.csv --label2 randomK256

Outputs PNGs next to the CSV (or to --out-dir if provided).
"""

import argparse
import os
import csv
import matplotlib.pyplot as plt


def _to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _read(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    if not rows:
        return cols
    keys = rows[0].keys()
    for k in keys:
        cols[k] = [r.get(k, "") for r in rows]
    return cols


def _safe(cols, key, default=None):
    return cols.get(key, default)


def plot_one(ax, cols, label, x_key, y_key):
    x = _safe(cols, x_key)
    y = _safe(cols, y_key)
    if x is None or y is None:
        return
    xf = [_to_float(v) for v in x]
    yf = [_to_float(v) for v in y]
    # drop Nones synchronously
    out_x, out_y = [], []
    for a, b in zip(xf, yf):
        if a is None or b is None:
            continue
        out_x.append(a)
        out_y.append(b)
    if out_x:
        ax.plot(out_x, out_y, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="metrics.csv path")
    ap.add_argument("--csv2", default=None, help="optional second metrics.csv")
    ap.add_argument("--label", default=None, help="label for first curve")
    ap.add_argument("--label2", default=None, help="label for second curve")
    ap.add_argument("--out-dir", default=None, help="output directory for PNGs")
    args = ap.parse_args()

    cols1 = _read(args.csv)
    cols2 = _read(args.csv2) if args.csv2 else None

    label1 = args.label or os.path.basename(os.path.dirname(args.csv))
    label2 = args.label2 or (os.path.basename(os.path.dirname(args.csv2)) if args.csv2 else None)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    # 1) Reward vs iter (rm_mean + eval_greedy if present)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_one(ax, cols1, label1 + ": rm_mean", "iter", "rm_mean")
    if "r_eval_greedy" in cols1:
        plot_one(ax, cols1, label1 + ": eval_greedy", "iter", "r_eval_greedy")
    if cols2 is not None:
        plot_one(ax, cols2, label2 + ": rm_mean", "iter", "rm_mean")
        if "r_eval_greedy" in cols2:
            plot_one(ax, cols2, label2 + ": eval_greedy", "iter", "r_eval_greedy")
    ax.set_xlabel("iter")
    ax.set_ylabel("reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_vs_iter.png"), dpi=180)
    plt.close(fig)

    # 2) Reward vs cumulative selected tokens (Neff cumulative)
    if "sel_tokens_total" in cols1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x1_raw = [_to_float(v) or 0.0 for v in cols1.get("sel_tokens_total", [])]
        x1 = []
        acc = 0.0
        for v in x1_raw:
            acc += float(v)
            x1.append(acc)
        y1 = [_to_float(v) for v in cols1.get("rm_mean", [])]
        ax.plot(x1, [vv if vv is not None else float("nan") for vv in y1], label=label1)
        if cols2 is not None and "sel_tokens_total" in cols2:
            x2_raw = [_to_float(v) or 0.0 for v in cols2.get("sel_tokens_total", [])]
            x2 = []
            acc2 = 0.0
            for v in x2_raw:
                acc2 += float(v)
                x2.append(acc2)
            y2 = [_to_float(v) for v in cols2.get("rm_mean", [])]
            ax.plot(x2, [vv if vv is not None else float("nan") for vv in y2], label=label2)
        ax.set_xlabel("cumulative sel_tokens_total")
        ax.set_ylabel("rm_mean")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "reward_vs_cumSel.png"), dpi=180)
        plt.close(fig)

    # 3) KL vs iter (and grad_norm optional)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_one(ax, cols1, label1, "iter", "kl_mean")
    if cols2 is not None:
        plot_one(ax, cols2, label2, "iter", "kl_mean")
    ax.set_xlabel("iter")
    ax.set_ylabel("kl_mean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kl_vs_iter.png"), dpi=180)
    plt.close(fig)

    if "grad_norm" in cols1 or (cols2 is not None and "grad_norm" in cols2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if "grad_norm" in cols1:
            plot_one(ax, cols1, label1, "iter", "grad_norm")
        if cols2 is not None and "grad_norm" in cols2:
            plot_one(ax, cols2, label2, "iter", "grad_norm")
        ax.set_xlabel("iter")
        ax.set_ylabel("grad_norm")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gradnorm_vs_iter.png"), dpi=180)
        plt.close(fig)

    print(f"[ok] wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
