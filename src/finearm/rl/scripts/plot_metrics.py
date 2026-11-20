#!/usr/bin/env python3
"""Plot rl_metrics.log curves (Phase 4/5 monitoring).

Two figures from one train_stage2 output dir:
  - episodes: success rate (rolling 20), transitions, interventions per episode
    (from '#' summary lines);
  - updates: q1_mean / q1_max / td_target / actorQ / bc penalty vs update count
    (from the TSV lines; non-finite cells are skipped).

Usage:
  python scripts/plot_metrics.py --log <output_dir>/rl_metrics.log \
      --out <output_dir>/metrics.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EPISODE_KEYS = ("reward", "transitions", "interventions", "success_last20",
                "avg_int_last20", "buffer", "updates")
TSV_COLUMNS = ("updates", "episode", "buffer", "q1_mean", "q1_max", "q2_max",
               "td_mean", "td_max", "td_err", "actorQ", "bc", "delta", "actor_on")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default="", help="PNG path (default: alongside log)")
    parser.add_argument("--smooth", type=int, default=20,
                        help="Rolling window for the episode success curve")
    return parser.parse_args()


def _f(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_episode_line(line: str) -> dict:
    entry = {}
    for chunk in line.strip().lstrip("#").split("|"):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip().split()[0]
        if key in ("reward", "transitions", "interventions", "buffer", "updates"):
            entry[key] = _f(value)
        elif key in ("success_last20", "avg_int_last20"):
            entry[key] = _f(value)
        elif key.startswith("sources"):
            pass
    return entry


def main():
    args = parse_args()
    log_path = Path(args.log).expanduser()
    episodes, rows = [], []
    header = None
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("# updates"):
            header = line.lstrip("# ").split("\t")  # the TSV header itself
            continue
        if line.startswith("#"):
            entry = parse_episode_line(line)
            if entry:
                entry["index"] = len(episodes) + 1
                episodes.append(entry)
        else:
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            rows.append({_k: _f(_v) for _k, _v in zip(header, parts)})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    if episodes:
        idx = [e["index"] for e in episodes]
        reward = [e.get("reward") or 0.0 for e in episodes]
        rolling = []
        for i in range(len(reward)):
            window = reward[max(0, i - args.smooth + 1):i + 1]
            rolling.append(sum(window) / len(window))
        ax.plot(idx, rolling, "o-", label=f"success rate (roll {args.smooth})", color="tab:green")
        ax.plot(idx, [e.get("transitions") or 0 for e in episodes], label="transitions",
                color="tab:blue", alpha=0.6)
        ax.plot(idx, [e.get("interventions") or 0 for e in episodes], label="interventions",
                color="tab:red", alpha=0.6)
        ax.set_xlabel("episode")
        ax.set_title(f"Episodes ({len(episodes)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no episode lines", ha="center")

    ax = axes[1]
    if rows:
        updates = [r.get("updates") for r in rows]

        def series(key):
            xs = [r["updates"] for r in rows if r.get(key) is not None]
            ys = [r[key] for r in rows if r.get(key) is not None]
            return xs, ys

        for key, color in (("q1_mean", "tab:blue"), ("q1_max", "tab:cyan"),
                           ("td_mean", "tab:purple"), ("actorQ", "tab:orange"),
                           ("bc", "tab:red")):
            xs, ys = series(key)
            if xs:
                ax.plot(xs, ys, label=key, color=color, linewidth=1.2)
        ax.set_xlabel("updates")
        ax.set_title(f"RL metrics ({len(rows)} lines, {int(updates[-1]) if updates else 0} updates)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no TSV lines", ha="center")

    fig.tight_layout()
    out = Path(args.out).expanduser() if args.out else log_path.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"saved: {out} | episodes={len(episodes)} metric_lines={len(rows)}")


if __name__ == "__main__":
    main()
