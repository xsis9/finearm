"""画 pi05_panthera_p3 的训练 loss + 逐 checkpoint 验证集评估图。

上面板: 训练 loss(对数轴,来自 train-queue 日志的 Step N 行)
下面板: 每个 checkpoint 的验证集回放误差比 ratio = 模型误差/hold 基线(来自 auto_eval results.jsonl)

用法:
  uv run python scripts/plot_panthera_p3_loss.py <train_log> [results.jsonl] [out.png]
默认:
  train_log = <repo>/train-queue/logs/20261107175553-pi05_panthera_p3.log
  results   = checkpoints/pi05_panthera_p3/panthera_p3/auto_eval/results.jsonl
  out       = <train_log 去扩展名>_loss.png (放在日志目录)
"""

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# ---- 中文字体: ttc 不能直接用,从 fontTools 抽出的 ttf(见 memory) ----
FONT = "/root/.cache/matplotlib-cjk/NotoSansSC-Regular.ttf"
if os.path.exists(FONT):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ---- dataviz 参考调色板(palette.md,已过 validate_palette.js 全项) ----
C1, C2 = "#2a78d6", "#eb6834"          # categorical slot1 蓝 / slot2 橙
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRIT = "#d03b3b"                        # status.critical,仅标记评估失败项

TRAIN_LOG = sys.argv[1] if len(sys.argv) > 1 else "<repo>/train-queue/logs/20261107175553-pi05_panthera_p3.log"
RESULTS = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/pi05_panthera_p3/panthera_p3/auto_eval/results.jsonl"
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(TRAIN_LOG)[0] + "_loss.png"

# ---- 训练 loss ----
pat = re.compile(r"Step (\d+): grad_norm=([\d.eE+-]+), loss=([\d.eE+-]+), param_norm=([\d.eE+-]+)")
steps, losses = [], []
for line in open(TRAIN_LOG, encoding="utf-8", errors="replace"):
    m = pat.search(line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(3)))
if not steps:
    sys.exit(f"未在 {TRAIN_LOG} 找到训练 Step 行")

# ---- 验证集逐 checkpoint 指标 ----
ck_steps, ratio_first, ratio_chunk, flag_bad = [], [], [], []
if os.path.exists(RESULTS):
    for line in open(RESULTS, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ck_steps.append(int(r["step"]))
        ratio_first.append(float(r["replay"]["ratio_first"]))
        ratio_chunk.append(float(r["replay"]["ratio_chunk"]))
        f = r.get("flags", {})
        flag_bad.append(not all(f.get(k, False) for k in
                                ("replay_learn", "vision_black", "rollout_no_runaway", "rollout_state_sensitive")))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.2), dpi=150, sharex=True,
                               gridspec_kw={"height_ratios": [3, 2], "hspace": 0.32})
fig.set_facecolor(SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)

# ---- 上:训练 loss ----
ax1.plot(steps, losses, color=C1, lw=2, solid_joinstyle="round")
ax1.set_yscale("log")
ax1.set_title("pi05_panthera_p3 训练 loss(10k 步,batch 64)", color=INK, fontsize=12, loc="left", pad=10)
ax1.set_ylabel("train loss(log)", color=INK2, fontsize=10)
ax1.annotate(f"{losses[-1]:.4f}", xy=(steps[-1], losses[-1]), xytext=(-4, 8),
             textcoords="offset points", ha="right", color=INK2, fontsize=9)

# ---- 下:验证集回放误差比(<1 = 优于 hold 基线) ----
if ck_steps:
    order = sorted(range(len(ck_steps)), key=lambda i: ck_steps[i])
    xs = [ck_steps[i] for i in order]
    y1 = [ratio_first[i] for i in order]
    y2 = [ratio_chunk[i] for i in order]
    bad = [i for i in order if flag_bad[i]]

    ax2.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 4)))
    ax2.text(0.995, 1.0, "hold 基线 = 1", transform=ax2.get_yaxis_transform(),
             ha="right", va="bottom", color=MUTED, fontsize=8.5)
    ax2.plot(xs, y1, color=C1, lw=2, marker="o", ms=6, label="首步误差比 ratio_first")
    ax2.plot(xs, y2, color=C2, lw=2, marker="o", ms=6, label="整段误差比 ratio_chunk")
    if bad:  # 有 flags 未全过的 checkpoint 用 critical × 标出
        ax2.scatter([ck_steps[i] for i in bad], [min(ratio_first[i], ratio_chunk[i]) for i in bad],
                    marker="x", s=70, color=CRIT, zorder=5)
        ax2.text(0.01, 0.06, "× = 行为检查未全过(详见 results.jsonl flags)", transform=ax2.transAxes,
                 color=CRIT, fontsize=8.5)
    ax2.set_title("验证集逐 checkpoint:回放误差比 = 模型误差 / hold 基线(越小越好)",
                  color=INK, fontsize=12, loc="left", pad=10)
    ax2.set_ylabel("误差比", color=INK2, fontsize=10)
    leg = ax2.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
else:
    ax2.text(0.5, 0.5, "暂无 auto_eval 结果", ha="center", va="center",
             transform=ax2.transAxes, color=MUTED, fontsize=11)

ax2.set_xlabel("step", color=INK2, fontsize=10)
ax2.set_xlim(-250, max(steps) + 250)

fig.savefig(OUT, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved:", OUT)
print(f"train points={len(steps)} ({steps[0]}..{steps[-1]}), ckpt eval points={len(ck_steps)}")
