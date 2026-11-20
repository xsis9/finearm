"""打印 auto_eval results.jsonl 汇总表。

用法: uv run python scripts/auto_eval_report.py [results.jsonl]
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pi05_panthera_p3/panthera_p3/auto_eval/results.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

hdr = (f"{'step':>6} {'ratio_first':>11} {'ratio_chunk':>11} {'blackΔarm':>9} "
       f"{'rollAB':>7} {'rollAC':>7} {'maxExc':>7}  flags")
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda x: x["step"]):
    f = r["flags"]
    ok = [k for k in ("replay_learn", "vision_black", "rollout_no_runaway", "rollout_state_sensitive") if f.get(k)]
    print(
        f"{r['step']:>6} {r['replay']['ratio_first']:>11.2f} {r['replay']['ratio_chunk']:>11.2f}"
        f" {r['vision']['black_first_arm']:>9.4f} {r['rollout']['final_AB']:>7.3f} {r['rollout']['final_AC']:>7.3f}"
        f" {r['rollout']['max_excursion_A']:>7.3f}  {'+'.join(ok) if ok else 'NONE'}"
    )
print("\n指标含义: ratio=模型误差/hold基线(越小越好) | blackΔ=涂黑相机后动作变化rad(大=在看图)"
      " | rollAB=同起点扰动终点距(太小=漏斗/盲) | maxExc=闭环离起点最大偏移rad(大=发散)")
