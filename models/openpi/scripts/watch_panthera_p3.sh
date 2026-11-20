#!/bin/bash
# watcher: 盯 checkpoints/pi05_panthera_p3/panthera_p3(train.py 按 checkpoints/<config>/<exp_name> 落盘),
# 每个新写完的 checkpoint 自动跑 auto_eval_p3(CPU 推理)。
# 退出条件: TRAINING_DONE 标记存在,且没有待评估的已完成 checkpoint。
#
# 注意: 本脚本故意不在启动时 mkdir 任何目录——checkpoint 目录必须由
# train.py 的 initialize_checkpoint_dir 创建,提前创建会让训练报 FileExistsError。
set -u
cd <openpi-tree>
CKPT_DIR="checkpoints/pi05_panthera_p3/panthera_p3"
OUT="$CKPT_DIR/auto_eval/results.jsonl"
POLL=60

is_done() {  # 该 step 是否已有评估结果(或已标记跳过)
    local step="$1"
    grep -q "\"step\": *$step," "$OUT" 2>/dev/null && return 0
    [ -f "$CKPT_DIR/auto_eval/.skip_$step" ] && return 0
    return 1
}

echo "[watch] $(date '+%F %T') 启动,监视 $CKPT_DIR (每 ${POLL}s 轮询)"
while true; do
    pending=0
    for d in "$CKPT_DIR"/[0-9]*; do
        [ -d "$d" ] || continue
        [ -f "$d/_CHECKPOINT_METADATA" ] || continue   # orbax 写完标记
        step=$(basename "$d")
        is_done "$step" && continue
        pending=$((pending + 1))
        mkdir -p "$CKPT_DIR/auto_eval"   # 此时 checkpoint 目录必然已由 train.py 建好
        echo "[watch] $(date '+%F %T') 评估 step $step ..."
        if nice -n 10 /root/.local/bin/uv run python scripts/auto_eval_p3.py \
            --ckpt "$d" --step "$step" --out "$OUT" \
            > "$CKPT_DIR/auto_eval/step_${step}.log" 2>&1; then
            echo "[watch] step $step 完成"
            /root/.local/bin/uv run python scripts/auto_eval_report.py "$OUT" || true
        else
            echo "[watch] step $step 评估失败(详见 auto_eval/step_${step}.log),标记跳过"
            touch "$CKPT_DIR/auto_eval/.skip_$step"
        fi
    done

    if [ -f "$CKPT_DIR/TRAINING_DONE" ] && [ "$pending" -eq 0 ]; then
        echo "[watch] $(date '+%F %T') 训练已结束且无待评估 checkpoint,退出"
        echo "[watch] ===== 最终汇总 ====="
        /root/.local/bin/uv run python scripts/auto_eval_report.py "$OUT" || true
        break
    fi
    sleep "$POLL"
done
