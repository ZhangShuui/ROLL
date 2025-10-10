#!/usr/bin/env bash
set -euo pipefail

# ===== 可配置区 =====
BASE="/project/hdtaccuracy/trains/grpo"
MODELS=( "qwen3_4b_choice_v4")
STEPS=(  399 )

# 是否在收集后立刻导出 HF（0=不导出, 1=导出）
DO_CONVERT="${DO_CONVERT:-1}"
CONVERT_PY="${CONVERT_PY:-python mcore_adapter/tools/convert.py}"
CONVERT_ARGS="${CONVERT_ARGS:-}"   # 需要额外参数可通过环境变量传入
HF_OUT_ROOT="${HF_OUT_ROOT:-/project/hdtaccuracy/trains/converted_hf/grpo}"

TOK_FILES=( tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json mca_config.json )

collect_one() {
  local root="$1"      # /project/.../output
  local step="$2"      # 50/100/...

  # 只拿“目录名本身是 iter_* 的那一层”
  local iter_dir
  iter_dir="$(
    find "$root" -type d -name 'iter_*' \
         -path "*/actor_train-*/checkpoint-$step/iter_*" \
         -printf "%f\n" 2>/dev/null | sort | head -n1 || true
  )"
  [[ -z "${iter_dir:-}" ]] && iter_dir="iter_0000001"

  local dest="$root/pipeline/checkpoint-$step/$iter_dir"
  mkdir -p "$dest"

  echo "==> [$root] step=$step | iter=$iter_dir"

  # 自动探测有哪些 TP 切片（mp_rank_XX）
  mapfile -t ranks < <(
    find "$root" -type f \
      -path "*/actor_train-*/checkpoint-$step/*/mp_rank_*/model_optim_rng.pt" \
      -printf "%h\n" | awk -F/ '{print $NF}' | sort -u
  )

  if [[ "${#ranks[@]}" -eq 0 ]]; then
    echo "   [WARN] 找不到任何 mp_rank_* 切片，跳过"
    return 0
  fi
  echo "   Ranks: ${ranks[*]}"

  # 每个 rank 取第一份（不同 DP 目录里内容重复）
  for r in "${ranks[@]}"; do
    local src destfile
    src="$(find "$root" -type f -path "*/actor_train-*/checkpoint-$step/*/$r/model_optim_rng.pt" \
           | sort | head -n1 || true)"
    if [[ -z "${src:-}" ]]; then
      echo "   [ERR] 未找到 $r 的切片，终止"; return 1
    fi
    mkdir -p "$dest/$r"
    destfile="$dest/$r/model_optim_rng.pt"

    # —— 跳过已存在/同 inode/内容一致 —— 
    if [[ -e "$destfile" ]]; then
      if [[ "$src" -ef "$destfile" ]]; then
        echo "   skip $r: same inode ($destfile)"
        continue
      fi
      if cmp -s "$src" "$destfile"; then
        echo "   skip $r: identical ($destfile)"
        continue
      fi
    fi

    # 先硬链接省空间，失败回退到复制
    ln -f "$src" "$destfile" 2>/dev/null || cp -f "$src" "$destfile"
  done

  # 补齐 tokenizer（找任意一个 checkpoint-$step 下带 tokenizer.json 的目录）
  local tok_src=""
  tok_src="$(find "$root" -type f -path "*/checkpoint-$step/tokenizer.json" -printf "%h\n" | head -n1 || true)"
  if [[ -n "${tok_src:-}" ]]; then
    echo "   Tokenizer from: $tok_src"
    for f in "${TOK_FILES[@]}"; do
      local s="$tok_src/$f" d="$root/pipeline/checkpoint-$step/$f"
      [[ -f "$s" ]] || continue
      if [[ -e "$d" ]]; then
        if [[ "$s" -ef "$d" ]] || cmp -s "$s" "$d"; then
          echo "   skip tokenizer: $f"
          continue
        fi
      fi
      cp -f "$s" "$d"
    done
  else
    echo "   [WARN] 未找到 tokenizer 源，pipeline/$step 将不含 tokenizer 文件"
  fi

  echo "   Collected -> $root/pipeline/checkpoint-$step"
  return 0
}

# ===== 主流程 =====
for m in "${MODELS[@]}"; do
  ROOT="$BASE/$m/output"
  if [[ ! -d "$ROOT" ]]; then
    echo "[WARN] 模型目录不存在，跳过：$ROOT"
    continue
  fi
  echo "######## 模型：$m ########"

  for s in "${STEPS[@]}"; do
    # 该 step 不存在则跳过
    if ! find "$ROOT" -type d -path "*/checkpoint-$s" | grep -q . ; then
      echo "-- step $s: 没有 checkpoint-$s，跳过"
      continue
    fi

    collect_one "$ROOT" "$s"

    if [[ "$DO_CONVERT" == "1" ]]; then
      HF_OUT="$HF_OUT_ROOT/$m/checkpoint-$s"
      mkdir -p "$HF_OUT"
      echo "   [HF] 导出到：$HF_OUT"

      # 如需传 HF 模式开关等参数，放到 CONVERT_ARGS 里
      # 例：export CONVERT_ARGS="--output_format hf --save_safetensors"
      $CONVERT_PY \
        --checkpoint_path "$ROOT/pipeline/checkpoint-$s" \
        --output_path     "$HF_OUT" \
        $CONVERT_ARGS

      # 健康检查
      if ls "$HF_OUT"/config.json "$HF_OUT"/model*.safetensors "$HF_OUT"/tokenizer.json >/dev/null 2>&1; then
        echo "   [HF] OK: $HF_OUT"
      else
        echo "   [HF][WARN] 似乎未产生完整 HF 文件，请检查日志：$HF_OUT"
      fi
    fi
  done
done

echo "=== All done ==="
