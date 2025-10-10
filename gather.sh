#!/usr/bin/env bash
set -euo pipefail

# ========= 可配置区 =========
BASE="/project/hdtaccuracy/trains/gspo"
MODELS=( "qwen3_1_7b_choice_v4_token" )
STEPS=( 150 )

# 收集后是否立刻导出为 HF（0=不导出, 1=导出）
DO_CONVERT="${DO_CONVERT:-0}"
CONVERT_PY="${CONVERT_PY:-python mcore_adapter/tools/convert.py}"
CONVERT_ARGS="${CONVERT_ARGS:-}"   # 例：--output_format hf --save_safetensors
HF_OUT_ROOT="${HF_OUT_ROOT:-/project/hdtaccuracy/trains/converted_hf/gspo}"

# 需要放到 pipeline/checkpoint-$STEP 顶层的一些文件
TOK_FILES=( tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json mca_config.json scheduler.pt latest_checkpointed_iteration.txt )

log() { echo -e "$*" >&2; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }

# 目录整树复制：优先硬链接，失败回退 rsync/cp（当前脚本未直接使用，保留备用）
copy_tree() {
  local src="$1" dst="$2"
  mkdir -p "$dst"
  if cp -al "$src"/ "$dst"/ 2>/dev/null; then
    return 0
  fi
  if has_cmd rsync; then
    rsync -aH --delete "$src"/ "$dst"/ 2>/dev/null || rsync -a "$src"/ "$dst"/
    return 0
  fi
  shopt -s dotglob nullglob
  for f in "$src"/*; do
    local rel="${f#$src/}"
    local dstd="$dst/$rel"
    mkdir -p "$(dirname "$dstd")"
    ln "$f" "$dstd" 2>/dev/null || cp -a "$f" "$dstd"
  done
  shopt -u dotglob nullglob
}

# 合并单个文件到目标；若已存在但内容不同，自动重命名 *_dupN
merge_one_file() {
  local src="$1" dst_dir="$2" base dst newdst n
  base="$(basename "$src")"
  mkdir -p "$dst_dir"
  dst="$dst_dir/$base"
  if [[ -e "$dst" ]]; then
    if [[ "$src" -ef "$dst" ]] || cmp -s "$src" "$dst"; then
      log "      skip identical: $base"
      return 0
    fi
    n=1
    while :; do
      newdst="$dst_dir/${base%.*}_dup${n}.${base##*.}"
      if [[ ! -e "$newdst" ]]; then
        log "      name clash: $base -> $(basename "$newdst")"
        ln "$src" "$newdst" 2>/dev/null || cp -a "$src" "$newdst"
        return 0
      fi
      ((n++))
    done
  fi
  ln "$src" "$dst" 2>/dev/null || cp -a "$src" "$dst"
}

collect_one() {
  local root="$1"   # /.../output
  local step="$2"   # 50/100/...

  # 先从任意一个 DP 找到 iter_* 名称（多数情况下相同）；找不到就用默认名
  local iter_dir
  iter_dir="$(
    find "$root" -type d -name 'iter_*' \
         -path "*/actor_train-*/checkpoint-$step/iter_*" \
         -printf "%f\n" 2>/dev/null | sort | head -n1 || true
  )"
  [[ -z "${iter_dir:-}" ]] && iter_dir="iter_0000001"

  local pipeline_root="$root/pipeline/checkpoint-$step"
  local dest="$pipeline_root/$iter_dir"
  mkdir -p "$dest"

  log "==> [$root] step=$step | iter=$iter_dir"

  # 1) 列出每个 DP 的 checkpoint 根目录：actor_train-*/checkpoint-$step
  local -a dp_dirs
  mapfile -t dp_dirs < <(find "$root" -type d -path "*/actor_train-*/checkpoint-$step" | sort -u)
  if [[ "${#dp_dirs[@]}" -eq 0 ]]; then
    log "   [WARN] 未找到任何 DP 目录（actor_train-*/checkpoint-$step），跳过"
    return 0
  fi

  # 2) 统计所有 DP 中的 mp_rank_*（用 iter_* 通配，避免 iter 名不一致导致漏掉）
  local -a ranks
  mapfile -t ranks < <(
    find "${dp_dirs[@]}" -type f \
      -path "*/iter_*/mp_rank_*/model_optim_rng.pt" \
      -printf "%h\n" | awk -F/ '{print $NF}' | sort -u
  )
  if [[ "${#ranks[@]}" -eq 0 ]]; then
    log "   [WARN] 未发现 mp_rank_*，跳过 mp 切片复制"
  else
    log "   Ranks: ${ranks[*]}"
    local r src dstfile
    for r in "${ranks[@]}"; do
      # 从所有 DP 中拿该 rank 的第一份
      src="$(find "${dp_dirs[@]}" -type f -path "*/iter_*/$r/model_optim_rng.pt" | sort | head -n1 || true)"
      if [[ -z "${src:-}" ]]; then
        log "   [ERR] 未找到 $r 的切片"; continue
      fi
      mkdir -p "$dest/$r"
      dstfile="$dest/$r/model_optim_rng.pt"
      if [[ -e "$dstfile" ]] && { [[ "$src" -ef "$dstfile" ]] || cmp -s "$src" "$dstfile"; }; then
        log "   skip $r: identical"
        continue
      fi
      ln -f "$src" "$dstfile" 2>/dev/null || cp -f "$src" "$dstfile"
    done
  fi

  # 3) 合并 dist_optimizer：优先 iter_*/dist_optimizer，其次根上的 dist_optimizer（兜底）
  local opt_dst="$dest/dist_optimizer"
  mkdir -p "$opt_dst"
  local found_any=0
  local dp_dir
  for dp_dir in "${dp_dirs[@]}"; do
    local -a opt_candidates
    mapfile -t opt_candidates < <(
      find "$dp_dir" -maxdepth 2 -type d -name "dist_optimizer" -path "*/iter_*/dist_optimizer" | sort
    )
    if [[ "${#opt_candidates[@]}" -eq 0 && -d "$dp_dir/dist_optimizer" ]]; then
      opt_candidates=("$dp_dir/dist_optimizer")
    fi
    if [[ "${#opt_candidates[@]}" -eq 0 ]]; then
      log "   [WARN] 找不到 dist_optimizer 于: $dp_dir"
      continue
    fi
    local opt_src
    for opt_src in "${opt_candidates[@]}"; do
      found_any=1
      log "   Merge Dist-Optimizer from: $opt_src"
      shopt -s nullglob dotglob
      for f in "$opt_src"/*; do
        [[ -f "$f" ]] || continue
        merge_one_file "$f" "$opt_dst"
      done
      shopt -u nullglob dotglob
    done
  done
  if [[ "$found_any" -eq 0 ]]; then
    log "   [WARN] 本 step 未发现任何 dist_optimizer 目录（可能未保存优化器状态或已被清理）"
  fi

  # 4) tokenizer/metadata：找任一 DP 的 checkpoint-$step 下带 tokenizer.json 的目录作为来源
  local tok_src=""
  tok_src="$(find "${dp_dirs[@]}" -type f -name tokenizer.json -printf "%h\n" | head -n1 || true)"
  if [[ -n "${tok_src:-}" ]]; then
    log "   Tokenizer from: $tok_src"
    local s d f
    for f in "${TOK_FILES[@]}"; do
      s="$tok_src/$f"; d="$pipeline_root/$f"
      [[ -f "$s" ]] || continue
      if [[ -e "$d" ]] && { [[ "$s" -ef "$d" ]] || cmp -s "$s" "$d"; }; then
        log "   skip tokenizer/meta: $f"
        continue
      fi
      cp -f "$s" "$d"
    done
  else
    log "   [WARN] 未找到 tokenizer 源，pipeline/$step 将不含 tokenizer 文件"
  fi

  log "   Collected -> $pipeline_root"
  return 0
}

main() {
  for m in "${MODELS[@]}"; do
    local ROOT="$BASE/$m/output"
    if [[ ! -d "$ROOT" ]]; then
      log "[WARN] 模型目录不存在，跳过：$ROOT"; continue
    fi
    log "######## 模型：$m ########"

    local s
    for s in "${STEPS[@]}"; do
      if ! find "$ROOT" -type d -path "*/checkpoint-$s" | grep -q . ; then
        log "-- step $s: 没有 checkpoint-$s，跳过"
        continue
      fi

      collect_one "$ROOT" "$s"

      if [[ "$DO_CONVERT" == "1" ]]; then
        local HF_OUT="$HF_OUT_ROOT/$m/checkpoint-$s"
        mkdir -p "$HF_OUT"
        log "   [HF] 导出到：$HF_OUT"
        $CONVERT_PY --checkpoint_path "$ROOT/pipeline/checkpoint-$s" --output_path "$HF_OUT" $CONVERT_ARGS
        if ls "$HF_OUT"/config.json "$HF_OUT"/model*.safetensors "$HF_OUT"/tokenizer.json >/dev/null 2>&1; then
          log "   [HF] OK: $HF_OUT"
        else
          log "   [HF][WARN] 似乎未产生完整 HF 文件，请检查日志：$HF_OUT"
        fi
      fi
    done
  done
  log "=== All done ==="
}

main "$@"
