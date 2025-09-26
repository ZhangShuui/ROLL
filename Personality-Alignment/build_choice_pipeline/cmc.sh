#!/bin/bash

# 配置参数
INPUT_DIR="/project/hdtaccuracy/Personality-Alignment/choice_ver/v10/parallel_choice_results_gpt_v2"
OUTPUT_FILE="/project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_enhanced/merged_choice_questions.json"
API_KEY="sk-K6tq07IP2UM744DR1YkZSqZ3MGpab7bJ6IImmBoUWxoT2Jpa"
MAX_WORKERS=16  # 并行线程数

# 新增：原始prompt文件路径
PROMPT_FILE="/project/hdtaccuracy/Personality-Alignment/choice_ver/v10/dialogue_dataset_all_v9_summarized_cleaned.jsonl"  # 请修改为实际路径

echo "开始并行合并多选题结果..."
echo "使用 $MAX_WORKERS 个并行线程"

# 检查prompt文件是否存在
if [ -f "$PROMPT_FILE" ]; then
    echo "使用原始prompt文件: $PROMPT_FILE"
    PROMPT_ARG="--prompt-file $PROMPT_FILE"
else
    echo "警告: 原始prompt文件不存在: $PROMPT_FILE"
    echo "将使用原有的从original_prompt字段提取的方法"
    PROMPT_ARG=""
fi

# 首先进行干运行检查问题
echo "=== 步骤1: 并行检查问题 ==="
python /home/szhangfa/ROLL/Personality-Alignment/build_choice_pipeline/clean_and_merge_choice.py \
    --input-dir "$INPUT_DIR" \
    --pattern "choice_questions_part_*_range_*.json" \
    --output "$OUTPUT_FILE" \
    --api-key "$API_KEY" \
    --max-workers "$MAX_WORKERS" \
    $PROMPT_ARG \
    --dry-run

echo ""
read -p "发现问题后是否继续并行修复? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "=== 步骤2: 并行合并并修复 (使用 $MAX_WORKERS 线程) ==="
    python /home/szhangfa/ROLL/Personality-Alignment/build_choice_pipeline/clean_and_merge_choice.py \
        --input-dir "$INPUT_DIR" \
        --pattern "choice_questions_part_*_range_*.json" \
        --output "$OUTPUT_FILE" \
        --api-key "$API_KEY" \
        --max-workers "$MAX_WORKERS" \
        $PROMPT_ARG \
        --fix-problems
    
    echo "=== 完成! ==="
    echo "合并结果: $OUTPUT_FILE"
    echo "修复报告: ${OUTPUT_FILE%.*}_fix_report.json"
else
    echo "=== 仅并行合并不修复 ==="
    python /home/szhangfa/ROLL/Personality-Alignment/build_choice_pipeline/clean_and_merge_choice.py \
        --input-dir "$INPUT_DIR" \
        --pattern "choice_questions_part_*_range_*.json" \
        --output "$OUTPUT_FILE" \
        --api-key "$API_KEY" \
        --max-workers "$MAX_WORKERS" \
        $PROMPT_ARG
    
    echo "=== 完成! ==="
    echo "合并结果: $OUTPUT_FILE"
fi

# 显示文件信息
echo ""
echo "=== 结果统计 ==="
if [ -f "$OUTPUT_FILE" ]; then
    echo "文件大小: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "问题数量: $(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))")"
fi

# 显示修复报告（如果存在）
REPORT_FILE="${OUTPUT_FILE%.*}_fix_report.json"
if [ -f "$REPORT_FILE" ]; then
    echo ""
    echo "=== 修复报告 ==="
    python -c "
import json
report = json.load(open('$REPORT_FILE'))
print(f'总问题数: {report[\"total_questions\"]}')
print(f'发现问题选项: {report[\"problematic_items_found\"]}')
print(f'成功修复: {report.get(\"successfully_fixed\", 0)}')
print(f'修复失败: {report.get(\"failed_to_fix\", 0)}')
print(f'使用线程数: {report.get(\"parallel_processing\", {}).get(\"max_workers\", \"未知\")}')
print(f'Prompt源: 已加载{report.get(\"prompt_source\", {}).get(\"loaded_prompts\", 0)}个原始prompts')
"
fi