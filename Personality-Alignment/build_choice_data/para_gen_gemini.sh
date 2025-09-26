#!/bin/bash

# filepath: /home/shurui/shurui/build_choice_data/para_gen_gemini.sh

# ========================= 配置参数 ========================= #
# 数据配置
DATA_PATH="/Users/lizli/Desktop/build_choice_data/dialogue_dataset_all_v8_summarized_cleaned.jsonl"
OUTPUT_DIR="/Users/lizli/Desktop/build_choice_data/parallel_results_gemini"
FINAL_OUTPUT="/Users/lizli/Desktop/build_choice_data/raw_choice_data_v9_hard_gemini.jsonl"

# Gemini配置
GEMINI_MODEL="gemini-2.5-flash"  # 或者 "gemini-1.5-pro"
GEMINI_API_KEY="sk-ejiZhvl2oKmLW3GtXv5Uv8G1KyZN4gsWeFHt26g22w4Ty4A2"  # 请替换为你的Gemini API密钥

# 并行配置
NUM_PROCESSES=8  # Gemini的并发限制通常比GPT更严格，建议使用较少的进程
BATCH_SIZE=16     # Gemini建议使用较小的批处理大小

# 其他配置
MAX_RETRIES=5    # Gemini可能需要更多重试
RETRY_DELAY=3    # Gemini建议更长的重试延迟

# ========================= 准备工作 ========================= #
echo "开始并行Gemini数据生成..."
echo "进程数: $NUM_PROCESSES"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "Gemini模型: $GEMINI_MODEL"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

# 检查API密钥
if [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
    echo "错误: 请设置正确的Gemini API密钥"
    exit 1
fi

# 计算数据总数
TOTAL_LINES=$(wc -l < "$DATA_PATH")
echo "数据总数: $TOTAL_LINES"

# 计算每个进程处理的数据量
LINES_PER_PROCESS=$((TOTAL_LINES / NUM_PROCESSES))
REMAINDER=$((TOTAL_LINES % NUM_PROCESSES))

echo "每个进程处理约 $LINES_PER_PROCESS 条数据"

# ========================= 启动并行进程 ========================= #
pids=()
start_time=$(date +%s)

for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    # 计算当前进程的开始和结束索引
    start_index=$((i * LINES_PER_PROCESS))
    
    if [ $i -eq $((NUM_PROCESSES - 1)) ]; then
        # 最后一个进程处理剩余的所有数据
        end_index=$TOTAL_LINES
    else
        end_index=$(((i + 1) * LINES_PER_PROCESS))
    fi
    
    # 输出文件名
    output_file="$OUTPUT_DIR/part_${i}_${start_index}_${end_index}.jsonl"
    log_file="$OUTPUT_DIR/logs/process_${i}.log"
    
    echo "启动进程 $i: 处理索引 [$start_index:$end_index] -> $output_file"
    
    # 启动Python进程
    python3 build_choice_data.py \
        --data_path "$DATA_PATH" \
        --save_path "$output_file" \
        --model_type gemini \
        --gemini_model "$GEMINI_MODEL" \
        --gemini_api_key "$GEMINI_API_KEY" \
        --batch_size "$BATCH_SIZE" \
        --max_retries "$MAX_RETRIES" \
        --retry_delay "$RETRY_DELAY" \
        --start_index "$start_index" \
        --end_index "$end_index" \
        > "$log_file" 2>&1 &
    
    # 记录进程ID
    pid=$!
    pids+=($pid)
    
    echo "进程 $i 已启动 (PID: $pid)"
    
    # Gemini API有更严格的速率限制，增加启动延迟
    sleep 2
done

echo "所有 $NUM_PROCESSES 个进程已启动"
echo "进程PID列表: ${pids[*]}"

# ========================= 监控进程状态 ========================= #
echo "开始监控进程状态..."

# 定义监控函数
monitor_processes() {
    while true; do
        running_count=0
        completed_count=0
        failed_count=0
        
        for i in "${!pids[@]}"; do
            pid=${pids[$i]}
            if kill -0 "$pid" 2>/dev/null; then
                running_count=$((running_count + 1))
            else
                # 检查进程退出状态
                wait "$pid"
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    completed_count=$((completed_count + 1))
                else
                    failed_count=$((failed_count + 1))
                    echo "警告: 进程 $i (PID: $pid) 异常退出，退出码: $exit_code"
                fi
            fi
        done
        
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        
        echo "状态更新 (运行时间: ${elapsed_time}s): 运行中=$running_count, 已完成=$completed_count, 失败=$failed_count"
        
        # 显示各进程的进度信息
        for i in $(seq 0 $((NUM_PROCESSES - 1))); do
            log_file="$OUTPUT_DIR/logs/process_${i}.log"
            if [ -f "$log_file" ]; then
                # 获取最后几行日志来显示进度
                last_line=$(tail -n 1 "$log_file" 2>/dev/null | grep -E "(正在生成|Generating|处理完成|保存)" | head -1)
                if [ -n "$last_line" ]; then
                    echo "  进程 $i: $last_line"
                fi
            fi
        done
        
        if [ $running_count -eq 0 ]; then
            break
        fi
        
        sleep 60  # Gemini处理较慢，每60秒检查一次
    done
}

# 启动监控
monitor_processes

# ========================= 等待所有进程完成 ========================= #
echo "等待所有进程完成..."

failed_processes=()
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    echo "等待进程 $i (PID: $pid)..."
    
    if wait "$pid"; then
        echo "进程 $i 成功完成"
    else
        echo "进程 $i 失败"
        failed_processes+=($i)
    fi
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "所有进程已完成，总耗时: ${total_time}s"

# ========================= 检查结果和合并文件 ========================= #
if [ ${#failed_processes[@]} -gt 0 ]; then
    echo "警告: 以下进程执行失败: ${failed_processes[*]}"
    echo "请检查对应的日志文件获取详细错误信息:"
    for failed_process in "${failed_processes[@]}"; do
        echo "  - 进程 $failed_process 日志: $OUTPUT_DIR/logs/process_${failed_process}.log"
    done
fi

echo "开始合并结果文件..."

# 检查输出文件
successful_files=()
total_processed_lines=0

for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    start_index=$((i * LINES_PER_PROCESS))
    if [ $i -eq $((NUM_PROCESSES - 1)) ]; then
        end_index=$TOTAL_LINES
    else
        end_index=$(((i + 1) * LINES_PER_PROCESS))
    fi
    
    output_file="$OUTPUT_DIR/part_${i}_${start_index}_${end_index}.jsonl"
    
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        line_count=$(wc -l < "$output_file")
        echo "找到输出文件 $output_file (包含 $line_count 行)"
        successful_files+=("$output_file")
        total_processed_lines=$((total_processed_lines + line_count))
    else
        echo "警告: 输出文件不存在或为空: $output_file"
    fi
done

# 合并所有成功的文件
if [ ${#successful_files[@]} -gt 0 ]; then
    echo "合并 ${#successful_files[@]} 个输出文件到 $FINAL_OUTPUT"
    
    # 确保最终输出目录存在
    mkdir -p "$(dirname "$FINAL_OUTPUT")"
    
    # 合并文件
    cat "${successful_files[@]}" > "$FINAL_OUTPUT"
    
    final_line_count=$(wc -l < "$FINAL_OUTPUT")
    echo "合并完成！最终文件包含 $final_line_count 条数据"
    echo "最终输出文件: $FINAL_OUTPUT"
    
    # 验证数据完整性
    if [ $final_line_count -eq $total_processed_lines ]; then
        echo "✓ 数据完整性验证通过"
    else
        echo "⚠ 警告: 合并后的行数($final_line_count)与处理的总行数($total_processed_lines)不匹配"
    fi
else
    echo "错误: 没有找到任何成功的输出文件"
    exit 1
fi

# ========================= 生成汇总报告 ========================= #
echo "生成汇总报告..."

report_file="$OUTPUT_DIR/generation_report.txt"
cat > "$report_file" << EOF
Gemini数据生成汇总报告
=====================

生成时间: $(date)
总耗时: ${total_time}s ($(echo "scale=2; $total_time/60" | bc)分钟)

配置信息:
- 并行进程数: $NUM_PROCESSES
- 批处理大小: $BATCH_SIZE
- Gemini模型: $GEMINI_MODEL
- 最大重试次数: $MAX_RETRIES
- 重试延迟: ${RETRY_DELAY}s

数据统计:
- 输入数据总数: $TOTAL_LINES
- 最终输出数据: $final_line_count
- 数据完整率: $(echo "scale=2; $final_line_count*100/$TOTAL_LINES" | bc)%
- 成功进程数: ${#successful_files[@]}
- 失败进程数: ${#failed_processes[@]}

性能统计:
- 平均处理速度: $(echo "scale=2; $final_line_count*3600/$total_time" | bc) 条/小时
- 每进程平均处理: $(echo "scale=2; $final_line_count/${#successful_files[@]}" | bc) 条

输出文件:
- 最终合并文件: $FINAL_OUTPUT
- 临时文件目录: $OUTPUT_DIR
- 日志目录: $OUTPUT_DIR/logs

$(if [ ${#failed_processes[@]} -gt 0 ]; then
    echo "失败的进程:"
    for failed_process in "${failed_processes[@]}"; do
        echo "- 进程 $failed_process (日志: $OUTPUT_DIR/logs/process_${failed_process}.log)"
    done
    echo ""
    echo "失败原因分析:"
    for failed_process in "${failed_processes[@]}"; do
        log_file="$OUTPUT_DIR/logs/process_${failed_process}.log"
        if [ -f "$log_file" ]; then
            echo "进程 $failed_process 错误信息:"
            tail -n 10 "$log_file" | grep -i error | head -3
        fi
    done
fi)

建议:
$(if [ ${#failed_processes[@]} -gt 0 ]; then
    echo "- 对于失败的进程，可以使用以下命令单独重新处理:"
    for failed_process in "${failed_processes[@]}"; do
        start_index=$((failed_process * LINES_PER_PROCESS))
        if [ $failed_process -eq $((NUM_PROCESSES - 1)) ]; then
            end_index=$TOTAL_LINES
        else
            end_index=$(((failed_process + 1) * LINES_PER_PROCESS))
        fi
        echo "  python3 build_choice_data.py --model_type gemini --gemini_model $GEMINI_MODEL --gemini_api_key $GEMINI_API_KEY --start_index $start_index --end_index $end_index"
    done
fi)
- 如果遇到频繁的API错误，建议减少并行进程数或增加重试延迟
- Gemini-1.5-flash 速度更快但质量略低，Gemini-1.5-pro 质量更高但速度较慢
EOF

echo "汇总报告已保存到: $report_file"

# ========================= 数据质量检查 ========================= #
echo "进行数据质量检查..."

if [ -f "$FINAL_OUTPUT" ] && [ -s "$FINAL_OUTPUT" ]; then
    # 检查JSON格式是否正确
    echo "检查JSON格式..."
    invalid_json_count=0
    line_num=0
    while IFS= read -r line; do
        line_num=$((line_num + 1))
        if ! echo "$line" | python3 -m json.tool >/dev/null 2>&1; then
            invalid_json_count=$((invalid_json_count + 1))
            if [ $invalid_json_count -le 5 ]; then
                echo "  无效JSON行 $line_num: $(echo "$line" | head -c 100)..."
            fi
        fi
    done < "$FINAL_OUTPUT"
    
    if [ $invalid_json_count -eq 0 ]; then
        echo "✓ JSON格式检查通过"
    else
        echo "⚠ 发现 $invalid_json_count 行无效JSON"
    fi
    
    # 检查是否有API错误标记
    error_count=$(grep -c "GEMINI.*ERROR" "$FINAL_OUTPUT" 2>/dev/null || echo "0")
    if [ $error_count -eq 0 ]; then
        echo "✓ 未发现API错误标记"
    else
        echo "⚠ 发现 $error_count 个API错误标记"
    fi
fi

# ========================= 清理临时文件 (可选) ========================= #
read -p "是否删除临时的分片文件? (y/N): " cleanup_temp
if [[ $cleanup_temp =~ ^[Yy]$ ]]; then
    echo "清理临时文件..."
    rm -f "$OUTPUT_DIR"/part_*.jsonl
    echo "临时文件已清理"
else
    echo "保留临时文件以供调试"
fi

echo ""
echo "============================================"
echo "Gemini并行数据生成完成！"
echo "============================================"
echo "最终输出: $FINAL_OUTPUT"
echo "总耗时: ${total_time}s ($(echo "scale=2; $total_time/60" | bc)分钟)"
echo "处理速度: $(echo "scale=2; $final_line_count*3600/$total_time" | bc) 条/小时"

if [ ${#failed_processes[@]} -eq 0 ]; then
    echo "✓ 所有进程都成功完成！"
    exit 0
else
    echo "⚠ 有 ${#failed_processes[@]} 个进程失败，请查看汇总报告获取详细信息"
    echo "报告文件: $report_file"
    exit 1
fi