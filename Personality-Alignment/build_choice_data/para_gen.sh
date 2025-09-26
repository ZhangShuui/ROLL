#!/bin/bash

# filepath: /home/shurui/run_parallel_gpt_generation.sh

# ========================= 配置参数 ========================= #
# 数据配置
DATA_PATH="/Users/lizli/Desktop/build_choice_data/dialogue_dataset_all_v8_summarized_cleaned.jsonl"
OUTPUT_DIR="/Users/lizli/Desktop/build_choice_data/parallel_results_gpt"
FINAL_OUTPUT="/Users/lizli/Desktop/build_choice_data/raw_choice_data_v9_hard_gpt.jsonl"

# GPT配置
GPT_MODEL="gpt-5-mini-2025-08-07"
GPT_API_KEY="sk-K6tq07IP2UM744DR1YkZSqZ3MGpab7bJ6IImmBoUWxoT2Jpa"  # 请替换为你的API密钥
GPT_BASE_URL="https://yunwu.zeabur.app/v1"  # 如果需要，请设置你的API基础URL

# 并行配置
NUM_PROCESSES=10
BATCH_SIZE=16

# 其他配置
MAX_RETRIES=3
RETRY_DELAY=2

# ========================= 准备工作 ========================= #
echo "开始并行GPT数据生成..."
echo "进程数: $NUM_PROCESSES"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
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
        --model_type gpt \
        --gpt_model "$GPT_MODEL" \
        --gpt_api_key "$GPT_API_KEY" \
        --gpt_base_url "$GPT_BASE_URL" \
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
    
    # 避免同时启动太多进程，稍微延迟
    sleep 1
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
        
        if [ $running_count -eq 0 ]; then
            break
        fi
        
        sleep 30  # 每30秒检查一次
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
    echo "请检查对应的日志文件获取详细错误信息"
fi

echo "开始合并结果文件..."

# 检查输出文件
successful_files=()
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
else
    echo "错误: 没有找到任何成功的输出文件"
    exit 1
fi

# ========================= 生成汇总报告 ========================= #
echo "生成汇总报告..."

report_file="$OUTPUT_DIR/generation_report.txt"
cat > "$report_file" << EOF
GPT数据生成汇总报告
====================

生成时间: $(date)
总耗时: ${total_time}s

配置信息:
- 并行进程数: $NUM_PROCESSES
- 批处理大小: $BATCH_SIZE
- GPT模型: $GPT_MODEL
- 最大重试次数: $MAX_RETRIES

数据统计:
- 输入数据总数: $TOTAL_LINES
- 最终输出数据: $final_line_count
- 成功进程数: ${#successful_files[@]}
- 失败进程数: ${#failed_processes[@]}

输出文件:
- 最终合并文件: $FINAL_OUTPUT
- 临时文件目录: $OUTPUT_DIR

$(if [ ${#failed_processes[@]} -gt 0 ]; then
    echo "失败的进程:"
    for failed_process in "${failed_processes[@]}"; do
        echo "- 进程 $failed_process"
    done
fi)
EOF

echo "汇总报告已保存到: $report_file"

# ========================= 清理临时文件 (可选) ========================= #
read -p "是否删除临时的分片文件? (y/N): " cleanup_temp
if [[ $cleanup_temp =~ ^[Yy]$ ]]; then
    echo "清理临时文件..."
    rm -f "$OUTPUT_DIR"/part_*.jsonl
    echo "临时文件已清理"
else
    echo "保留临时文件以供调试"
fi

echo "并行GPT数据生成完成！"
echo "最终输出: $FINAL_OUTPUT"
echo "总耗时: ${total_time}s"

if [ ${#failed_processes[@]} -eq 0 ]; then
    echo "所有进程都成功完成！"
    exit 0
else
    echo "有 ${#failed_processes[@]} 个进程失败，请检查日志"
    exit 1
fi