#!/bin/bash

set -e  # 遇到错误时退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/build_choice_data_hard.py"

# 默认配置
MODEL_TYPE="local"
BATCH_SIZE=32
DATA_LIMIT_FLAG=""
DATA_LIMIT_VALUE=""
START_INDEX_FLAG=""
START_INDEX_VALUE=""
END_INDEX_FLAG=""
END_INDEX_VALUE=""
MODES_FLAG=""
MODES_VALUE=""
TEST_MODE_FLAG=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data-limit)
            DATA_LIMIT_FLAG="--data_limit"
            DATA_LIMIT_VALUE="$2"
            shift 2
            ;;
        --start-index)
            START_INDEX_FLAG="--start_index"
            START_INDEX_VALUE="$2"
            shift 2
            ;;
        --end-index)
            END_INDEX_FLAG="--end_index"
            END_INDEX_VALUE="$2"
            shift 2
            ;;
        --modes)
            MODES_FLAG="--modes"
            MODES_VALUE="$2"
            shift 2
            ;;
        --test)
            TEST_MODE_FLAG="--test_mode"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --model TYPE         模型类型 (local/gpt/gemini)"
            echo "  --batch-size SIZE    批处理大小"
            echo "  --data-limit NUM     数据限制条数"
            echo "  --start-index START  开始处理的索引"
            echo "  --end-index END      结束处理的索引"
            echo "  --modes MODE1,MODE2  指定生成模式"
            echo "  --test               测试模式"
            echo "  -h, --help           显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "Hard级别选择题数据生成"
echo "=================================="
echo "模型类型: $MODEL_TYPE"
echo "批处理大小: $BATCH_SIZE"
if [ -n "$START_INDEX_VALUE" ]; then
    echo "开始索引: $START_INDEX_VALUE"
fi
if [ -n "$END_INDEX_VALUE" ]; then
    echo "结束索引: $END_INDEX_VALUE"
fi
if [ -n "$MODES_VALUE" ]; then
    echo "指定模式: $MODES_VALUE"
fi
if [ -n "$TEST_MODE_FLAG" ]; then
    echo "运行模式: 测试模式"
fi
echo "=================================="

# 构建Python命令参数数组
PYTHON_ARGS=(
    --model_type "$MODEL_TYPE"
    --batch_size "$BATCH_SIZE"
)

# 根据模型类型添加模型路径
case $MODEL_TYPE in
    local)
        PYTHON_ARGS+=(--model_path "/project/hdtaccuracy/models/base/Qwen3-4B")
        ;;
    gpt)
        if [ -z "$GPT_API_KEY" ]; then
            echo "错误: 使用 GPT 模型需要设置 GPT_API_KEY 环境变量"
            exit 1
        fi
        PYTHON_ARGS+=(--gpt_model "gpt-3.5-turbo" --gpt_api_key "$GPT_API_KEY")
        ;;
    gemini)
        if [ -z "$GEMINI_API_KEY" ]; then
            echo "错误: 使用 Gemini 模型需要设置 GEMINI_API_KEY 环境变量"
            exit 1
        fi
        PYTHON_ARGS+=(--gemini_model "gemini-1.5-flash" --gemini_api_key "$GEMINI_API_KEY")
        ;;
    *)
        echo "错误: 不支持的模型类型: $MODEL_TYPE"
        echo "支持的类型: local, gpt, gemini"
        exit 1
        ;;
esac

# 添加可选参数 - 只有当标志非空时才添加
if [ -n "$DATA_LIMIT_FLAG" ] && [ -n "$DATA_LIMIT_VALUE" ]; then
    PYTHON_ARGS+=("$DATA_LIMIT_FLAG" "$DATA_LIMIT_VALUE")
fi

if [ -n "$START_INDEX_FLAG" ] && [ -n "$START_INDEX_VALUE" ]; then
    PYTHON_ARGS+=("$START_INDEX_FLAG" "$START_INDEX_VALUE")
fi

if [ -n "$END_INDEX_FLAG" ] && [ -n "$END_INDEX_VALUE" ]; then
    PYTHON_ARGS+=("$END_INDEX_FLAG" "$END_INDEX_VALUE")
fi

if [ -n "$MODES_FLAG" ] && [ -n "$MODES_VALUE" ]; then
    PYTHON_ARGS+=("$MODES_FLAG")
    # 处理多个模式（以空格或逗号分隔）
    IFS=',' read -ra MODE_ARRAY <<< "$MODES_VALUE"
    for mode in "${MODE_ARRAY[@]}"; do
        # 去除前后空格
        mode=$(echo "$mode" | xargs)
        PYTHON_ARGS+=("$mode")
    done
fi

if [ -n "$TEST_MODE_FLAG" ]; then
    PYTHON_ARGS+=("$TEST_MODE_FLAG")
fi

# 执行Python脚本
echo "执行命令: python3 $PYTHON_SCRIPT ${PYTHON_ARGS[*]}"
python3 "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"

echo "脚本执行完成！"