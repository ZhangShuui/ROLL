#!/usr/bin/env python3
"""
merge_parts.py
合并并行生成的part文件到单个JSONL文件
"""

import os
import json
import argparse
import glob
from typing import List
import re


def find_part_files(output_dir: str) -> List[tuple]:
    """
    查找所有part文件并按索引排序
    返回: [(文件路径, 进程ID, 开始索引, 结束索引), ...]
    """
    part_files = []
    pattern = os.path.join(output_dir, "part_*.jsonl")

    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path)

        # 尝试匹配两种格式:
        # 格式1: part_<process_id>_<start_index>_<end_index>_range_<start>_<end>.jsonl
        # 格式2: part_<process_id>_<start_index>_<end_index>.jsonl

        # 先尝试匹配新格式 (带range)
        match = re.match(r"part_(\d+)_(\d+)_(\d+)_range_(\d+)_(\d+)\.jsonl", filename)
        if match:
            process_id = int(match.group(1))
            start_index = int(match.group(2))
            end_index = int(match.group(3))
            # range部分的start和end应该和前面的一致
            range_start = int(match.group(4))
            range_end = int(match.group(5))

            # 验证range部分是否一致
            if start_index != range_start or end_index != range_end:
                print(
                    f"警告: 文件 {filename} 中的索引不一致: "
                    f"[{start_index}:{end_index}] vs range[{range_start}:{range_end}]"
                )
        else:
            # 尝试匹配原格式
            match = re.match(r"part_(\d+)_(\d+)_(\d+)\.jsonl", filename)
            if match:
                process_id = int(match.group(1))
                start_index = int(match.group(2))
                end_index = int(match.group(3))
            else:
                print(f"警告: 文件名格式不匹配: {filename}")
                continue

        # 检查文件是否存在且不为空
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            part_files.append((file_path, process_id, start_index, end_index))
        else:
            print(f"警告: 文件不存在或为空: {file_path}")

    # 按开始索引排序，确保正确的顺序
    part_files.sort(key=lambda x: x[2])  # 按start_index排序

    return part_files


def validate_part_files(part_files: List[tuple]) -> bool:
    """
    验证part文件的完整性和连续性
    """
    if not part_files:
        print("错误: 没有找到任何part文件")
        return False

    print(f"找到 {len(part_files)} 个part文件:")
    total_expected = 0

    for i, (file_path, process_id, start_index, end_index) in enumerate(part_files):
        line_count = count_lines(file_path)
        expected_lines = end_index - start_index
        total_expected += expected_lines

        print(
            f"  {os.path.basename(file_path)}: "
            f"索引[{start_index}:{end_index}] "
            f"期望{expected_lines}行, 实际{line_count}行"
        )

        if line_count != expected_lines:
            print(f"    警告: 行数不匹配!")

    # 检查索引连续性
    for i in range(1, len(part_files)):
        prev_end = part_files[i - 1][3]  # 前一个文件的end_index
        curr_start = part_files[i][2]  # 当前文件的start_index

        if prev_end != curr_start:
            print(f"警告: 索引不连续! " f"文件{i-1}结束于{prev_end}, 文件{i}开始于{curr_start}")

    print(f"总计期望行数: {total_expected}")
    return True


def count_lines(file_path: str) -> int:
    """计算文件行数"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"读取文件行数失败 {file_path}: {e}")
        return 0


def merge_files(part_files: List[tuple], output_path: str, validate_json: bool = True) -> bool:
    """
    合并所有part文件到单个文件
    """
    print(f"\n开始合并文件到: {output_path}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_lines = 0
    valid_lines = 0
    invalid_lines = 0

    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            for file_path, process_id, start_index, end_index in part_files:
                print(f"处理: {os.path.basename(file_path)}")

                with open(file_path, "r", encoding="utf-8") as input_file:
                    for line_num, line in enumerate(input_file, 1):
                        line = line.strip()
                        if not line:
                            continue

                        total_lines += 1

                        # 验证JSON格式（可选）
                        if validate_json:
                            try:
                                json.loads(line)
                                valid_lines += 1
                            except json.JSONDecodeError as e:
                                invalid_lines += 1
                                print(f"    警告: 第{line_num}行JSON格式错误: {e}")
                                print(f"    内容: {line[:100]}...")
                                continue
                        else:
                            valid_lines += 1

                        # 写入输出文件
                        output_file.write(line + "\n")

        print(f"\n合并完成!")
        print(f"总行数: {total_lines}")
        print(f"有效行数: {valid_lines}")
        if invalid_lines > 0:
            print(f"无效行数: {invalid_lines}")

        return True

    except Exception as e:
        print(f"合并文件时发生错误: {e}")
        return False


def generate_report(part_files: List[tuple], output_path: str, merge_success: bool) -> str:
    """生成合并报告"""
    report_lines = [
        "文件合并报告",
        "=" * 50,
        f"合并时间: {__import__('datetime').datetime.now()}",
        f"合并状态: {'成功' if merge_success else '失败'}",
        "",
        "输入文件:",
    ]

    total_expected = 0
    for file_path, process_id, start_index, end_index in part_files:
        line_count = count_lines(file_path)
        expected_lines = end_index - start_index
        total_expected += expected_lines

        report_lines.append(
            f"  {os.path.basename(file_path)}: "
            f"[{start_index}:{end_index}] "
            f"期望{expected_lines}行, 实际{line_count}行"
        )

    report_lines.extend(
        [
            "",
            f"总计输入文件: {len(part_files)}",
            f"总计期望行数: {total_expected}",
            f"输出文件: {output_path}",
        ]
    )

    if merge_success and os.path.exists(output_path):
        final_count = count_lines(output_path)
        report_lines.append(f"最终输出行数: {final_count}")

    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="合并并行生成的part文件")

    parser.add_argument("--output_dir", type=str, required=True, help="包含part文件的目录")
    parser.add_argument("--output_file", type=str, required=True, help="合并后的输出文件路径")
    parser.add_argument("--no_validate", action="store_true", help="跳过JSON格式验证（加快处理速度）")
    parser.add_argument("--report_file", type=str, default=None, help="生成报告文件路径（可选）")

    args = parser.parse_args()

    print("=" * 60)
    print("Part文件合并工具")
    print("=" * 60)
    print(f"输入目录: {args.output_dir}")
    print(f"输出文件: {args.output_file}")
    print(f"JSON验证: {'关闭' if args.no_validate else '开启'}")

    # 检查输入目录
    if not os.path.exists(args.output_dir):
        print(f"错误: 输入目录不存在: {args.output_dir}")
        return 1

    # 查找part文件
    print("\n查找part文件...")
    part_files = find_part_files(args.output_dir)

    if not part_files:
        print("错误: 没有找到任何part文件")
        return 1

    # 验证文件
    print("\n验证part文件...")
    if not validate_part_files(part_files):
        return 1

    # 合并文件
    validate_json = not args.no_validate
    merge_success = merge_files(part_files, args.output_file, validate_json)

    # 生成报告
    report_content = generate_report(part_files, args.output_file, merge_success)
    print(f"\n{report_content}")

    # 保存报告文件
    if args.report_file:
        try:
            with open(args.report_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"\n报告已保存到: {args.report_file}")
        except Exception as e:
            print(f"保存报告失败: {e}")

    if merge_success:
        print(f"\n✓ 合并完成! 输出文件: {args.output_file}")
        return 0
    else:
        print(f"\n✗ 合并失败!")
        return 1


if __name__ == "__main__":
    exit(main())
