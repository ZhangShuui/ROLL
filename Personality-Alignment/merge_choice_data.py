#!/usr/bin/env python3
# filepath: /home/szhangfa/ROLL/Personality-Alignment/merge_choice_data.py

import json
import glob
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="合并任务数组生成的选择题数据")
    parser.add_argument(
        "--base_path", type=str, default="/project/hdtaccuracy/Personality-Alignment/choice_ver/v9", help="基础路径"
    )
    parser.add_argument(
        "--pattern", type=str, default="raw_choice_data_v9_hard_chunk_*_range_*.jsonl", help="文件名模式"
    )
    parser.add_argument("--output", type=str, default="raw_choice_data_v9_hard_merged.jsonl", help="输出文件名")
    parser.add_argument("--verify", action="store_true", help="执行详细验证")
    parser.add_argument("--check_only", action="store_true", help="只检查不合并")
    parser.add_argument("--force", action="store_true", help="强制合并，忽略缺失文件")
    return parser.parse_args()


class ChunkMerger:
    def __init__(self, base_path, pattern, output_file):
        self.base_path = Path(base_path)
        self.pattern = pattern
        self.output_file = self.base_path / output_file
        self.chunk_files = []
        self.merged_data = []

    def find_chunk_files(self):
        """查找所有chunk文件"""
        # 支持多个文件模式
        patterns = [self.pattern, "raw_choice_data_v9_hard_0_5000_chunk_*_range_*.jsonl"]  # 原有模式  # 新增模式

        all_files = []
        for pattern in patterns:
            pattern_path = self.base_path / pattern
            files = list(glob.glob(str(pattern_path)))
            all_files.extend(files)

        self.chunk_files = all_files

        # 按chunk编号和起始范围排序
        def extract_chunk_info(filename):
            # 匹配原有格式: chunk_XXX_range_START_END.jsonl
            match = re.search(r"chunk_(\d+)_range_(\d+)_(\d+)", filename)
            if match:
                chunk_num = int(match.group(1))
                start_range = int(match.group(2))
                end_range = int(match.group(3))
                return (start_range, chunk_num, end_range)  # 改为以start_range为主要排序键

            # 匹配新格式: 0_5000_chunk_XXX_range_START_END.jsonl
            new_match = re.search(r"0_5000_chunk_(\d+)_range_(\d+)_(\d+)", filename)
            if new_match:
                chunk_num = int(new_match.group(1))
                start_range = int(new_match.group(2))
                end_range = int(new_match.group(3))
                return (start_range, chunk_num, end_range)

            return (999999, 0, 0)  # 无法解析的文件放在最后

        # 按起始范围排序，确保数据连续性
        self.chunk_files.sort(key=extract_chunk_info)
        return self.chunk_files

    def check_completeness(self):
        """检查文件完整性"""
        print("=== 文件完整性检查 ===")

        if not self.chunk_files:
            print("❌ 没有找到匹配的文件")
            return False, []

        chunk_info = []
        total_expected = 0

        for chunk_file in self.chunk_files:
            chunk_name = os.path.basename(chunk_file)

            # 匹配原有格式
            match = re.search(r"chunk_(\d+)_range_(\d+)_(\d+)", chunk_name)
            # 匹配新格式
            new_match = re.search(r"0_5000_chunk_(\d+)_range_(\d+)_(\d+)", chunk_name)

            if match or new_match:
                if new_match:
                    chunk_num = int(new_match.group(1))
                    start_range = int(new_match.group(2))
                    end_range = int(new_match.group(3))
                    file_type = "0_5000"
                else:
                    chunk_num = int(match.group(1))
                    start_range = int(match.group(2))
                    end_range = int(match.group(3))
                    file_type = "regular"

                size = Path(chunk_file).stat().st_size

                # 计算文件行数
                try:
                    with open(chunk_file, "r", encoding="utf-8") as f:
                        lines = sum(1 for _ in f)
                except Exception as e:
                    lines = 0
                    print(f"  ❌ 读取文件失败: {chunk_name} - {e}")
                    continue

                expected_lines = end_range - start_range
                total_expected += expected_lines

                chunk_info.append(
                    {
                        "chunk_num": chunk_num,
                        "start": start_range,
                        "end": end_range,
                        "file": chunk_file,
                        "size": size,
                        "lines": lines,
                        "expected_lines": expected_lines,
                        "type": file_type,
                    }
                )

                status = "✓" if lines > 0 else "❌"
                print(
                    f"  {status} [{file_type}] chunk_{chunk_num:03d}: {start_range}-{end_range} "
                    f"({lines:,}/{expected_lines:,} lines, {size:,} bytes)"
                )
            else:
                print(f"  ❌ 无法解析文件名: {chunk_name}")

        print(f"\n统计:")
        print(f"  找到文件: {len(chunk_info)} 个")
        print(f"  预期总数据: {total_expected:,} 条")

        # 检查范围连续性
        chunk_info.sort(key=lambda x: x["start"])

        coverage_issues = []
        for i in range(len(chunk_info) - 1):
            current_end = chunk_info[i]["end"]
            next_start = chunk_info[i + 1]["start"]

            if current_end != next_start:
                gap = next_start - current_end
                if gap > 0:
                    coverage_issues.append(f"缺失范围 {current_end}-{next_start} (缺失{gap}条)")
                elif gap < 0:
                    overlap = current_end - next_start
                    coverage_issues.append(f"重叠范围 {next_start}-{current_end} (重叠{overlap}条)")

        if coverage_issues:
            print(f"  ❌ 覆盖问题:")
            for issue in coverage_issues:
                print(f"    - {issue}")
            return False, coverage_issues
        else:
            print(f"  ✓ 范围覆盖连续")

        # 检查是否覆盖了预期的总范围 (0-19161)
        if chunk_info:
            first_start = chunk_info[0]["start"]
            last_end = chunk_info[-1]["end"]
            print(f"  实际覆盖范围: {first_start} - {last_end}")

            if first_start != 0:
                print(f"  ❌ 缺少开头数据: 0-{first_start}")
                return False, [f"missing start: 0-{first_start}"]

            expected_total = 19161
            if last_end < expected_total:
                print(f"  ❌ 缺少结尾数据: {last_end}-{expected_total}")
                return False, [f"missing end: {last_end}-{expected_total}"]

        return True, []

    def merge_files(self):
        """合并文件"""
        print("=== 开始合并 ===")

        total_lines = 0
        file_stats = []

        for chunk_file in self.chunk_files:
            chunk_name = os.path.basename(chunk_file)
            print(f"处理: {chunk_name}")

            try:
                lines_count = 0
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue

                        try:
                            data = json.loads(line)
                            self.merged_data.append(data)
                            lines_count += 1
                        except json.JSONDecodeError as e:
                            print(f"  警告: 第{line_num}行JSON错误: {e}")
                            print(f"  内容: {line[:100]}...")

                total_lines += lines_count
                file_stats.append({"file": chunk_name, "lines": lines_count, "size": os.path.getsize(chunk_file)})
                print(f"  成功: {lines_count:,} 行")

            except Exception as e:
                print(f"  错误: {e}")

        print(f"\n合并完成: {total_lines:,} 条数据")
        return file_stats

    def sort_data(self):
        """排序数据"""
        if self.merged_data and "original_index" in self.merged_data[0]:
            print("按original_index排序...")
            self.merged_data.sort(key=lambda x: x.get("original_index", 0))
            print("✓ 排序完成")
        else:
            print("警告: 没有找到original_index字段，保持原有顺序")

    def save_merged_data(self):
        """保存合并后的数据"""
        print(f"保存到: {self.output_file}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f:
            for data in self.merged_data:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        size = self.output_file.stat().st_size
        print(f"✓ 保存完成: {len(self.merged_data):,} 行, {size:,} 字节")

    def verify_data(self):
        """验证数据质量"""
        print("\n=== 数据验证 ===")

        if not self.merged_data:
            print("❌ 没有数据")
            return False

        # 基本统计
        print(f"总数据量: {len(self.merged_data):,}")

        # 检查必需字段
        required_fields = ["qid", "prompt", "output"]
        sample = self.merged_data[0]

        print(f"数据字段: {list(sample.keys())}")

        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            print(f"❌ 缺少必需字段: {missing_fields}")
            return False
        else:
            print(f"✓ 包含所有必需字段: {required_fields}")

        # 检查干扰项字段
        distractor_modes = [
            "style_violation",
            "topic_violation",
            "richness_violation",
            "free_violation",
            "profile_violation_w",
            "conversation_violation_w",
            "both_violation_w",
            "profile_violation_w/o",
            "conversation_violation_w/o",
            "both_violation_w/o",
        ]

        available_modes = []
        for mode in distractor_modes:
            field = f"{mode}_distractor"
            if field in sample:
                available_modes.append(mode)

        print(f"✓ 可用干扰项模式: {len(available_modes)}/10")
        for mode in available_modes:
            print(f"  - {mode}")

        if len(available_modes) < 10:
            missing_modes = [mode for mode in distractor_modes if f"{mode}_distractor" not in sample]
            print(f"❌ 缺失的干扰项模式: {missing_modes}")

        # 检查QID唯一性
        qids = [item["qid"] for item in self.merged_data]
        unique_qids = len(set(qids))

        if unique_qids == len(qids):
            print(f"✓ QID唯一性: {unique_qids:,} 个唯一QID")
        else:
            duplicates = len(qids) - unique_qids
            print(f"❌ QID重复: {duplicates:,} 个重复")

            # 显示重复的QID
            qid_counts = Counter(qids)
            duplicated_qids = [qid for qid, count in qid_counts.items() if count > 1]
            print(f"  重复的QID (前10个): {duplicated_qids[:10]}")

        # 检查索引连续性
        if "original_index" in sample:
            indices = [item["original_index"] for item in self.merged_data]
            min_idx, max_idx = min(indices), max(indices)
            expected_range = max_idx - min_idx + 1

            print(f"✓ 索引范围: {min_idx} - {max_idx} (跨度: {expected_range:,})")

            unique_indices = len(set(indices))
            if unique_indices == expected_range == len(indices):
                print("✓ 索引连续完整")
            else:
                if unique_indices != len(indices):
                    duplicates = len(indices) - unique_indices
                    print(f"❌ 索引重复: {duplicates} 个重复索引")

                if unique_indices != expected_range:
                    missing_count = expected_range - unique_indices
                    print(f"❌ 索引不连续: 缺失 {missing_count} 个")

        # 检查数据质量
        self._check_data_quality()

        return True

    def _check_data_quality(self):
        """检查数据质量"""
        print("\n--- 数据质量检查 ---")

        # 检查空值
        empty_outputs = sum(1 for item in self.merged_data if not item.get("output", "").strip())
        empty_prompts = sum(1 for item in self.merged_data if not item.get("prompt", "").strip())

        print(f"空output: {empty_outputs}")
        print(f"空prompt: {empty_prompts}")

        # 检查干扰项质量
        distractor_fields = [f for f in self.merged_data[0].keys() if f.endswith("_distractor")]

        for field in distractor_fields:
            empty_distractors = sum(
                1
                for item in self.merged_data
                if not item.get(field, "").strip()
                or item.get(field, "").startswith(("Failed_", "Missing_", "GPT_API_ERROR", "FUTURE_ERROR"))
            )
            success_rate = ((len(self.merged_data) - empty_distractors) / len(self.merged_data)) * 100
            print(f"{field}: {empty_distractors} 个问题项 (成功率: {success_rate:.1f}%)")

    def show_summary(self):
        """显示处理摘要"""
        print(f"\n=== 处理摘要 ===")

        # 文件信息
        chunk_ranges = []
        for chunk_file in self.chunk_files:
            match = re.search(r"range_(\d+)_(\d+)", os.path.basename(chunk_file))
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                chunk_ranges.append((start, end))

        if chunk_ranges:
            chunk_ranges.sort()
            print(f"处理的chunk数量: {len(chunk_ranges)}")
            print(f"数据范围: {chunk_ranges[0][0]} - {chunk_ranges[-1][1]}")
            total_expected = sum(end - start for start, end in chunk_ranges)
            print(f"预期数据量: {total_expected:,}")
            print(f"实际数据量: {len(self.merged_data):,}")

            if len(self.merged_data) == total_expected:
                print("✓ 数据量匹配")
            else:
                diff = len(self.merged_data) - total_expected
                print(f"❌ 数据量差异: {diff:+,}")


def main():
    args = parse_args()

    print("任务数组结果合并工具")
    print("=" * 50)
    print(f"基础路径: {args.base_path}")
    print(f"文件模式: {args.pattern}")
    print(f"输出文件: {args.output}")
    print("")

    merger = ChunkMerger(args.base_path, args.pattern, args.output)

    # 查找文件
    chunk_files = merger.find_chunk_files()
    print(f"找到 {len(chunk_files)} 个chunk文件")

    if not chunk_files:
        print("❌ 没有找到匹配的文件，请检查路径和文件模式")
        return

    # 显示找到的文件
    print("找到的文件:")
    for f in chunk_files:
        print(f"  - {os.path.basename(f)}")

    # 检查完整性
    is_complete, missing = merger.check_completeness()

    if not is_complete and not args.force:
        print(f"\n发现问题: {len(missing) if missing else '未知'}")
        if not args.check_only:
            response = input("是否继续合并? (y/N): ").strip().lower()
            if response != "y":
                print("合并取消")
                return

    if args.check_only:
        print("仅检查模式，不执行合并")
        return

    # 执行合并
    file_stats = merger.merge_files()
    merger.sort_data()
    merger.save_merged_data()

    # 显示摘要
    merger.show_summary()

    # 验证数据
    if args.verify:
        merger.verify_data()

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
