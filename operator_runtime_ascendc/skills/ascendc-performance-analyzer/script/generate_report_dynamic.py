#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import logging
import os
import re
import glob
import argparse
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _is_valid_table_row(line):
    """Return True if line is a valid data row in a markdown table."""
    return line.startswith('|') and 'Level' not in line and '---' not in line and len(line) > 5


def extract_table_data(trace_file_path):
    """
    核心逻辑：解析单个 trace.md，只提取纯数据行
    """
    if not os.path.exists(trace_file_path):
        return []

    try:
        with open(trace_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        # 如果文件不存在或读不到，直接返回空列表，不报错
        return []

    # 1. 定位表格区域
    # 只要找到这个标题，我们就认为下面跟着表格
    start_idx = content.find('## 汇总表报告')
    if start_idx == -1:
        return [] # 没找到标题，直接跳过，不产生空行

    # 截取标题之后的内容
    section_content = content[start_idx:]
    lines = section_content.split('\n')
    
    valid_rows = []
    
    # 2. 逐行扫描
    for line in lines:
        line = line.strip()
        
        # 核心过滤逻辑：
        # A. 必须是表格行 (以 | 开头)
        # B. 不能是表头 (不包含 Level)
        # C. 不能是分隔线 (不包含 ---)  <-- 这一步是去除空行的关键！
        # D. 不能是空行 (防止提取到纯回车)
        if _is_valid_table_row(line):
            valid_rows.append(line)
            
    return valid_rows


def _parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 算子性能汇总工具 - 命令行版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python generate_report_dynamic.py -i /path/to/output_0410_l2_ziji
  python generate_report_dynamic.py -i ./my_ops -o ./result/batch_report.md
        """
    )
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='【必填】算子输出根目录路径 (例如: /home/user/output_0410_l2_ziji)'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='batch_report.md',
        help='【可选】汇总报告保存路径 (默认: batch_report.md)'
    )
    return parser.parse_args()


def _collect_trace_data(base_dir):
    search_pattern = os.path.join(base_dir, '*', 'trace.md')
    trace_files = sorted(glob.glob(search_pattern))

    if not trace_files:
        logger.warning("在 %s 下未找到任何算子 trace.md 文件", base_dir)
        logger.warning("   请检查目录结构是否包含 '子目录/trace.md'")
        return []

    logger.info("正在扫描目录: %s", base_dir)
    logger.info("找到 %d 个算子报告，正在处理...", len(trace_files))

    all_rows = []
    for file in trace_files:
        rows = extract_table_data(file)
        if rows:
            all_rows.extend(rows)
    return all_rows


def _write_report(output_path, all_rows):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    header = [
        "| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | "
        "PyTorch 参考延迟 | 生成AscendC代码延迟 | 加速比 | 最终状态 | "
        "精度正确 | 性能0.6x pytorch | 性能0.8x pytorch |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("## 📊 算子批量测试汇总报告\n\n")
        f.write("\n".join(header) + "\n")
        f.write("\n".join(all_rows) + "\n")

    logger.info("处理完成！")
    logger.info("报告已保存至: %s", os.path.abspath(output_path))


def main():
    args = _parse_args()
    base_dir = args.input
    output_path = args.output

    if not os.path.exists(base_dir):
        logger.error("源目录不存在 -> %s", base_dir)
        sys.exit(1)
    if not os.path.isdir(base_dir):
        logger.error("输入的路径不是目录 -> %s", base_dir)
        sys.exit(1)

    all_rows = _collect_trace_data(base_dir)

    try:
        _write_report(output_path, all_rows)
    except Exception as e:
        logger.error("写入文件失败: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()