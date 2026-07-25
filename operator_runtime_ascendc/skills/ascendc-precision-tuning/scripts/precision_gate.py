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

"""
precision_gate.py — 精度调优 Gate 结构化验证 + 链式前置检查 + 循环控制

核心设计:
  每个 Gate 不仅检查当前步骤的产物, 还验证前序步骤是否已完成。
  即使 Agent 试图跳步, Gate 脚本会拒绝 (返回码 ≠ 0)。

链式依赖:
  Gate-F (forensics)  → 无前置依赖
  Gate-A (audit)      → 前置: forensics_report 存在且 attempt 匹配
  Gate-PRE-EDIT       → 前置: 6-section audit 报告完成（修改 kernel 前的最后防线）
  Gate-X (fix)        → 前置: precision_audit_{attempt}.md 存在
  Gate-V (validate)   → 前置: 代码已修复 (kernel 存在)

循环控制:
  Gate-V 额外输出 loop_signal: PASS / CONTINUE / STOP
  Agent 无权覆盖此信号。

用法:
    python3 precision_gate.py --step <step> --op-name <n> --output-path <path> --attempt <N>

返回码: 0=通过, 1=未通过(可重试), 2=致命错误(前置缺失)
"""

import argparse
import json
import logging
import os
import re
from typing import NamedTuple
import sys


MAX_ATTEMPTS = 6  # 与 CLAUDE.md max_d_iterations 一致
MAX_STAGNANT_ROUNDS = 2


logger = logging.getLogger(__name__)


class TuningSummaryMeta(NamedTuple):
    """封装 _read_tuning_summary_meta 的返回值."""
    fix_type: str | None
    direction_verdict: str | None
    forensics_hint: str | None
    improvement_ratio: float | None
    absolute_improvement: float | None
    match_rate: float | None
    mismatch_ratio: float | None


class GateChecker:

    def __init__(self, op_name: str, output_path: str, attempt: int = 0,
                 project_type: str = "openops"):
        self.op_name = op_name
        self.output_path = output_path
        self.attempt = attempt
        self.project_type = project_type
        self.tuning_dir = os.path.join(output_path, "precision_tuning")

    # ================================================================
    # @staticmethod 方法
    # ================================================================

    @staticmethod
    def _count_stagnant(trend: list) -> int:
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 2:
            return 0
        count = 0
        for i in range(len(ratios) - 1, 0, -1):
            if ratios[i] >= ratios[i - 1]:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _detect_harmful_regression(trend: list) -> bool:
        """
        检测 A→B→A 振荡型有害回退。
        条件: 最近3轮中，中间轮 mismatch 改善 > 1%（绝对），当前轮回退至 ≥ 初始轮 - 0.5%。
        """
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 3:
            return False
        r_prev, r_mid, r_curr = ratios[-3], ratios[-2], ratios[-1]
        mid_improved = (r_prev - r_mid) > 0.01      # 中间轮改善 > 1%
        curr_regressed = r_curr >= (r_prev - 0.005)  # 当前轮回退至接近初始水平
        return mid_improved and curr_regressed

    @staticmethod
    def _compute_improvement_ratio(prev_mismatch: float, curr_mismatch: float):
        """
        计算本轮改善比率（剩余可修复空间中的改善幅度）。
        improvement_ratio = (curr_match_rate - prev_match_rate) / (100 - prev_match_rate)
        返回 None 如果上一轮 match_rate 已为 100%（无剩余空间）。
        """
        prev_match = (1 - prev_mismatch) * 100
        curr_match = (1 - curr_mismatch) * 100
        remaining = 100 - prev_match
        if remaining <= 0:
            return None
        return round((curr_match - prev_match) / remaining, 4)

    @staticmethod
    def _extract_direction_first_word(text: str) -> str:
        """从答案文本中提取首个判断词，去除尾部标点干扰。
        例: "否，已换方向" → "否"；"是（继续分析）" → "是（继续分析）".rstrip → "是"
        """
        if not text:
            return ""
        first = text.split()[0] if text.split() else text
        return first.rstrip("，。！？、；：…,.")

    @staticmethod
    def _extract_section(content: str, section_name: str):
        """提取 precision_audit_{attempt}.md 中指定 section 的内容。
        边界: [SECTION_NAME] 到下一个 \\n[ 或 === END AUDIT === 结束。
        返回 None 如果 section 不存在（不阻断 Gate-A 通过）。
        """
        marker = f"[{section_name}]"
        start = content.find(marker)
        if start == -1:
            return None
        start += len(marker)
        end_marker = content.find("\n[", start)
        end_audit = content.find("=== END AUDIT ===", start)
        candidates = [pos for pos in [end_marker, end_audit] if pos != -1]
        end = min(candidates) if candidates else len(content)
        text = content[start:end].strip()
        return text if text else None

    @staticmethod
    def _result(gate_name: str, checks: dict) -> dict:
        return {"gate": gate_name, "passed": all(checks.values()), "checks": checks}

    @staticmethod
    def _check_nearly_success(match_rate_str) -> tuple | None:
        if match_rate_str is None:
            return None
        try:
            mr = float(match_rate_str)
            if mr >= 99.0:
                return (
                    "NEARLY_SUCCESS",
                    f"精度接近通过 (match_rate={mr:.2f}%), 可能为量化误差或精度阈值过严，建议人工确认后决定是否接受",
                    "nearly_success",
                )
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _resolve_outcome(stop_reason_code: str, improvement_ratio: float | None) -> str:
        """根据 stop_reason 和 improvement_ratio 判断 outcome."""
        if stop_reason_code in ("precision_passed", "nearly_success"):
            return "passed"
        if improvement_ratio is None:
            return "stagnant"
        if improvement_ratio < -0.05:
            return "regressed"
        if improvement_ratio >= 0.1:
            return "improved"
        return "stagnant"

    @staticmethod
    def _parse_line_field(section: str, field_name: str) -> str | None:
        for line in section.split("\n"):
            if field_name not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            val = line[colon_pos + 1:].strip()
            if val:
                return val
        return None

    @staticmethod
    def _read_file(path: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            return ""

    @staticmethod
    def _load_match_rate(result_path: str) -> tuple[float | None, float | None]:
        """从 validation_result 文件读取 match_rate 和 mismatch_ratio."""
        if not os.path.exists(result_path):
            return None, None
        try:
            with open(result_path) as f:
                r = json.load(f)
            mr_str = r.get("match_rate")
            if mr_str is not None:
                match_rate = round(float(mr_str), 4)
                mismatch_ratio = round(1 - match_rate / 100, 8)
                return match_rate, mismatch_ratio
        except (KeyError, OSError, ValueError):
            pass
        return None, None

    @staticmethod
    def _write_one_section(sections_dir: str, key: str, tag: str,
                           sec_text: str, rel_path: str) -> str | None:
        """将单个 section 写入文件，返回相对路径或 None."""
        abs_path = os.path.join(sections_dir, f"{key}.md")
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(f"[{tag}]\n\n{sec_text}\n")
            return rel_path
        except OSError:
            return None

    # ================================================================
    # 公共方法
    # ================================================================

    # ---- Gate-F: 取证报告 (无前置依赖) ----

    def check_forensics(self) -> dict:
        # AscendOpGenAgent: forensics comes from parsing evaluate_ascendc.sh output,
        # not from precision_forensics.py. Gate-F is non-blocking in this mode.
        if self.project_type == "ascendopgen":
            return {"gate": "GATE-F", "passed": True,
                    "checks": {"ascendopgen_skip": True},
                    "note": "AscendOpGenAgent: forensics from evaluate_ascendc.sh, Gate-F auto-pass"}
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        checks = {
            "report_exists": os.path.exists(path),
            "report_parseable": False,
            "status_completed": False,
            "has_primary_hint": False,
            "has_outputs": False,
            "has_basic_stats": False,
            "attempt_matches": False,
        }
        r = None
        if checks["report_exists"]:
            try:
                with open(path) as f:
                    r = json.load(f)
                checks["report_parseable"] = True
                checks["status_completed"] = r.get("status") == "completed"
                checks["has_primary_hint"] = bool(r.get("primary_hint"))
                checks["has_outputs"] = len(r.get("outputs", [])) > 0
                if checks["has_outputs"]:
                    checks["has_basic_stats"] = "basic_stats" in r["outputs"][0]
                # 验证 attempt 号匹配 (防止用旧报告)
                checks["attempt_matches"] = r.get("attempt", -1) == self.attempt
            except (json.JSONDecodeError, KeyError):
                r = None

        gate_result = self._result("GATE-F", checks)

        # Gate-F 通过且为 attempt 0 时：从 forensics outputs 写 baseline_state.json
        # 此时代码尚未被修改，forensics 中的精度数据就是真正的 baseline
        # 仅当 baseline_state.json 不存在时写入（幂等）
        if gate_result["passed"] and self.attempt == 0 and r is not None:
            self._write_baseline_from_forensics(r)

        return gate_result

    # ---- Gate-PRE-EDIT: 修改 kernel 前的最后防线 ----

    SECTION_TAGS = [
        ("FORENSICS_SUMMARY", "has_forensics_summary"),
        ("COMPUTATION_DECOMPOSITION", "has_computation_decomposition"),
        ("REFERENCE_IMPL_SPEC", "has_reference_impl_spec"),
        ("KERNEL_STEP_TRACE", "has_kernel_step_trace"),
        ("ROOT_CAUSE", "has_root_cause"),
        ("FIX_PLAN", "has_fix_plan"),
        ("TARGET_FILES", "has_target_files"),
    ]

    def check_pre_edit(self) -> dict:
        """
        ⛔ PRE-EDIT GATE — 在修改 kernel 代码之前必须通过此检查。

        验证 precision-tuning 审计是否完成（6-section audit report 存在且完整）。
        若未完成 → 返回码非零 → agent 禁止使用 Edit/Write 修改 kernel 文件。
        """
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")

        checks = {
            "gate_activated": True,
            "precision_tuning_dir_exists": os.path.isdir(self.tuning_dir),
            "audit_report_exists": os.path.exists(path),
            "audit_report_nonempty": False,
            **{key: False for _, key in self.SECTION_TAGS},
        }

        all_sections_present = False
        if checks["audit_report_exists"]:
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                all_sections_present = self._check_audit_sections(content, checks)
            except (OSError, UnicodeDecodeError):
                pass

        gate_result = {
            "gate": "GATE-PRE-EDIT",
            "passed": all_sections_present,
            "checks": checks,
        }

        if not gate_result["passed"]:
            missing = [k for k, v in checks.items() if not v]
            gate_result["error"] = (
                f"⛔ 审计报告未完成！缺少: {missing}\n"
                f"   在修改 {self.output_path}/kernel/ 下的任何文件之前，\n"
                f"   必须先完成 precision-tuning 的 6-section 审计报告：\n"
                f"   {path}\n"
                f"   步骤: (1) Gate-A audit check (2) precision_knowledge.py search (3) 写入完整的 6-section 审计报告\n"
                f"   ⛔ 禁止跳过 precision-tuning 直接修改 kernel 代码！"
            )

        return gate_result

    # ---- Gate-A: 审计报告 — 前置: forensics 已完成 ----

    def check_audit(self) -> dict:
        # 链式前置检查: forensics 必须存在且 attempt 匹配
        prereq = self._check_prerequisite_forensics()
        if not prereq["satisfied"]:
            checks = {"prerequisite_forensics": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-A", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result

        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        checks = {
            "prerequisite_forensics": True,
            "report_exists": os.path.exists(path),
            "report_nonempty": False,
            "has_forensics_summary": False,
            "has_computation_decomposition": False,
            "has_reference_impl_spec": False,
            "has_kernel_step_trace": False,
            "has_root_cause": False,
            "has_fix_plan": False,
            "has_target_files": False,
            "has_direction_assessment": True,  # 第一轮可选，后续必填
        }
        content = None
        if checks["report_exists"]:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            checks["report_nonempty"] = len(content) > 200
            for tag, key in [("FORENSICS_SUMMARY", "has_forensics_summary"),
                             ("COMPUTATION_DECOMPOSITION", "has_computation_decomposition"),
                             ("REFERENCE_IMPL_SPEC", "has_reference_impl_spec"),
                             ("KERNEL_STEP_TRACE", "has_kernel_step_trace"),
                             ("ROOT_CAUSE", "has_root_cause"),
                             ("FIX_PLAN", "has_fix_plan"),
                             ("TARGET_FILES", "has_target_files")]:
                checks[key] = f"[{tag}]" in content
            # DIRECTION_ASSESSMENT 仅在 attempt > 0 时要求
            checks["has_direction_assessment"] = (
                self.attempt == 0 or "[DIRECTION_ASSESSMENT]" in content
            )
            # 二值格式校验：attempt > 0 时额外验证答案为 "是" 或 "否"（防止 Agent 填写模糊内容）
            if self.attempt > 0:
                if "[DIRECTION_ASSESSMENT]" in content:
                    checks["direction_assessment_binary"] = self._validate_direction_binary(content)
                else:
                    checks["direction_assessment_binary"] = False

        gate_result = self._result("GATE-A", checks)

        # Gate-A 通过后：自动提取 sections + 写 round_summary 初始字段（diagnostics + index）
        if gate_result["passed"] and content:
            self._write_audit_index(content)

        return gate_result

    # ---- Gate-X: 代码完整性 — 前置: audit 已完成 ----

    def check_fix(self) -> dict:
        # 链式前置检查: audit 必须存在
        prereq = self._check_prerequisite_audit()
        if not prereq["satisfied"]:
            checks = {"prerequisite_audit": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-X", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result

        project_dir = self._find_project_dir()
        checks = {
            "prerequisite_audit": True,
            "project_dir_exists": project_dir is not None,
            "kernel_exists": False,
            "kernel_nonempty": False,
            "host_exists": False,
        }
        if project_dir:
            if self.project_type == "ascendopgen":
                kp = os.path.join(project_dir, "op_kernel", f"{self.op_name.lower()}.cpp")
                hp = os.path.join(project_dir, "op_host", f"{self.op_name.lower()}.cpp")
            else:
                kp = os.path.join(project_dir, "op_kernel", f"{self.op_name.lower()}_custom.cpp")
                hp = os.path.join(project_dir, "op_host", f"{self.op_name.lower()}_custom.cpp")
            checks["kernel_exists"] = os.path.exists(kp)
            if checks["kernel_exists"]:
                checks["kernel_nonempty"] = os.path.getsize(kp) > 100
            checks["host_exists"] = os.path.exists(hp)
        return self._result("GATE-X", checks)

    # ---- Gate-V: 验证结果 + 循环控制 — 前置: 代码已修复 ----

    def check_validate(self) -> dict:
        # 链式前置检查: 代码文件必须存在
        prereq = self._check_prerequisite_code()
        if not prereq["satisfied"]:
            checks = {"prerequisite_code": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-V", checks)
            result["prerequisite_error"] = prereq["reason"]
            result["loop_signal"] = "STOP"
            result["loop_reason"] = f"前置条件不满足: {prereq['reason']}"
            result["stop_reason_code"] = "prerequisite_failure"
            return result

        result_path = os.path.join(self.tuning_dir,
                                   f"validation_result_attempt_{self.attempt}.json")
        checks = {
            "prerequisite_code": True,
            "result_exists": os.path.exists(result_path),
            "result_parseable": False,
            "precision_passed": False,
        }

        correctness_passed = False
        match_rate_str = None
        if checks["result_exists"]:
            try:
                with open(result_path) as f:
                    r = json.load(f)
                checks["result_parseable"] = True
                correctness_passed = r.get("correctness_passed", False)
                checks["precision_passed"] = correctness_passed
                match_rate_str = r.get("match_rate")
            except (json.JSONDecodeError, KeyError):
                pass

        loop_signal, loop_reason, stop_reason_code = self._compute_loop_signal(
            correctness_passed, match_rate_str
        )

        gate_result = self._result("GATE-V", checks)
        gate_result["loop_signal"] = loop_signal
        gate_result["loop_reason"] = loop_reason
        gate_result["stop_reason_code"] = stop_reason_code
        gate_result["attempt"] = self.attempt
        gate_result["max_attempts"] = MAX_ATTEMPTS

        # 写入 round_summary_{N}.json（合并 Agent 语义字段 + Gate 数值字段）
        self._write_round_summary(stop_reason_code)

        # 追加本轮方向记录到 tuning_directions.json（读 round_summary 获取 diagnostics）
        self._write_tuning_directions(stop_reason_code)

        return gate_result

    # ================================================================
    # 保护/私有方法
    # ================================================================

    def _check_audit_sections(self, content: str, checks: dict) -> bool:
        """检查审计报告各 section 是否存在，返回 all_sections_present."""
        checks["audit_report_nonempty"] = len(content) > 200
        for tag, key in self.SECTION_TAGS:
            checks[key] = f"[{tag}]" in content
        return all(checks[key] for _, key in self.SECTION_TAGS)

    # ---- baseline 写入 ----

    def _write_baseline_from_forensics(self, forensics: dict) -> None:
        """
        从 forensics_report 的 outputs[0].basic_stats 提取精度数据，
        写入 baseline_state.json。

        只在 baseline_state.json 不存在时写入（幂等），确保 baseline
        永远记录第一次 Gate-F 时代码未修改的原始状态。
        """
        baseline_path = os.path.join(self.tuning_dir, "baseline_state.json")
        if os.path.exists(baseline_path):
            return  # 已存在，不覆盖

        try:
            outputs = forensics.get("outputs", [])
            if not outputs:
                return
            stats = outputs[0].get("basic_stats", {})
            raw_match_rate = stats.get("match_rate")
            raw_mismatch_ratio = stats.get("mismatch_ratio")
            if raw_match_rate is None:
                return

            # forensics 里的 match_rate 单位是 0~1 的比例（非百分比）
            # 需要乘以 100 转换为百分比
            baseline_match_rate = round(float(raw_match_rate) * 100, 4)
            baseline_mismatch_ratio = float(raw_mismatch_ratio) if raw_mismatch_ratio is not None else None

            baseline_state = {
                "match_rate": baseline_match_rate,
                "mismatch_ratio": baseline_mismatch_ratio,
                "max_abs_diff": stats.get("max_abs_diff"),
                "mean_abs_diff": stats.get("mean_abs_diff"),
                "primary_hint": forensics.get("primary_hint"),
                "source": "forensics_report_0.json/outputs[0]/basic_stats",
                "note": "Initial precision captured at Gate-F before any code modification"
            }
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline_state, f, indent=2, ensure_ascii=False)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # ---- 前置依赖检查 ----

    def _check_prerequisite_forensics(self) -> dict:
        """检查 forensics_report_{attempt}.json 存在且 attempt 匹配"""
        if self.project_type == "ascendopgen":
            return {"satisfied": True, "reason": "ascendopgen: forensics from evaluate_ascendc.sh", "detail": {}}
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if not os.path.exists(path):
            return {"satisfied": False,
                    "reason": f"forensics_report_{self.attempt}.json 不存在, 必须先运行 precision_forensics.py",
                    "detail": {"forensics_exists": False, "forensics_attempt_match": False}}
        try:
            with open(path) as f:
                r = json.load(f)
            if r.get("status") != "completed":
                return {"satisfied": False,
                        "reason": f"forensics 状态异常: {r.get('status')}",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            if r.get("attempt", -1) != self.attempt:
                return {"satisfied": False,
                        "reason": f"forensics attempt={r.get('attempt')} 不匹配当前 attempt={self.attempt}, "
                                  f"必须重新运行 precision_forensics.py",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            return {"satisfied": True, "reason": "", "detail": {}}
        except (json.JSONDecodeError, KeyError) as e:
            return {"satisfied": False, "reason": f"forensics 解析失败: {e}",
                    "detail": {"forensics_exists": True, "forensics_attempt_match": False}}

    def _check_prerequisite_audit(self) -> dict:
        """检查 precision_audit_{attempt}.md 存在"""
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return {"satisfied": False,
                    "reason": f"precision_audit_{self.attempt}.md 不存在, 必须先完成审计 (Step 2)",
                    "detail": {"audit_exists": False}}
        if os.path.getsize(path) < 100:
            return {"satisfied": False,
                    "reason": f"precision_audit_{self.attempt}.md 内容过少, 审计可能未完成",
                    "detail": {"audit_exists": True}}
        return {"satisfied": True, "reason": "", "detail": {}}

    def _check_prerequisite_code(self) -> dict:
        """检查代码文件存在"""
        project_dir = self._find_project_dir()
        if not project_dir:
            return {"satisfied": False,
                    "reason": "找不到项目目录",
                    "detail": {"project_exists": False}}
        if self.project_type == "ascendopgen":
            kp = os.path.join(project_dir, "op_kernel", f"{self.op_name.lower()}.cpp")
        else:
            kp = os.path.join(project_dir, "op_kernel", f"{self.op_name.lower()}_custom.cpp")
        if not os.path.exists(kp) or os.path.getsize(kp) < 100:
            return {"satisfied": False,
                    "reason": f"{kp} 不存在或为空, 必须先完成代码修复 (Step 3)",
                    "detail": {"kernel_exists": False}}
        return {"satisfied": True, "reason": "", "detail": {}}

    # ---- 循环控制 ----

    def _check_forensics_stagnation(self) -> tuple | None:
        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if not os.path.exists(forensics_path):
            return None
        try:
            with open(forensics_path) as f:
                fr = json.load(f)
        except (json.JSONDecodeError, KeyError):
            return None
        trend = fr.get("history_trend")
        if not trend:
            return None
        trend_list = trend.get("trend", [])
        if self._detect_harmful_regression(trend_list):
            return "STOP", "检测到 A→B→A 振荡型有害回退，需人工分析", "harmful_regression"
        if not trend.get("mismatch_improving", True):
            stagnant = self._count_stagnant(trend_list)
            if stagnant >= MAX_STAGNANT_ROUNDS:
                if self._check_direction_assessment() == "continue":
                    return "CONTINUE", (
                        f"mismatch 连续 {stagnant} 轮未改善, "
                        f"但 Agent 已明确换方向，继续探索"
                    ), "stagnant_new_direction"
                return "STOP", (
                    f"mismatch 连续 {stagnant} 轮未改善, "
                    f"Agent 仍沿用同一方向，可能方向错误，需人工分析"
                ), "stagnant_same_direction"
        return None

    def _compute_loop_signal(self, passed: bool, match_rate_str=None) -> tuple:
        if passed:
            return "PASS", "精度验证通过", "precision_passed"

        nearly = self._check_nearly_success(match_rate_str)
        if nearly:
            return nearly

        if self.attempt + 1 >= MAX_ATTEMPTS:
            return "STOP", f"已达最大轮次 ({MAX_ATTEMPTS})", "max_attempts_reached"

        stagnation = self._check_forensics_stagnation()
        if stagnation:
            return stagnation

        return "CONTINUE", f"精度未通过, 进入第 {self.attempt + 2} 轮", None

    # ---- round_summary / tuning_directions 写入 ----

    def _try_rebuild_from_audit(self, summary_path: str, existing: dict) -> dict:
        """尝试从 audit 文件重建 summary."""
        audit_path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(audit_path):
            return existing
        try:
            with open(audit_path, encoding="utf-8") as f:
                self._write_audit_index(f.read())
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        return existing

    def _load_existing_summary(self, summary_path: str) -> dict:
        """加载已有 round_summary，必要时从 audit 文件自愈重建."""
        existing = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        if not existing.get("gate_a_complete"):
            existing = self._try_rebuild_from_audit(summary_path, existing)
        return existing

    def _compute_vs_baseline(self, match_rate: float) -> tuple[float, float]:
        """与 baseline 比较计算 improvement."""
        baseline_match_rate = self._get_baseline_match_rate()
        if baseline_match_rate is None:
            return None, None
        baseline_mismatch = 1 - baseline_match_rate / 100
        curr_mismatch = 1 - match_rate / 100
        improvement_ratio = self._compute_improvement_ratio(baseline_mismatch, curr_mismatch)
        absolute_improvement = round(match_rate - baseline_match_rate, 4)
        return improvement_ratio, absolute_improvement

    def _compute_vs_previous(self, match_rate: float) -> tuple[float | None, float | None]:
        """与上一 attempt 比较计算 improvement."""
        prev_result_path = os.path.join(
            self.tuning_dir, f"validation_result_attempt_{self.attempt - 1}.json"
        )
        if not os.path.exists(prev_result_path):
            return None, None
        try:
            with open(prev_result_path) as f:
                prev_r = json.load(f)
            prev_mr_str = prev_r.get("match_rate")
            if prev_mr_str is None:
                return None, None
            prev_match_rate = float(prev_mr_str)
            prev_mismatch = 1 - prev_match_rate / 100
            curr_mismatch = 1 - match_rate / 100
            improvement_ratio = self._compute_improvement_ratio(prev_mismatch, curr_mismatch)
            absolute_improvement = round(match_rate - prev_match_rate, 4)
            return improvement_ratio, absolute_improvement
        except (KeyError, OSError, ValueError):
            pass
        return None, None

    def _read_validation_metrics(self) -> tuple[float | None, float | None,
                                                float | None, float | None]:
        """从 validation_result 读取 match_rate 并计算 improvement."""
        result_path = os.path.join(self.tuning_dir, f"validation_result_attempt_{self.attempt}.json")
        match_rate, mismatch_ratio = self._load_match_rate(result_path)

        if match_rate is None:
            return None, None, None, None

        if self.attempt == 0:
            improvement_ratio, absolute_improvement = self._compute_vs_baseline(match_rate)
        else:
            improvement_ratio, absolute_improvement = self._compute_vs_previous(match_rate)
        return match_rate, mismatch_ratio, improvement_ratio, absolute_improvement

    def _read_forensics_meta(self) -> tuple[str | None, str | None]:
        """从 forensics_report 读取 hint 和 op_type 元数据."""
        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if not os.path.exists(forensics_path):
            return None, None
        try:
            with open(forensics_path) as f:
                fr = json.load(f)
            hint = fr.get("primary_hint")
            op_type = fr.get("op_type") or fr.get("L8_operator", {}).get("op_type")
            return hint, op_type
        except (json.JSONDecodeError, KeyError, OSError):
            return None, None

    def _write_round_summary(self, stop_reason_code) -> None:
        """Gate-V 调用：将 metrics 合并进 round_summary."""
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        if stop_reason_code is None:
            stop_reason_code = "validation_failed"

        existing = self._load_existing_summary(summary_path)
        match_rate, mismatch_ratio, improvement_ratio, absolute_improvement = \
            self._read_validation_metrics()
        forensics_hint, op_type = self._read_forensics_meta()

        compile_log_abs = os.path.join(self.tuning_dir, f"compilation_log_{self.attempt}.json")
        compilation_log_ref = (
            f"precision_tuning/compilation_log_{self.attempt}.json"
            if os.path.exists(compile_log_abs) else None
        )

        summary = dict(existing)
        summary["attempt"] = self.attempt
        summary["metrics"] = {
            **summary.get("metrics", {}),
            "match_rate": match_rate,
            "mismatch_ratio": mismatch_ratio,
            "improvement_ratio": improvement_ratio,
            "absolute_improvement": absolute_improvement,
            "stop_reason_code": stop_reason_code,
        }
        summary["diagnostics"] = {
            **summary.get("diagnostics", {}),
            "forensics_hint": forensics_hint,
            "op_type": op_type,
        }
        summary["index"] = {
            **summary.get("index", {}),
            "compilation_log": compilation_log_ref,
        }

        try:
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _write_tuning_directions(self, stop_reason_code) -> None:
        """Gate-V 调用：将本轮方向学习记录追加到 tuning_directions.json."""
        directions_path = os.path.join(self.tuning_dir, "tuning_directions.json")
        data = {"op_name": self.op_name, "final_status": "in_progress", "entries": []}
        if os.path.exists(directions_path):
            try:
                with open(directions_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        meta = self._read_tuning_summary_meta()
        fix_type = meta.fix_type
        direction_verdict = meta.direction_verdict
        forensics_hint = meta.forensics_hint
        improvement_ratio = meta.improvement_ratio
        absolute_improvement = meta.absolute_improvement
        match_rate = meta.match_rate
        mismatch_ratio = meta.mismatch_ratio

        outcome = self._resolve_outcome(stop_reason_code, improvement_ratio)
        direction_reason = self._extract_direction_reason()

        new_entry = {
            "attempt": self.attempt,
            "fix_type": fix_type,
            "forensics_hint": forensics_hint,
            "direction_verdict": direction_verdict,
            "direction_reason": direction_reason,
            "improvement_ratio": improvement_ratio,
            "absolute_improvement": absolute_improvement,
            "outcome": outcome,
            "evidence": {
                "forensics_ref": f"precision_tuning/forensics_report_{self.attempt}.json",
                "audit_ref": f"precision_tuning/precision_audit_{self.attempt}.md",
                "match_rate": match_rate,
                "mismatch_ratio": mismatch_ratio,
            }
        }

        data["entries"] = [e for e in data["entries"] if e.get("attempt") != self.attempt]
        data["entries"].append(new_entry)
        data["entries"].sort(key=lambda e: e.get("attempt", 0))
        self._apply_tuning_final_status(data, stop_reason_code, fix_type)

        try:
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(directions_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _read_tuning_summary_meta(self) -> TuningSummaryMeta:
        """从 round_summary 读取 diagnostics 和 metrics 元数据."""
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        meta = TuningSummaryMeta(None, None, None, None, None, None, None)

        if not os.path.exists(summary_path):
            return meta

        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            diag = summary.get("diagnostics", {})
            metrics = summary.get("metrics", {})
            meta = TuningSummaryMeta(
                fix_type=diag.get("fix_type"),
                direction_verdict=diag.get("direction_verdict"),
                forensics_hint=diag.get("forensics_hint"),
                improvement_ratio=metrics.get("improvement_ratio"),
                absolute_improvement=metrics.get("absolute_improvement"),
                match_rate=metrics.get("match_rate"),
                mismatch_ratio=metrics.get("mismatch_ratio"),
            )
        except (json.JSONDecodeError, OSError):
            pass

        return meta

    def _apply_tuning_final_status(self, data: dict, stop_reason_code: str,
                                    fix_type: str | None) -> None:
        """更新 final_status 并回溯 contributed 标注."""
        terminal_codes = {
            "max_attempts_reached", "stagnant_same_direction",
            "stagnant_new_direction", "harmful_regression", "prerequisite_failure",
            "nearly_success",
        }
        if stop_reason_code == "precision_passed":
            data["final_status"] = "success"
            for entry in data["entries"]:
                ir = entry.get("improvement_ratio")
                same_fix = entry.get("fix_type") == fix_type
                nonneg = ir is None or ir >= 0
                entry["contributed"] = same_fix and nonneg
            for entry in data["entries"]:
                if entry.get("attempt") == self.attempt:
                    entry["contributed"] = True
        elif stop_reason_code == "nearly_success":
            data["final_status"] = "nearly_success"
        elif stop_reason_code in terminal_codes:
            data["final_status"] = "failed"

    # ---- direction assessment 相关 ----

    def _extract_direction_reason(self) -> str | None:
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return None
        section = self._extract_section(self._read_file(path), "DIRECTION_ASSESSMENT")
        if not section:
            return None
        return self._parse_line_field(section, "换方向理由")

    def _check_direction_assessment(self) -> str:
        """
        读取当前轮 precision_audit_{attempt}.md 中的 [DIRECTION_ASSESSMENT] section，
        判断 Agent 是否在主动换方向。

        返回:
          "continue" — Agent 明确换了方向，可以继续探索
          "stop"     — Agent 仍沿用同一方向，停滞无改善，大概率方向错了
          "unknown"  — section 不存在或解析失败，保守返回 stop
        """
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return "unknown"

        try:
            with open(path) as f:
                content = f.read()

            marker = "[DIRECTION_ASSESSMENT]"
            start = content.find(marker)
            if start == -1:
                return "unknown"
            start += len(marker)
            next_bracket = content.find("\n[", start)
            section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()

            if not section:
                return "unknown"

            # 提取"本轮是否延续上一轮方向"字段，只检查冒号后的答案值
            for line in section.split("\n"):
                key = "本轮是否延续上一轮方向"
                if key not in line and "本轮是否延续" not in line:
                    continue
                # 提取冒号后的值并去除空白
                colon_pos = line.find(":")
                if colon_pos == -1:
                    continue
                answer = line[colon_pos + 1:].strip()
                # 检查答案值：必须精确匹配冒号后的词
                # 否/换方向/换了 → 换方向
                # 是 → 沿用方向
                first_word = self._extract_direction_first_word(answer)
                if first_word == "否":
                    return "continue"
                elif first_word == "是":
                    return "stop"
            return "unknown"
        except (OSError, UnicodeDecodeError):
            return "unknown"

    def _validate_direction_binary(self, content: str) -> bool:
        """验证 [DIRECTION_ASSESSMENT] section 中的答案是严格二值（是/否）。"""
        marker = "[DIRECTION_ASSESSMENT]"
        start = content.find(marker)
        if start == -1:
            return False
        start += len(marker)
        next_bracket = content.find("\n[", start)
        section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()
        for line in section.split("\n"):
            if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            first_word = self._extract_direction_first_word(line[colon_pos + 1:].strip())
            return first_word in ("是", "否")
        return False

    # ---- Gate-A: Section 提取与 round_summary 初始写入 ----

    def _extract_fix_type(self, content: str):
        """从 [FIX_PLAN] section 中提取 FIX_PRECISION_XXX 类型标识。"""
        section = self._extract_section(content, "FIX_PLAN")
        if not section:
            return None
        m = re.search(r"FIX_PRECISION_\w+", section)
        return m.group(0) if m else None

    def _extract_changed_locations(self, content: str) -> list:
        """从 [TARGET_FILES] section 中解析被修改的文件列表（保留文件名 token）。"""
        section = self._extract_section(content, "TARGET_FILES")
        if not section:
            return []
        locations = []
        seen = set()
        for line in section.split("\n"):
            line = line.strip().lstrip("-*•·").strip()
            for part in line.split():
                part = part.rstrip(",:;")
                if re.search(r"\.\w{1,5}$", part) and part not in seen:
                    locations.append(part)
                    seen.add(part)
        return locations

    def _extract_direction_verdict_value(self, content: str):
        """从 [DIRECTION_ASSESSMENT] 提取实际的 '是'/'否' 字符串。attempt==0 时返回 None。"""
        if self.attempt == 0:
            return None
        section = self._extract_section(content, "DIRECTION_ASSESSMENT")
        if not section:
            return None
        for line in section.split("\n"):
            if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            first_word = self._extract_direction_first_word(line[colon_pos + 1:].strip())
            if first_word in ("是", "否"):
                return first_word
        return None

    _SECTION_MAP = [
        ("forensics_summary", "FORENSICS_SUMMARY"),
        ("computation_decomposition", "COMPUTATION_DECOMPOSITION"),
        ("reference_impl_spec", "REFERENCE_IMPL_SPEC"),
        ("kernel_step_trace", "KERNEL_STEP_TRACE"),
        ("knowledge_match", "KNOWLEDGE_MATCH"),
        ("root_cause", "ROOT_CAUSE"),
        ("fix_plan", "FIX_PLAN"),
        ("target_files", "TARGET_FILES"),
        ("direction_assessment", "DIRECTION_ASSESSMENT"),
    ]

    def _build_sections_index(self, content: str, sections_dir: str) -> dict:
        """提取各 section 并写入小文件，返回 sections_index."""
        sections_index = {}
        base = f"precision_tuning/history/attempt_{self.attempt}/sections"
        for key, tag in self._SECTION_MAP:
            sec_text = self._extract_section(content, tag)
            rel_path = f"{base}/{key}.md"
            if sec_text is not None:
                sections_index[key] = self._write_one_section(
                    sections_dir, key, tag, sec_text, rel_path
                )
            else:
                sections_index[key] = None
        return sections_index

    def _write_audit_index(self, content: str) -> None:
        """Gate-A 通过后：提取各 section 为小文件，写 round_summary 的 diagnostics + index 初始字段。"""
        attempt_dir = os.path.join(self.tuning_dir, "history", f"attempt_{self.attempt}")
        sections_dir = os.path.join(attempt_dir, "sections")
        try:
            os.makedirs(sections_dir, exist_ok=True)
        except OSError:
            return

        sections_index = self._build_sections_index(content, sections_dir)
        n = self.attempt

        diagnostics = {
            "forensics_hint": None,
            "op_type": None,
            "fix_type": self._extract_fix_type(content),
            "changed_locations": self._extract_changed_locations(content),
            "direction_verdict": self._extract_direction_verdict_value(content),
        }

        index = {
            "forensics": f"precision_tuning/history/attempt_{n}/forensics_report.json",
            "audit_full": f"precision_tuning/precision_audit_{n}.md",
            "sections": sections_index,
            "code_snapshot": f"precision_tuning/history/attempt_{n}/code_snapshot/",
            "validation": f"precision_tuning/validation_result_attempt_{n}.json",
            "compilation_log": None,
            "tuning_directions": "precision_tuning/tuning_directions.json",
            "forensics_used": f"precision_tuning/forensics_report_{n}.json",
        }

        initial_summary = {
            "attempt": self.attempt,
            "gate_a_complete": True,
            "metrics": {
                "match_rate": None, "mismatch_ratio": None,
                "improvement_ratio": None, "absolute_improvement": None,
                "stop_reason_code": None,
            },
            "diagnostics": diagnostics,
            "index": index,
        }
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(initial_summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    # ---- 工具 ----

    def _get_baseline_match_rate(self) -> float | None:
        """
        获取 baseline match_rate（修改前的原始精度）。

        三层回退策略（按优先级）：
        1. 读 baseline_state.json（由 Gate-F 在 attempt 0 写入，最可靠）
        2. 读 forensics_report_{attempt}.json/history_trend/trend[0]（多轮时有历史数据）
        3. 直接读 forensics_report_{attempt}.json/outputs[0]/basic_stats（当前轮的原始数据）
        """
        mr = self._read_baseline_state()
        if mr is not None:
            return mr
        return self._read_baseline_from_forensics()

    def _read_baseline_state(self) -> float | None:
        baseline_path = os.path.join(self.tuning_dir, "baseline_state.json")
        if not os.path.exists(baseline_path):
            return None
        try:
            with open(baseline_path) as f:
                bs = json.load(f)
            mr = bs.get("match_rate")
            return float(mr) if mr is not None else None
        except (OSError, ValueError):
            return None

    def _read_baseline_from_forensics(self) -> float | None:
        forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if not os.path.exists(forensics_path):
            return None
        try:
            with open(forensics_path) as f:
                fr = json.load(f)
        except (OSError, ValueError, KeyError):
            return None

        # 层 2：history_trend
        history_trend = fr.get("history_trend")
        if history_trend:
            trend_list = history_trend.get("trend", [])
            if len(trend_list) >= 2:
                baseline_mismatch = trend_list[0].get("mismatch_ratio")
                if baseline_mismatch is not None:
                    return round((1 - float(baseline_mismatch)) * 100, 4)

        # 层 3：当前 forensics outputs（仅 attempt 0 语义正确）
        if self.attempt == 0:
            outputs = fr.get("outputs", [])
            if outputs:
                raw_mr = outputs[0].get("basic_stats", {}).get("match_rate")
                if raw_mr is not None:
                    return round(float(raw_mr) * 100, 4)
        return None

    def _find_project_dir(self) -> str | None:
        # AscendOpGenAgent: kernel lives at {output_dir}/kernel/
        if self.project_type == "ascendopgen":
            kdir = os.path.join(self.output_path, "kernel")
            if os.path.isdir(kdir):
                return kdir
            return None
        # OpenOps: kernel lives at {output_path}/{OpName}Custom/
        pascal = "".join(w.capitalize() for w in self.op_name.split("_")) + "Custom"
        c = os.path.join(self.output_path, pascal)
        if os.path.isdir(c):
            return c
        try:
            for item in os.listdir(self.output_path):
                full = os.path.join(self.output_path, item)
                if item.endswith("Custom") and os.path.isdir(full):
                    return full
        except FileNotFoundError:
            pass
        return None


# ================================================================
# CLI
# ================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="精度调优 Gate 验证 (链式)")
    parser.add_argument("--step", required=True,
                        choices=["forensics", "audit", "pre-edit", "fix", "validate"])
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--project-type", default="openops",
                        choices=["openops", "ascendopgen"])
    args = parser.parse_args()

    ck = GateChecker(args.op_name, args.output_path, args.attempt, args.project_type)
    dispatch = {
        "forensics": ck.check_forensics,
        "audit": ck.check_audit,
        "pre-edit": ck.check_pre_edit,
        "fix": ck.check_fix,
        "validate": ck.check_validate,
    }

    result = dispatch.get(args.step, lambda: {"gate": "UNKNOWN", "passed": False})()
    if not result.get("gate") or result["gate"] == "UNKNOWN":
        logger.error(f"Unknown step: {args.step}")
        sys.exit(2)
    logger.info("%s", json.dumps(result, indent=2, ensure_ascii=False))

    # 前置依赖失败用返回码 2 (致命), 区别于产物不完整的返回码 1
    if result.get("prerequisite_error"):
        logger.error(f"[{result['gate']}] PREREQUISITE FAILED — {result['prerequisite_error']}")
        sys.exit(2)

    if result["passed"]:
        logger.info(f"[{result['gate']}] PASSED")
        if args.step == "validate":
            logger.info(f"  loop_signal: {result.get('loop_signal')}")
            logger.info(f"  reason: {result.get('loop_reason')}")
        sys.exit(0)
    else:
        failed = [k for k, v in result["checks"].items() if not v]
        logger.warning(f"[{result['gate']}] FAILED — missing: {failed}")
        if args.step == "validate":
            logger.info(f"  loop_signal: {result.get('loop_signal')}")
            logger.info(f"  reason: {result.get('loop_reason')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
