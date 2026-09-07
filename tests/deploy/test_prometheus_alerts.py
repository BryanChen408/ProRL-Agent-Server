from __future__ import annotations

from pathlib import Path

import yaml

RULES = Path("deploy/observability/prometheus/polar_alerts.yaml")
RECORDING_RULES = Path(
    "deploy/observability/prometheus/polar_engine_recording_rules.yaml"
)


def test_polar_alert_rules_are_valid_and_cover_core_failure_modes() -> None:
    payload = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    rules = [rule for group in payload["groups"] for rule in group["rules"]]
    alerts = {rule["alert"] for rule in rules}

    assert alerts == {
        "PolarGatewayTargetMissing",
        "PolarGatewayDown",
        "PolarInferenceMetricsTargetDown",
        "PolarPdBackendMetricsDown",
        "PolarTraceExportFailing",
        "PolarSessionBacklogHigh",
        "PolarSessionFailureRatioHigh",
        "PolarSessionLatencyP95High",
        "PolarInferenceLatencyP95High",
    }
    assert all(rule.get("for") for rule in rules)
    assert all(rule["labels"]["severity"] in {"warning", "critical"} for rule in rules)


def test_polar_alert_rules_do_not_introduce_high_cardinality_labels() -> None:
    payload = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    expressions = [
        rule["expr"] for group in payload["groups"] for rule in group["rules"]
    ]

    assert all("session_id" not in expression for expression in expressions)
    assert all("task_id" not in expression for expression in expressions)


def test_engine_recording_rules_normalize_native_vllm_histograms() -> None:
    payload = yaml.safe_load(RECORDING_RULES.read_text(encoding="utf-8"))
    rules = [rule for group in payload["groups"] for rule in group["rules"]]
    records = {rule["record"] for rule in rules}

    for phase in ("queue", "ttft", "prefill", "decode"):
        for suffix in ("bucket", "sum", "count"):
            assert f"polar_inference_{phase}_seconds_{suffix}" in records
    assert "polar_inference_prefix_cache_hit_ratio" in records
    assert all(rule["labels"]["engine"] == "vllm-native" for rule in rules)
    assert all('job="polar-inference-engine"' in rule["expr"] for rule in rules)
    cache_rule = next(
        rule
        for rule in rules
        if rule["record"] == "polar_inference_prefix_cache_hit_ratio"
    )
    assert "vllm:prefix_cache_hits_total" in cache_rule["expr"]
    assert "vllm:prefix_cache_queries_total" in cache_rule["expr"]
