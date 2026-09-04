from __future__ import annotations

from pathlib import Path

import yaml

RULES = Path("deploy/observability/prometheus/polar_alerts.yaml")


def test_polar_alert_rules_are_valid_and_cover_core_failure_modes() -> None:
    payload = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    rules = [rule for group in payload["groups"] for rule in group["rules"]]
    alerts = {rule["alert"] for rule in rules}

    assert alerts == {
        "PolarGatewayTargetMissing",
        "PolarGatewayDown",
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
