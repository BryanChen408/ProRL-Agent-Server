from __future__ import annotations

import json
from pathlib import Path


DASHBOARD = Path("deploy/observability/grafana/polar_gateway.json")


def test_polar_grafana_dashboard_is_valid_and_uses_bounded_labels() -> None:
    payload = json.loads(DASHBOARD.read_text(encoding="utf-8"))

    assert payload["uid"] == "polar-gateway"
    assert len(payload["panels"]) >= 10
    expressions = [
        target["expr"]
        for panel in payload["panels"]
        for target in panel.get("targets", [])
    ]
    assert any("polar_gateway_sessions" in expression for expression in expressions)
    assert any("polar_inference_ttft_seconds" in expression for expression in expressions)
    assert any("polar_inference_tokens_total" in expression for expression in expressions)
    assert all("session_id" not in expression for expression in expressions)
    assert all("task_id" not in expression for expression in expressions)


def test_polar_grafana_dashboard_uses_importable_prometheus_datasource() -> None:
    payload = json.loads(DASHBOARD.read_text(encoding="utf-8"))

    assert payload["__inputs"][0]["name"] == "DS_PROMETHEUS"
    assert payload["__inputs"][0]["pluginId"] == "prometheus"
    assert all(
        panel["datasource"] == {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
        for panel in payload["panels"]
    )
