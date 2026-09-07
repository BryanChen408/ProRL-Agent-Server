# Polar observability assets

For the complete Chinese deployment and usage guide, see
[`POLAR_RL_INSIGHT_GUIDE.zh-CN.md`](POLAR_RL_INSIGHT_GUIDE.zh-CN.md).

`grafana/polar_gateway.json` is an importable Grafana dashboard for metrics
exposed by a Polar Gateway and registered with RL-Insight.

Import it from **Dashboards → New → Import → Upload dashboard JSON file**, then
select RL-Insight's `Prometheus` data source and click **Import**. The dashboard
defaults to the last hour, refreshes every ten seconds, and filters by one or
more `node_id` values.

The inference panels require a Gateway that exposes live inference engine
telemetry. Older Gateway processes still populate the session, queue, aggregate
token, and export-failure metrics.

## Prometheus alert rules

`prometheus/polar_alerts.yaml` contains low-cardinality alerts for Gateway
availability, failed OTLP exports, sustained scheduler backlog, session failure
ratio, session p95 latency, and inference p95 latency. The latency and backlog
thresholds are operational defaults; tune them for the workload before routing
notifications.

`prometheus/polar_engine_recording_rules.yaml` normalizes native vLLM queue,
TTFT, prefill, decode, and prefix-cache metrics into the `polar_inference_*`
namespace. Polar registers the configured inference endpoint under the
`polar-inference-engine` job. A Mooncake PD proxy must expose its backend metrics
at `/metrics`; the VIME PD proxy integration fans in all prefill and decode
engines and adds `pd_role` and `pd_backend` labels.

RL-Insight renders its runtime Prometheus configuration from the file configured
by `prometheus.config_file`. Add the rule file to that source configuration using
an absolute path, for example:

```yaml
rule_files:
  - /home/docker/polar_can/ProRL-Agent-Server/deploy/observability/prometheus/polar_alerts.yaml
  - /home/docker/polar_can/ProRL-Agent-Server/deploy/observability/prometheus/polar_engine_recording_rules.yaml
```

Then restart the RL-Insight services so the generated runtime configuration is
refreshed:

```bash
rl-insight server stop
rl-insight server start --detach
```

The rules are evaluated by Prometheus. Sending notifications additionally
requires an Alertmanager (or Grafana alerting) integration; RL-Insight does not
configure a notification destination by itself.
