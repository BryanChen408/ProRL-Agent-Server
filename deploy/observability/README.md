# Polar observability assets

`grafana/polar_gateway.json` is an importable Grafana dashboard for metrics
exposed by a Polar Gateway and registered with RL-Insight.

Import it from **Dashboards → New → Import → Upload dashboard JSON file**, then
select RL-Insight's `Prometheus` data source and click **Import**. The dashboard
defaults to the last hour, refreshes every ten seconds, and filters by one or
more `node_id` values.

The inference panels require a Gateway containing commit `feat(metrics): expose
live inference engine telemetry`. Older Gateway processes still populate the
session, queue, aggregate token, and export-failure metrics.

## Prometheus alert rules

`prometheus/polar_alerts.yaml` contains low-cardinality alerts for Gateway
availability, failed OTLP exports, sustained scheduler backlog, session failure
ratio, session p95 latency, and inference p95 latency. The latency and backlog
thresholds are operational defaults; tune them for the workload before routing
notifications.

RL-Insight renders its runtime Prometheus configuration from the file configured
by `prometheus.config_file`. Add the rule file to that source configuration using
an absolute path, for example:

```yaml
rule_files:
  - /home/docker/polar_can/ProRL-Agent-Server/deploy/observability/prometheus/polar_alerts.yaml
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
