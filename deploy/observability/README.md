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
