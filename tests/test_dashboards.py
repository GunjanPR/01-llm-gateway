"""Every metric the Grafana dashboard and the alert rules query must exist in the gateway.

A renamed metric otherwise fails silently: the panel shows "No data" and the alert never
fires.
"""

import json
import re
from pathlib import Path

import yaml

from llm_gateway.metrics import GatewayMetrics

ROOT = Path(__file__).resolve().parents[1]
METRIC = re.compile(r"\bllmgw_[a-z_]+")
SUFFIXES = ("_bucket", "_count", "_sum", "_created")


def _exported_names() -> set[str]:
    names: set[str] = set()
    for family in GatewayMetrics().registry.collect():
        names.add(family.name)
        names.update(sample.name for sample in family.samples)
    return names


def _referenced(text: str) -> set[str]:
    return set(METRIC.findall(text))


def _known(name: str, exported: set[str]) -> bool:
    base = name
    for suffix in SUFFIXES:
        base = base.removesuffix(suffix)
    # Counter families are named without "_total"; labelled metrics have no samples yet.
    return name in exported or base in exported or base.removesuffix("_total") in exported


def test_dashboard_queries_use_real_metrics():
    dashboard = json.loads((ROOT / "deploy/grafana/dashboards/llm-gateway.json").read_text())
    exprs = [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]
    exported = _exported_names()
    missing = {m for e in exprs for m in _referenced(e) if not _known(m, exported)}
    assert not missing, f"dashboard queries unknown metrics: {sorted(missing)}"
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids)), "duplicate panel ids"


def test_alert_rules_use_real_metrics():
    rules = yaml.safe_load((ROOT / "deploy/prometheus/alerts.yml").read_text())
    exported = _exported_names()
    for group in rules["groups"]:
        for rule in group["rules"]:
            missing = {m for m in _referenced(rule["expr"]) if not _known(m, exported)}
            assert not missing, f"{rule['alert']} queries unknown metrics: {sorted(missing)}"
            assert rule["labels"]["severity"] in {"warning", "critical"}
            assert rule["annotations"]["summary"]
