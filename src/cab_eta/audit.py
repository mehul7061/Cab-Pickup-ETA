"""Reproduce the MP054 Stage 1 checks without altering the source data."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
NUMERIC = (
    "drivers_nearby", "distance_to_driver_km", "hour",
    "rain_mm", "surge", "road_closures",
)
FACTORS = (10, 12, 60, 100, 1000)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames, list(reader)


def constant_runs(rows):
    """Find runs with at least seven consecutive days of a constant reading."""
    runs = []
    for column in NUMERIC:
        daily = defaultdict(set)
        for row in rows:
            daily[date.fromisoformat(row["date"])].add(float(row[column]))
        start = previous = None
        reading = None
        for day, values in sorted(daily.items()):
            if start and len(values) == 1 and day == previous + timedelta(days=1) and values == reading:
                previous = day
                continue
            if start and (previous - start).days >= 6:
                runs.append((column, start.isoformat(), previous.isoformat()))
            start = day if len(values) == 1 else None
            previous, reading = day, values
        if start and (previous - start).days >= 6:
            runs.append((column, start.isoformat(), previous.isoformat()))
    return runs


def audit(paths, target):
    columns, train = read_csv(paths["train"])
    holdout_columns, _ = read_csv(paths["holdout"])
    _, later = read_csv(paths["later"])

    train_only = sorted(set(columns) - set(holdout_columns) - {target})
    duplicates = len(train) - len({tuple(row.values()) for row in train})
    zones = sorted({row["zone"] for row in train})
    summaries = {}
    unit_candidates = []
    for column in NUMERIC:
        by_zone = {
            zone: [float(row[column]) for row in train if row["zone"] == zone]
            for zone in zones
        }
        summaries[column] = {
            zone: {"min": min(values), "mean": mean(values), "max": max(values)}
            for zone, values in by_zone.items()
        }
        for zone, stats in summaries[column].items():
            peers = [value for key, value in summaries[column].items() if key != zone]
            peer_max = mean(value["max"] for value in peers)
            peer_mean = mean(value["mean"] for value in peers)
            if not (peer_max and peer_mean):
                continue
            ratio = stats["max"] / peer_max
            average_ratio = stats["mean"] / peer_mean
            for factor in FACTORS:
                for scale in (factor, 1 / factor):
                    if abs(ratio / scale - 1) <= 0.04 and abs(average_ratio / scale - 1) <= 0.2:
                        unit_candidates.append((column, zone, scale))

    sentinel_values = []
    for column in NUMERIC:
        counts = Counter(float(row[column]) for row in train)
        for value in (-9999, -999, -1, 999, 9999):
            if counts[value] > 1:
                sentinel_values.append((column, value, counts[value]))

    ordered = sorted(float(row[target]) for row in train)
    # An isolated upper tail must make sense after dividing by ten.
    cutoff = max(
        range(int(len(ordered) * 0.9), len(ordered)),
        key=lambda i: ordered[i] / ordered[i - 1],
    )
    tail = ordered[cutoff:]
    target_typos = (
        ordered[cutoff] >= 2 * ordered[cutoff - 1]
        and all(ordered[0] <= value / 10 <= ordered[cutoff - 1] for value in tail)
    )

    variants = []
    category_counts = {}
    for column in ("car_type", "zone"):
        raw = Counter(row[column] for row in train)
        clean = Counter(row[column] for row in later)
        category_counts[column] = {"train": raw, "later": clean}
        if set(raw) > set(clean):
            variants.append((column, len(clean)))

    stuck = constant_runs(train)
    findings = {
        "leaking_column": finding(("column",), [(column,) for column in train_only]),
        "unit_mixup": finding(("column", "group_value", "factor"), unit_candidates),
        "sentinel_value": finding(("column", "value", "count"), sentinel_values),
        "duplicate_rows": finding(("count",), [(duplicates,)] if duplicates else []),
        "stuck_sensor": finding(("column", "start_date", "end_date"), stuck),
        "target_typos": finding(("count",), [(len(tail),)] if target_typos else []),
        "spelling_variants": finding(("column", "true_levels"), variants),
    }
    evidence = {
        "train_only_columns": train_only,
        "numeric_summaries_by_zone": summaries,
        "repeated_sentinels": sentinel_values,
        "duplicate_rows": duplicates,
        "constant_daily_runs": stuck,
        "target_tail_candidate": {"previous_value": ordered[cutoff - 1], "values": tail},
        "category_counts": category_counts,
    }
    return findings, evidence


def finding(keys, matches):
    if len(matches) > 1:
        raise ValueError(f"Multiple candidates require review: {matches}")
    values = matches[0] if matches else (None,) * len(keys)
    return {"present": bool(matches), **dict(zip(keys, values))}


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-findings", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    paths = {key: ROOT / value for key, value in config["paths"].items()}
    findings, evidence = audit(paths, config["target"])
    if args.write_findings:
        report = yaml.safe_load(paths["findings"].read_text(encoding="utf-8"))
        report["data_issues"] = findings
        paths["findings"].write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(json.dumps({"data_issues": findings, "evidence": evidence}, indent=2))


if __name__ == "__main__":
    main()
