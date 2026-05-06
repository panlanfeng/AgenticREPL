"""Utility functions for the synthetic project."""

import csv
import json
import os


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def summarize_sales(data):
    regions = {}
    for row in data:
        region = row["region"]
        amount = float(row["amount"])
        region_info = regions.setdefault(region, {"total": 0, "count": 0, "products": set()})
        region_info["total"] += amount
        region_info["count"] += 1
        region_info["products"].add(row["product"])
    return regions


def top_regions(data, n=3):
    summary = summarize_sales(data)
    sorted_regions = sorted(summary.items(), key=lambda x: x[1]["total"], reverse=True)
    return sorted_regions[:n]


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
