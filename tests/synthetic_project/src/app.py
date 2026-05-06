"""Main application entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_csv, summarize_sales, top_regions, DATA_DIR


def main():
    sales_path = os.path.join(DATA_DIR, "sales.csv")
    data = load_csv(sales_path)

    print(f"Loaded {len(data)} sales records")
    print()

    summary = summarize_sales(data)
    print("Sales by Region:")
    for region, info in sorted(summary.items()):
        print(f"  {region:8s}: ${info['total']:>8.2f} ({info['count']} orders, {', '.join(sorted(info['products']))})")

    print()
    top = top_regions(data)
    print("Top Regions:")
    for region, info in top:
        print(f"  {region}: ${info['total']:.2f}")


if __name__ == "__main__":
    main()
