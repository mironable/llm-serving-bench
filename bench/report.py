"""Aggregate run parquet files into a latency-versus-throughput markdown table."""

import argparse
import math
import re
from pathlib import Path

import pandas as pd

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"

COLUMNS = [
    "server",
    "concurrency",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "itl_mean_ms",
    "throughput_tok_s",
    "queue_depth",
    "kv_usage",
    "errors",
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.groupby(["server", "concurrency"], sort=True):
        rows.append(
            {
                "server": str(g["server"].iloc[0]),
                "concurrency": int(g["concurrency"].iloc[0]),
                "ttft_p50_ms": g["ttft_s"].median() * 1000,
                "ttft_p95_ms": g["ttft_s"].quantile(0.95) * 1000,
                "itl_mean_ms": g["itl_mean_s"].mean() * 1000,
                "throughput_tok_s": g["tokens"].sum() / g["level_wall_s"].iloc[0],
                "queue_depth": g["queue_depth"].iloc[0],
                "kv_usage": g["kv_usage"].iloc[0],
                "errors": int(g["level_errors"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def to_markdown(summary: pd.DataFrame) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, sep]
    for r in summary.to_dict("records"):
        cells = [str(r["server"]), str(r["concurrency"])]
        for col in COLUMNS[2:]:
            v = r[col]
            cells.append("n/a" if _missing(v) else f"{float(v):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def splice_readme(readme: str, table: str) -> str:
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    block = f"{README_START}\n{table}\n{README_END}"
    if not pattern.search(readme):
        raise ValueError("README has no results markers")
    return pattern.sub(block, readme)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet", nargs="+")
    p.add_argument("--write-readme", action="store_true")
    args = p.parse_args()
    df = pd.concat([pd.read_parquet(f) for f in args.parquet], ignore_index=True)
    table = to_markdown(summarize(df))
    print(table)
    if args.write_readme:
        path = Path("README.md")
        path.write_text(splice_readme(path.read_text(), table))
        print("README updated")


if __name__ == "__main__":
    main()
