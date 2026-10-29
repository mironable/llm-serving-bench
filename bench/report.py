"""Aggregate run parquet files into latency-versus-throughput markdown tables."""

import argparse
import math
import re
from pathlib import Path

import pandas as pd

MARKERS = {
    "results": ("<!-- results:start -->", "<!-- results:end -->"),
    "reasoning": ("<!-- reasoning:start -->", "<!-- reasoning:end -->"),
}
DEFAULT_MODEL = "Qwen2.5-7B-Instruct"

KEY = ["workload", "model", "server", "concurrency"]
COLUMNS = KEY + [
    "ttft_p50_ms",
    "ttft_p95_ms",
    "itl_mean_ms",
    "throughput_tok_s",
    "queue_depth",
    "kv_usage",
    "errors",
]

REASONING_KEY = ["model", "reasoning", "concurrency"]
REASONING_COLUMNS = REASONING_KEY + [
    "ttft_p50_ms",
    "ttfvt_p50_ms",
    "ttfvt_p95_ms",
    "reasoning_chunks_mean",
    "visible_chunks_mean",
    "answered_pct",
    "visible_tok_s",
    "errors",
]


def _with_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Runs recorded before a column existed get its default so old and new parquet mix."""
    defaults = {"workload": "short", "model": DEFAULT_MODEL, "reasoning": "off"}
    for col, val in defaults.items():
        if col not in df.columns:
            df = df.assign(**{col: val})
        else:
            df = df.assign(**{col: df[col].fillna(val)})
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_defaults(df)
    df = df.loc[df["reasoning"] == "off"]  # thinking runs get their own table
    rows = []
    for _, g in df.groupby(KEY, sort=True):
        rows.append(
            {
                **{k: g[k].iloc[0] for k in KEY},
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


def summarize_reasoning(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_defaults(df)
    if "ttfvt_s" not in df.columns:
        return pd.DataFrame(columns=REASONING_COLUMNS)
    df = df.loc[df["model"] != DEFAULT_MODEL]  # only models run with a thinking switch
    rows = []
    for _, g in df.groupby(REASONING_KEY, sort=True):
        answered = g["ttfvt_s"].notna()
        rows.append(
            {
                "model": g["model"].iloc[0],
                "reasoning": g["reasoning"].iloc[0],
                "concurrency": int(g["concurrency"].iloc[0]),
                "ttft_p50_ms": g["ttft_s"].median() * 1000,
                "ttfvt_p50_ms": g["ttfvt_s"].median() * 1000,
                "ttfvt_p95_ms": g["ttfvt_s"].quantile(0.95) * 1000,
                "reasoning_chunks_mean": g["reasoning_chunks"].mean(),
                "visible_chunks_mean": g["visible_chunks"].mean(),
                "answered_pct": answered.mean() * 100,
                "visible_tok_s": g["visible_chunks"].sum() / g["level_wall_s"].iloc[0],
                "errors": int(g["level_errors"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=REASONING_COLUMNS)


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def to_markdown(summary: pd.DataFrame, columns: list[str], key: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for r in summary.to_dict("records"):
        cells = [str(r[k]) for k in key]
        for col in columns[len(key) :]:
            v = r[col]
            cells.append("n/a" if _missing(v) else f"{float(v):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def splice_readme(readme: str, table: str, section: str = "results") -> str:
    start, end = MARKERS[section]
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    block = f"{start}\n{table}\n{end}"
    if not pattern.search(readme):
        raise ValueError(f"README has no {section} markers")
    return pattern.sub(block, readme)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet", nargs="+")
    p.add_argument("--write-readme", action="store_true")
    args = p.parse_args()
    frames = [_with_defaults(pd.read_parquet(f)) for f in args.parquet]
    df = pd.concat(frames, ignore_index=True)
    results = to_markdown(summarize(df), COLUMNS, KEY)
    reasoning = to_markdown(summarize_reasoning(df), REASONING_COLUMNS, REASONING_KEY)
    print(results)
    print()
    print(reasoning)
    if args.write_readme:
        path = Path("README.md")
        text = splice_readme(path.read_text(), results, "results")
        if len(reasoning.splitlines()) > 2:
            text = splice_readme(text, reasoning, "reasoning")
        path.write_text(text)
        print("README updated")


if __name__ == "__main__":
    main()
