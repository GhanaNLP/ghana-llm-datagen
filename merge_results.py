"""
Merge Results — Admin Tool
===========================
Combines all volunteer .jsonl files into a single clean dataset.

Usage:
    python scripts/merge_results.py --results-dir ./results --output final_dataset.jsonl
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output",      default="final_dataset.jsonl")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("*.jsonl"))

    if not files:
        print(f"❌  No .jsonl files found in {results_dir}")
        return

    print(f"📂  Found {len(files)} result files:\n")

    total, good, errors = 0, 0, 0
    seen_ids = set()
    duplicates = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for fpath in files:
            file_total, file_good, file_errors, file_dupes = 0, 0, 0, 0
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        cid    = record.get("chunk_id", "")
                        if cid in seen_ids:
                            file_dupes += 1
                            continue
                        seen_ids.add(cid)
                        file_total += 1
                        if record.get("parse_error"):
                            file_errors += 1
                        else:
                            file_good += 1
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError:
                        file_errors += 1
            print(f"  {fpath.name:<40} {file_total:>5} records  "
                  f"({file_good} good, {file_errors} errors, {file_dupes} dupes skipped)")
            total      += file_total
            good       += file_good
            errors     += file_errors
            duplicates += file_dupes

    print(f"""
{'='*55}
  ✅  Merge complete!
  Output         : {args.output}
  Total records  : {total:,}
  Clean records  : {good:,}  (written to output)
  Parse errors   : {errors:,}  (excluded)
  Duplicates     : {duplicates:,}  (skipped)
{'='*55}
""")


if __name__ == "__main__":
    main()
