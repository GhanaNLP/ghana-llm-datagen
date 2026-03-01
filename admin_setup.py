"""
Admin Setup Tool — Ghana LLM Dataset Generation
================================================
Run this ONCE as the project owner to generate volunteer codes.

Each code encodes:
  - data type (news or research)
  - row range (start_row, end_row) — their slice of the full CSV
  - the NVIDIA API key to use

No file splitting needed. Upload your two full CSVs to a GitHub Release,
and each volunteer's code points them to their row slice automatically.

Usage:
    python scripts/admin_setup.py \
        --nvidia-keys "nvapi-KEY1,nvapi-KEY2,nvapi-KEY3,nvapi-KEY4,nvapi-KEY5" \
        --repo "yourusername/ghana-llm-datagen" \
        --news-rows 10000 \
        --research-rows 8000

Tip — count your rows first:
    python -c "import pandas as pd; df=pd.read_csv('news_data.csv'); print(len(df))"
"""

import argparse
import base64
import json
import math
import sys


def encode_volunteer_code(data_type: str, row_start: int, row_end: int, api_key: str) -> str:
    payload = json.dumps({
        "t": data_type[0],   # "n" or "r"
        "s": row_start,
        "e": row_end,
        "k": api_key,
    }, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def generate_codes(keys: list, data_type: str, total_rows: int, n_volunteers: int) -> list:
    slice_size = math.ceil(total_rows / n_volunteers)
    codes = []
    for i, key in enumerate(keys[:n_volunteers]):
        start = i * slice_size
        end   = min(start + slice_size, total_rows)
        code  = encode_volunteer_code(data_type, start, end, key)
        codes.append({
            "code":      code,
            "type":      data_type,
            "row_start": start,
            "row_end":   end,
            "rows":      end - start,
            "api_key":   key,
        })
    return codes


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("--nvidia-keys",        required=True,
                        help="Comma-separated NVIDIA API keys (one per volunteer pair)")
    parser.add_argument("--repo",               required=True,
                        help="GitHub repo: username/reponame")
    parser.add_argument("--news-rows",          type=int, required=True,
                        help="Total data rows in your news CSV")
    parser.add_argument("--research-rows",      type=int, required=True,
                        help="Total data rows in your research CSV")
    parser.add_argument("--volunteers",         type=int, default=5,
                        help="Number of volunteers per data type (default: 5)")
    parser.add_argument("--release-tag",        default="v1.0-data")
    parser.add_argument("--news-filename",      default="news_data.csv",
                        help="Exact filename you'll upload to the GitHub Release")
    parser.add_argument("--research-filename",  default="research_data.csv",
                        help="Exact filename you'll upload to the GitHub Release")
    parser.add_argument("--out",                default="volunteer_codes.json",
                        help="Where to save the codes (keep private)")
    args = parser.parse_args()

    keys = [k.strip() for k in args.nvidia_keys.split(",")]
    if len(keys) < args.volunteers:
        sys.exit(f"❌  Need at least {args.volunteers} keys for {args.volunteers} volunteers per type. Got {len(keys)}.")

    news_codes     = generate_codes(keys, "news",     args.news_rows,     args.volunteers)
    research_codes = generate_codes(keys, "research", args.research_rows, args.volunteers)
    all_codes      = news_codes + research_codes

    # ── Print code table ───────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          VOLUNTEER CODES — GHANA LLM DATAGEN                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Repo    : https://github.com/{args.repo:<41} ║
║  Release : {args.release_tag:<60} ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    print(f"  {'#':<3} {'TYPE':<10} {'ROW RANGE':<22} {'COUNT':<8} CODE")
    print(f"  {'-'*3} {'-'*10} {'-'*22} {'-'*8} {'-'*60}")

    for i, c in enumerate(all_codes, 1):
        rng = f"rows {c['row_start']:,}–{c['row_end']:,}"
        print(f"  {i:<3} {c['type']:<10} {rng:<22} {c['rows']:<8,} {c['code']}")

    print()

    # ── Save full config ───────────────────────────────────────────────────
    config = {
        "repo":                args.repo,
        "release_tag":         args.release_tag,
        "news_filename":       args.news_filename,
        "research_filename":   args.research_filename,
        "news_total_rows":     args.news_rows,
        "research_total_rows": args.research_rows,
        "volunteers":          all_codes,
    }
    with open(args.out, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✅  Full table saved to: {args.out}  ⚠️  Keep this file PRIVATE.\n")

    # ── Next steps ─────────────────────────────────────────────────────────
    print(f"""{'='*65}
  NEXT STEPS
{'='*65}

1. Upload your two full CSV files to a GitHub Release:
     https://github.com/{args.repo}/releases/new
     Tag  : {args.release_tag}
     Files: {args.news_filename}  and  {args.research_filename}

2. In run.py, update the four config lines at the top:
     GITHUB_REPO        = "{args.repo}"
     RELEASE_TAG        = "{args.release_tag}"
     NEWS_FILENAME      = "{args.news_filename}"
     RESEARCH_FILENAME  = "{args.research_filename}"

3. Push run.py to GitHub, then send each volunteer their code.
   They only need to run:

     python run.py --code THEIR_CODE
""")


if __name__ == "__main__":
    main()
