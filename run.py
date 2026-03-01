"""
Ghana LLM Dataset Generator — Volunteer Entry Point
=====================================================
This is the ONLY script volunteers need to run.

Usage:
    python run.py --code YOUR_VOLUNTEER_CODE

Your code tells the script:
  - Whether you're processing news or research data
  - Exactly which rows of the dataset are yours (e.g. rows 2000–4000)
  - Your API key

The script will:
  1. Decode your assignment from the code
  2. Download the full dataset CSV (cached after first download)
  3. Process only YOUR assigned rows
  4. Auto-resume if interrupted — just re-run the same command
  5. Tell you how to submit when done

Requirements:
    pip install openai pandas tqdm
"""

import json
import argparse
import base64
import time
import hashlib
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
missing = []
try:    import pandas as pd
except: missing.append("pandas")
try:    from tqdm import tqdm
except: missing.append("tqdm")
try:    import openai
except: missing.append("openai")

if missing:
    sys.exit(f"❌  Missing packages. Run:\n\n    pip install {' '.join(missing)}\n")

# ── Config — owner updates these 4 lines before pushing ──────────────────────

GITHUB_REPO        = "YOUR_USERNAME/ghana-llm-datagen"
RELEASE_TAG        = "v1.0-data"
NEWS_FILENAME      = "news_data.csv"
RESEARCH_FILENAME  = "research_data.csv"

# ── Model config ──────────────────────────────────────────────────────────────

NVIDIA_BASE_URL   = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL      = "meta/llama-3.1-70b-instruct"
RETRY_ATTEMPTS    = 4
RETRY_DELAY       = 8
MAX_CONTENT_CHARS = 3500
PAGES_PER_CHUNK   = 2          # research only: rows grouped per API call


# ── Decode volunteer code ─────────────────────────────────────────────────────

def decode_code(code: str) -> dict:
    try:
        padded  = code + "=" * (4 - len(code) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        data_type = "news" if payload["t"] == "n" else "research"
        return {
            "type":      data_type,
            "row_start": payload["s"],
            "row_end":   payload["e"],
            "api_key":   payload["k"],
        }
    except Exception:
        sys.exit("❌  Invalid volunteer code. Please double-check and try again.")


# ── Download CSV (cached) ─────────────────────────────────────────────────────

def get_csv(data_type: str) -> Path:
    filename   = NEWS_FILENAME if data_type == "news" else RESEARCH_FILENAME
    cache_path = Path("data_cache") / filename

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1_048_576
        print(f"📂  Using cached file: {cache_path}  ({size_mb:.1f} MB)")
        return cache_path

    url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{filename}"
    print(f"⬇️   Downloading dataset...")
    print(f"    {url}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    last_pct = [-1]
    def progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size / total_size * 100), 100)
            if pct != last_pct[0]:
                mb = total_size / 1_048_576
                print(f"\r    {pct}% of {mb:.1f} MB", end="", flush=True)
                last_pct[0] = pct

    try:
        urllib.request.urlretrieve(url, cache_path, progress)
        print()
    except urllib.error.HTTPError as e:
        sys.exit(f"\n❌  Download failed (HTTP {e.code}).\n    Check that the file exists in the release:\n    {url}")
    except Exception as e:
        sys.exit(f"\n❌  Download failed: {e}")

    size_mb = cache_path.stat().st_size / 1_048_576
    print(f"    ✅  Saved to {cache_path}  ({size_mb:.1f} MB)\n")
    return cache_path


# ── Load assigned slice ───────────────────────────────────────────────────────

def load_slice(csv_path: Path, row_start: int, row_end: int) -> "pd.DataFrame":
    # Read only the assigned rows (0-indexed, after header)
    df = pd.read_csv(csv_path, skiprows=range(1, row_start + 1),
                     nrows=row_end - row_start)
    return df.reset_index(drop=True)


# ── API ───────────────────────────────────────────────────────────────────────

def make_client(api_key: str):
    return openai.OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)


def call_api(client, prompt: str) -> "str | None":
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.75,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = RETRY_DELAY * (attempt + 1)
            tqdm.write(f"  ⚠️  Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(wait)
    return None


# ── Chunk builders ────────────────────────────────────────────────────────────

def build_news_chunks(df: "pd.DataFrame", row_start: int) -> list:
    required = {"url", "title", "content", "date", "category"}
    if not required.issubset(df.columns):
        sys.exit(f"❌  News CSV missing columns. Expected: {required}\n    Found: {set(df.columns)}")
    df = df.dropna(subset=["content", "title"]).reset_index(drop=True)
    chunks = []
    for abs_idx, (_, row) in enumerate(df.iterrows(), start=row_start):
        title    = str(row["title"])
        content  = str(row["content"])[:MAX_CONTENT_CHARS]
        date     = str(row.get("date", ""))
        category = str(row.get("category", ""))
        url      = str(row.get("url", ""))
        combined = f"Title: {title}\nDate: {date}\nCategory: {category}\n\n{content}"
        chunk_id = hashlib.md5((url + title).encode()).hexdigest()
        chunks.append({
            "chunk_id": chunk_id, "abs_row": abs_idx,
            "title": title, "category": category,
            "url": url, "date": date, "combined_text": combined,
        })
    return chunks


def build_research_chunks(df: "pd.DataFrame", row_start: int) -> list:
    required = {"filename", "page_range", "content"}
    if not required.issubset(df.columns):
        sys.exit(f"❌  Research CSV missing columns. Expected: {required}\n    Found: {set(df.columns)}")
    df = df.dropna(subset=["filename", "content"]).reset_index(drop=True)
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""].reset_index(drop=True)
    chunks = []
    for filename, group in df.groupby("filename", sort=False):
        rows = group.reset_index(drop=True)
        for i in range(0, len(rows), PAGES_PER_CHUNK):
            chunk_rows  = rows.iloc[i:i + PAGES_PER_CHUNK]
            page_ranges = " + ".join(chunk_rows["page_range"].astype(str).tolist())
            combined    = "\n\n".join(chunk_rows["content"].astype(str).tolist())
            chunk_id    = hashlib.md5(
                f"{filename}::{row_start}::{combined[:200]}".encode()
            ).hexdigest()
            chunks.append({
                "chunk_id": chunk_id, "filename": filename,
                "page_ranges": page_ranges, "content": combined,
            })
    return chunks


# ── Prompts ───────────────────────────────────────────────────────────────────

def news_prompt(chunk: dict) -> str:
    return f"""You are a dataset creator. Generate a high-quality multi-turn conversation in the style of UltraChat, based strictly on this Ghanaian news article.

## News Article
{chunk['combined_text']}

## Instructions:
- Generate a realistic multi-turn conversation between a curious USER and a knowledgeable ASSISTANT.
- The conversation must have 4-6 turns (USER and ASSISTANT alternating).
- Ground all facts strictly in the article. Do not invent facts.
- USER asks progressively deeper questions (causes, implications, stakeholders, comparisons).
- ASSISTANT gives accurate, well-explained answers from the article.
- Output ONLY valid JSON — no markdown, no preamble, no extra text.

Required format:
{{
  "id": "ghana_news_conv",
  "source_title": "{chunk['title'].replace('"', '')}",
  "category": "{chunk['category']}",
  "conversations": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}"""


def research_prompt(chunk: dict) -> str:
    return f"""You are a dataset creator. Generate a high-quality multi-turn educational conversation in the style of UltraChat, grounded in this excerpt from a Ghanaian research article.

## Research Excerpt:
{chunk['content']}

## Instructions:
- Generate a realistic multi-turn conversation between a curious USER and a knowledgeable ASSISTANT.
- The conversation should have 4-6 turns (USER and ASSISTANT alternating).
- Base all factual content strictly on the excerpt. Do not invent facts.
- USER asks progressively deeper questions.
- ASSISTANT gives accurate, well-explained answers.
- Output ONLY valid JSON — no markdown, no preamble, no extra text.

Required format:
{{
  "id": "ghana_research_conv",
  "conversations": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}"""


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json(raw: str, chunk: dict, data_type: str):
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            for part in cleaned.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    cleaned = part
                    break
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        data = json.loads(cleaned)
        data["chunk_id"] = chunk["chunk_id"]
        if data_type == "news":
            data["source_url"]  = chunk["url"]
            data["source_date"] = chunk["date"]
        else:
            data["source_file"]  = chunk["filename"]
            data["source_pages"] = chunk["page_ranges"]
        return data
    except json.JSONDecodeError as e:
        tqdm.write(f"  ⚠️  JSON parse error: {e} | preview: {raw[:120]}")
        return None


# ── Resume support ────────────────────────────────────────────────────────────

def load_completed(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                cid = json.loads(line.strip()).get("chunk_id")
                if cid:
                    done.add(cid)
            except Exception:
                pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ghana LLM Dataset Generator",
        epilog="Example:\n  python run.py --code eyJ0IjoibiIsInMiOjAsImUiOjIwMDAsImsiOiJudmFwaS0uLi4ifQ"
    )
    parser.add_argument("--code",   required=True, help="Your volunteer code")
    parser.add_argument("--output", default=None,  help="Custom output path (optional)")
    args = parser.parse_args()

    info = decode_code(args.code)

    # Derive a short label for the output file: e.g. news_0_2000
    label       = f"{info['type']}_{info['row_start']}_{info['row_end']}"
    output_path = Path(args.output or f"results/{label}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════╗
║       Ghana LLM Dataset Generator — Volunteer        ║
╠══════════════════════════════════════════════════════╣
║  Type     : {info['type'].upper():<41} ║
║  Rows     : {info['row_start']:,} – {info['row_end']:,}{' '*(37 - len(f"{info['row_start']:,} – {info['row_end']:,}"))} ║
║  Count    : {(info['row_end']-info['row_start']):,} rows{' '*(38 - len(f"{(info['row_end']-info['row_start']):,} rows"))} ║
║  Model    : {NVIDIA_MODEL:<41} ║
║  Output   : {str(output_path):<41} ║
╚══════════════════════════════════════════════════════╝
""")

    # ── Download & slice ───────────────────────────────────────────────────
    csv_path = get_csv(info["type"])
    df       = load_slice(csv_path, info["row_start"], info["row_end"])
    print(f"📊  Loaded {len(df):,} rows (your slice: {info['row_start']:,}–{info['row_end']:,})\n")

    # ── Build chunks ───────────────────────────────────────────────────────
    if info["type"] == "news":
        chunks = build_news_chunks(df, info["row_start"])
    else:
        chunks = build_research_chunks(df, info["row_start"])
    print(f"📦  Chunks to process: {len(chunks):,}")

    # ── Resume ─────────────────────────────────────────────────────────────
    completed = load_completed(output_path)
    pending   = [c for c in chunks if c["chunk_id"] not in completed]
    print(f"✅  Already done: {len(completed):,}")
    print(f"⏳  Remaining   : {len(pending):,}")

    if not pending:
        print("\n🎉  All done! Ready to submit.")
        _print_submit(output_path)
        return

    est_h = len(pending) * 8 / 3600
    print(f"⏱️   Estimated time: ~{est_h:.1f}h  (auto-resumes if interrupted)\n")

    client = make_client(info["api_key"])

    # ── Generate ───────────────────────────────────────────────────────────
    with open(output_path, "a", encoding="utf-8") as out_f:
        for chunk in tqdm(pending, desc=f"{info['type'].upper()} rows {info['row_start']:,}–{info['row_end']:,}"):
            label_str = chunk.get("title", chunk.get("filename", ""))[:65]
            tqdm.write(f"\n  → {label_str}")

            prompt     = news_prompt(chunk) if info["type"] == "news" else research_prompt(chunk)
            raw_output = call_api(client, prompt)

            if raw_output is None:
                tqdm.write("  ⏭️  Skipped (all retries failed)")
                continue

            record = parse_json(raw_output, chunk, info["type"])

            if record:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                tqdm.write(f"  ✅  {len(record.get('conversations', []))} turns saved")
            else:
                fallback = {"chunk_id": chunk["chunk_id"], "raw_output": raw_output, "parse_error": True}
                if info["type"] == "news":
                    fallback.update({"source_url": chunk["url"], "category": chunk.get("category")})
                else:
                    fallback.update({"source_file": chunk["filename"], "source_pages": chunk["page_ranges"]})
                out_f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
                out_f.flush()
                tqdm.write("  ⚠️   Raw output saved (parse failed)")

    # ── Summary ────────────────────────────────────────────────────────────
    lines = [json.loads(l) for l in open(output_path) if l.strip()]
    good  = sum(1 for l in lines if not l.get("parse_error"))

    print(f"""
╔══════════════════════════════════════════════════════╗
║                  🎉  RUN COMPLETE!                   ║
╠══════════════════════════════════════════════════════╣
║  Total records : {len(lines):<35,} ║
║  Parsed OK     : {good:<35,} ║
╚══════════════════════════════════════════════════════╝
""")
    _print_submit(output_path)


def _print_submit(output_path: Path):
    print(f"""📤  Submit your results:

  1. Open: https://github.com/{GITHUB_REPO}/issues/new?template=result_submission.md
  2. Attach this file: {output_path.resolve()}
  3. Submit the issue — done!

Thank you for contributing to the Ghana LLM project! 🇬🇭
""")


if __name__ == "__main__":
    main()
