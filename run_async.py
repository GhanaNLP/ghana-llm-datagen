"""
Ghana LLM Dataset Generator — Volunteer Entry Point
=====================================================
Runs your assigned news and research data through Llama 3.1 70b for LLM data generation.

Usage:
    python run_main.py --code YOUR_VOLUNTEER_CODE
"""

import sys
import subprocess
from pathlib import Path


# ── Auto-install requirements ─────────────────────────────────────────────────


def install_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print("Warning: requirements.txt not found - skipping auto-install.")
        return
    print("Checking requirements...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Failed to install requirements:\n{result.stderr}")
        sys.exit(1)
    print("Requirements ready.\n")


install_requirements()


# ── Now safe to import third-party packages ───────────────────────────────────

import json
import argparse
import base64
import time
import hashlib
import urllib.request
import urllib.error
import lzma
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm
import openai
from progress_logger_async import ProgressLoggerAsync as ProgressLogger

# ── Config — owner updates these before pushing ───────────────────────────────

GITHUB_REPO = "GhanaNLP/ghana-llm-datagen"
RELEASE_TAG = "v1.0-data"
NEWS_FILENAME = "news_data.csv"
RESEARCH_FILENAME = "research_data.csv"

# ── Shared progress Gist — coordinator sets these once ────────────────────────
_TOK_A = "ghp_qkDY1necYIaLdLmp"  # first ~half of the github_pat_... token
_TOK_B = "70p5ueC60ScRfz4cIwtn"  # second half
PROGRESS_GIST_TOKEN = _TOK_A + _TOK_B  # reassembled at runtime only

PROGRESS_GIST_ID = "f36345805c8740981aa3bca09056ab9c"  # e.g. "a1b2c3d4e5f6..."

# ── Model config ──────────────────────────────────────────────────────────────

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
RETRY_DELAY = 8
MAX_CONTENT_CHARS = 3500
PAGES_PER_CHUNK = 2
ULTRACHAT_CSV = "ultrachat_sample.csv"  # style reference — lives in the repo


# ── Decode volunteer code ─────────────────────────────────────────────────────


def decode_code(code: str) -> dict:
    try:
        padded = code + "=" * (4 - len(code) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        return {
            "news_start": payload["ns"],
            "news_end": payload["ne"],
            "res_start": payload["rs"],
            "res_end": payload["re"],
            "api_key": payload["k"],
        }
    except Exception:
        sys.exit("❌  Invalid volunteer code. Please double-check and try again.")


# ── Download CSV (cached) ─────────────────────────────────────────────────────


def get_csv(data_type: str) -> Path:
    filename = NEWS_FILENAME if data_type == "news" else RESEARCH_FILENAME
    cache_path = Path("data_cache") / filename

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1_048_576
        print(f"📂  Using cached file: {cache_path}  ({size_mb:.1f} MB)")
        return cache_path

    url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{filename}"
    print(f"⬇️   Downloading {data_type} dataset...")
    print(f"    {url}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    last_pct = [-1]

    def progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size / total_size * 100), 100)
            if pct != last_pct[0]:
                print(
                    f"\r    {pct}% of {total_size / 1_048_576:.1f} MB",
                    end="",
                    flush=True,
                )
                last_pct[0] = pct

    try:
        urllib.request.urlretrieve(url, cache_path, progress)
        print()
    except urllib.error.HTTPError as e:
        sys.exit(f"\n❌  Download failed (HTTP {e.code}).\n    {url}")
    except Exception as e:
        sys.exit(f"\n❌  Download failed: {e}")

    print(
        f"    ✅  Saved to {cache_path}  ({cache_path.stat().st_size / 1_048_576:.1f} MB)\n"
    )
    return cache_path


# ── API ───────────────────────────────────────────────────────────────────────


def make_client(api_key: str):
    return openai.AsyncOpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)


async def call_api(client, prompt: str, semaphore: asyncio.Semaphore):
    """Retries indefinitely on network/server errors with capped exponential backoff.
    Only gives up if the error is non-retriable (e.g. bad request / auth failure)."""
    NON_RETRIABLE = (400, 401, 403)  # bad request or auth — retrying won't help
    attempt = 0
    async with semaphore:
        while True:
            try:
                resp = await client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.75,
                    max_tokens=2048,
                )
                return resp.choices[0].message.content.strip()
            except openai.APIStatusError as e:
                if e.status_code in NON_RETRIABLE:
                    tqdm.write(
                        f"  ✖  Non-retriable error ({e.status_code}): {e.message}. Skipping chunk."
                    )
                    return None
                wait = min(RETRY_DELAY * (2**attempt), 120)  # cap at 2 minutes
                attempt += 1
                tqdm.write(
                    f"  ⚠️  API error ({e.status_code}) — attempt {attempt}. Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            except Exception as e:
                wait = min(RETRY_DELAY * (2**attempt), 120)
                attempt += 1
                tqdm.write(
                    f"  ⚠️  Error: {e} — attempt {attempt}. Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)


# ── Chunk builders ────────────────────────────────────────────────────────────


def build_news_chunks(df, row_start: int) -> list:
    required = {"url", "title", "content", "date", "category"}
    if not required.issubset(df.columns):
        sys.exit(
            f"❌  News CSV missing columns. Expected: {required}\n    Found: {set(df.columns)}"
        )
    df = df.dropna(subset=["content", "title"]).reset_index(drop=True)
    chunks = []
    for row in df.itertuples(index=False):
        title = str(row.title)
        content = str(row.content)[:MAX_CONTENT_CHARS]
        date = str(getattr(row, "date", ""))
        category = str(getattr(row, "category", ""))
        url = str(getattr(row, "url", ""))
        combined = f"Title: {title}\nDate: {date}\nCategory: {category}\n\n{content}"
        chunk_id = hashlib.md5((url + title).encode()).hexdigest()
        chunks.append(
            {
                "chunk_id": chunk_id,
                "title": title,
                "category": category,
                "url": url,
                "date": date,
                "combined_text": combined,
            }
        )
    return chunks


def build_research_chunks(df, row_start: int) -> list:
    required = {"filename", "page_range", "content"}
    if not required.issubset(df.columns):
        sys.exit(
            f"❌  Research CSV missing columns. Expected: {required}\n    Found: {set(df.columns)}"
        )
    df = df.dropna(subset=["filename", "content"]).reset_index(drop=True)
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""].reset_index(drop=True)
    chunks = []
    for filename, group in df.groupby("filename", sort=False):
        rows = group.reset_index(drop=True)
        for i in range(0, len(rows), PAGES_PER_CHUNK):
            chunk_rows = rows.iloc[i : i + PAGES_PER_CHUNK]
            page_ranges = " + ".join(chunk_rows["page_range"].astype(str).tolist())
            combined = "\n\n".join(chunk_rows["content"].astype(str).tolist())
            chunk_id = hashlib.md5(
                f"{filename}::{row_start}::{combined[:200]}".encode()
            ).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "filename": filename,
                    "page_ranges": page_ranges,
                    "content": combined,
                }
            )
    return chunks


# ── UltraChat style samples ───────────────────────────────────────────────────


def load_ultrachat_samples() -> list:
    csv_path = Path(__file__).parent / ULTRACHAT_CSV
    if not csv_path.exists():
        sys.exit(
            f"❌  {ULTRACHAT_CSV} not found. It should be in the same folder as run_main.py."
        )
    import ast as _ast

    df = pd.read_csv(csv_path)
    if "data" not in df.columns:
        sys.exit(
            f"❌  {ULTRACHAT_CSV} must have a 'data' column. Found: {list(df.columns)}"
        )
    samples = []
    for raw in df["data"].dropna():
        try:
            turns = _ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not isinstance(turns, list) or len(turns) < 2:
            continue
        roles = ["user", "assistant"]
        msgs = [
            {"role": roles[i % 2], "content": str(turns[i])} for i in range(len(turns))
        ]
        samples.append(msgs)
    print(f"📚  Loaded {len(samples)} UltraChat style samples.")
    return samples


def format_ultrachat_example(msgs: list) -> str:
    lines = []
    for m in msgs[:6]:
        role = m.get("role", "user").capitalize() if isinstance(m, dict) else "User"
        content = (m.get("content", "") if isinstance(m, dict) else str(m))[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────


def news_prompt(chunk: dict, example: str) -> str:
    return f"""You are a dataset creator. Generate a high-quality multi-turn conversation in the style of UltraChat, based strictly on this Ghanaian news article.

## News Article
{chunk["combined_text"]}

## Example UltraChat-style Conversation (style reference only — do NOT copy its content):
{example}

## Instructions:
- Generate a realistic multi-turn conversation between a curious USER and a knowledgeable ASSISTANT.
- The conversation must have 4-6 turns (USER and ASSISTANT alternating).
- Ground all facts strictly in the article. Do not invent facts.
- The article's publication date is provided in the header above. The ASSISTANT must refer to events using their exact dates (e.g. "On 5 September 2025...") rather than vague terms like "recently" or "last week".
- USER asks progressively deeper questions (causes, implications, stakeholders, comparisons).
- ASSISTANT gives accurate, well-explained answers from the article.
- Output ONLY valid JSON — no markdown, no preamble, no extra text.

Required format:
{{
  "id": "ghana_news_conv",
  "source_title": "{chunk["title"].replace('"', "")}",
  "category": "{chunk["category"]}",
  "conversations": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}"""


def research_prompt(chunk: dict, example: str) -> str:
    return f"""You are a dataset creator. Generate a high-quality multi-turn educational conversation in the style of UltraChat, grounded in this excerpt from a Ghanaian research article.

## Research Excerpt:
{chunk["content"]}

## Example UltraChat-style Conversation (style reference only — do NOT copy its content):
{example}

## Instructions:
- Generate a realistic multi-turn conversation between a curious USER and a knowledgeable ASSISTANT.
- The conversation should have 4-6 turns (USER and ASSISTANT alternating).
- Base all factual content strictly on the excerpt. Do not invent facts.
- The ASSISTANT must refer to events using their exact dates (e.g. "On 5 September 2025...") rather than vague terms like "recently" or "last week".
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
            cleaned = cleaned[start : end + 1]
        data = json.loads(cleaned)
        data["chunk_id"] = chunk["chunk_id"]
        if data_type == "news":
            data["source_url"] = chunk["url"]
            data["source_date"] = chunk["date"]
        else:
            data["source_file"] = chunk["filename"]
            data["source_pages"] = chunk["page_ranges"]
        return data
    except json.JSONDecodeError as e:
        tqdm.write(f"  ⚠️  JSON parse error: {e} | preview: {raw[:120]}")
        return None


# ── Resume support ────────────────────────────────────────────────────────────


def load_completed(path: Path) -> tuple[set, int]:
    done = set()
    good_count = 0
    if not path.exists():
        return done, good_count
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                cid = data.get("chunk_id")
                if cid:
                    done.add(cid)
                if not data.get("parse_error"):
                    good_count += 1
            except Exception:
                pass
    return done, good_count


# ── Zip output ────────────────────────────────────────────────────────────────


def zip_output(jsonl_path: Path) -> Path:
    """Compress using LZMA (xz) — best compression available in Python stdlib."""
    xz_path = jsonl_path.with_suffix(".xz")
    with open(jsonl_path, "rb") as f_in:
        with lzma.open(xz_path, "wb", preset=9 | lzma.PRESET_EXTREME) as f_out:
            f_out.write(f_in.read())
    original_mb = jsonl_path.stat().st_size / 1_048_576
    compressed_mb = xz_path.stat().st_size / 1_048_576
    ratio = (1 - compressed_mb / original_mb) * 100 if original_mb else 0
    print(
        f"  Compressed: {jsonl_path.name} -> {xz_path.name}  "
        f"({original_mb:.1f} MB -> {compressed_mb:.1f} MB, {ratio:.0f}% smaller)"
    )
    return xz_path


# ── Run one data type ─────────────────────────────────────────────────────────

LOG_EVERY_N_CHUNKS = 10


async def process_chunk(
    chunk,
    chunk_idx,
    data_type,
    client,
    output_path,
    ultrachat_samples,
    semaphore,
    file_lock,
):
    example = format_ultrachat_example(
        ultrachat_samples[chunk_idx % len(ultrachat_samples)]
    )
    prompt = (
        news_prompt(chunk, example)
        if data_type == "news"
        else research_prompt(chunk, example)
    )
    raw_output = await call_api(client, prompt, semaphore)

    if raw_output is None:
        tqdm.write("  ⏭️  Skipped (all retries failed)")
        return False

    record = parse_json(raw_output, chunk, data_type)

    async with file_lock:
        with open(output_path, "a", encoding="utf-8") as out_f:
            if record:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                fallback = {
                    "chunk_id": chunk["chunk_id"],
                    "raw_output": raw_output,
                    "parse_error": True,
                }
                if data_type == "news":
                    fallback.update(
                        {"source_url": chunk["url"], "category": chunk.get("category")}
                    )
                else:
                    fallback.update(
                        {
                            "source_file": chunk["filename"],
                            "source_pages": chunk["page_ranges"],
                        }
                    )
                out_f.write(json.dumps(fallback, ensure_ascii=False) + "\n")

    if record:
        return True
    else:
        tqdm.write("  ⚠️   Raw output saved (parse failed)")
        return False


async def run_type(
    data_type: str,
    client,
    output_path: Path,
    ultrachat_samples: list,
    logger: ProgressLogger,
    chunks: list,
):

    print(f"\n{'─' * 55}")
    print(f"  Starting {data_type.upper()}")
    print(f"{'─' * 55}")

    completed, good_count = load_completed(output_path)
    pending = [c for c in chunks if c["chunk_id"] not in completed]

    print(
        f"📦  Chunks: {len(chunks):,} total  |  {len(completed):,} done  |  {len(pending):,} remaining"
    )

    if not pending:
        print(f"  ✅  {data_type.upper()} already complete!\n")
        return len(chunks), good_count

    already_done = len(completed)
    await logger.log_start(data_type, total_chunks=len(chunks))

    good_this_run = 0
    file_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(10)  # process up to 10 chunks concurrently

    tasks = []
    for chunk_idx, chunk in enumerate(pending):
        task = asyncio.create_task(
            process_chunk(
                chunk,
                already_done + chunk_idx,
                data_type,
                client,
                output_path,
                ultrachat_samples,
                semaphore,
                file_lock,
            )
        )
        tasks.append(task)

    for i, future in enumerate(
        tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=data_type.upper())
    ):
        success = await future
        if success:
            good_this_run += 1

        if (i + 1) % LOG_EVERY_N_CHUNKS == 0:
            done_so_far = already_done + i + 1
            await logger.log_progress(
                data_type,
                done=done_so_far,
                total=len(chunks),
                good=good_count + good_this_run,
            )

    total_now = len(completed) + len(pending)
    total_good_now = good_count + good_this_run
    await logger.log_done(data_type, total=total_now, good=total_good_now)

    return total_now, total_good_now


# ── Sync local output files to gist at startup ────────────────────────────────


def get_data_chunks(data_type: str, info: dict) -> list:
    csv_path = get_csv(data_type)
    key = "news" if data_type == "news" else "res"
    row_start, row_end = info[f"{key}_start"], info[f"{key}_end"]
    df = pd.read_csv(
        csv_path,
        skiprows=range(1, row_start + 1),
        nrows=row_end - row_start,
    ).reset_index(drop=True)
    if data_type == "news":
        return build_news_chunks(df, row_start)
    else:
        return build_research_chunks(df, row_start)


def sync_local_to_gist(
    output_path: Path, data_type: str, logger: ProgressLogger, chunks: list
):
    """Read the local output file (if it exists) and push an accurate
    done/progress event to the gist before the run starts."""
    if not output_path.exists():
        return

    # Count completed chunks from local output file
    completed, good = load_completed(output_path)
    if not completed:
        return

    grand_total = len(chunks)
    done_count = len(completed)

    if done_count >= grand_total:
        print(
            f"  📂  Syncing completed {data_type} run to gist ({done_count:,} chunks)..."
        )
        logger.force_push(
            {"event": "done", "type": data_type, "total": done_count, "good": good}
        )
    else:
        print(
            f"  📂  Syncing partial {data_type} progress to gist "
            f"({done_count:,}/{grand_total:,} chunks)..."
        )
        logger.force_push(
            {
                "event": "progress",
                "type": data_type,
                "done": done_count,
                "total": grand_total,
                "good": good,
            }
        )


# ── Main ──────────────────────────────────────────────────────────────────────


async def amain():
    parser = argparse.ArgumentParser(description="Ghana LLM Dataset Generator")
    parser.add_argument("--code", required=True, help="Your volunteer code")
    parser.add_argument("--output", default=None, help="Custom output path (optional)")
    args = parser.parse_args()

    info = decode_code(args.code)

    news_label = f"news_{info['news_start']}_{info['news_end']}"
    res_label = f"research_{info['res_start']}_{info['res_end']}"
    output_path = Path(args.output or f"results/volunteer_{news_label}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    news_count = info["news_end"] - info["news_start"]
    res_count = info["res_end"] - info["res_start"]
    # Estimated time is now much faster due to concurrency
    est_hours = (news_count + res_count) * (RETRY_DELAY / 10) / 3600

    gist_configured = (
        PROGRESS_GIST_ID != "YOUR_GIST_ID_HERE" and _TOK_A != "YOUR_FIRST_HALF"
    )
    logger = ProgressLogger(
        gist_id=PROGRESS_GIST_ID,
        gist_token=PROGRESS_GIST_TOKEN,
        api_key=info["api_key"],
        silent=not gist_configured,
    )

    print(f"""
╔══════════════════════════════════════════════════════╗
║       Ghana LLM Dataset Generator — Volunteer        ║
╠══════════════════════════════════════════════════════╣
║  News     : rows {info["news_start"]:,} – {info["news_end"]:,} ({news_count:,} rows){" " * (22 - len(f"{info['news_start']:,} – {info['news_end']:,} ({news_count:,} rows)"))}║
║  Research : rows {info["res_start"]:,} – {info["res_end"]:,} ({res_count:,} rows){" " * (22 - len(f"{info['res_start']:,} – {info['res_end']:,} ({res_count:,} rows)"))}║
║  Model    : {NVIDIA_MODEL:<41} ║
║  Output   : {str(output_path):<41} ║
║  Est. time: ~{est_hours:.1f}h (auto-resumes if interrupted){" " * (18 - len(f"~{est_hours:.1f}h (auto-resumes if interrupted)"))}║
║  Shadow ID: {logger.shadow_name:<41} ║
╚══════════════════════════════════════════════════════╝
""")

    client = make_client(info["api_key"])
    ultrachat_samples = load_ultrachat_samples()

    news_out = output_path.parent / f"{news_label}.jsonl"
    res_out = output_path.parent / f"{res_label}.jsonl"

    print("\n📦  Pre-building chunks from CSVs...")
    news_chunks = get_data_chunks("news", info)
    res_chunks = get_data_chunks("research", info)

    # ── Sync any locally-completed work to gist before starting ───────────
    if gist_configured:
        print("\n🔄  Syncing local progress to gist...")
        sync_local_to_gist(news_out, "news", logger, news_chunks)
        sync_local_to_gist(res_out, "research", logger, res_chunks)

    # ── Run news, then research ────────────────────────────────────────────
    news_total, news_good = await run_type(
        "news", client, news_out, ultrachat_samples, logger, news_chunks
    )
    news_zip = zip_output(news_out)

    res_total, res_good = await run_type(
        "research", client, res_out, ultrachat_samples, logger, res_chunks
    )
    res_zip = zip_output(res_out)

    # ── Final summary ──────────────────────────────────────────────────────
    total = news_total + res_total
    good = news_good + res_good

    print(f"""
╔══════════════════════════════════════════════════════╗
║              🎉  ALL DONE!                           ║
╠══════════════════════════════════════════════════════╣
║  Total records : {total:<35,} ║
║  Parsed OK     : {good:<35,} ║
╚══════════════════════════════════════════════════════╝

📤  Submit your results:
    1. Go to: https://github.com/{GITHUB_REPO}/issues/new?template=result_submission.md
    2. Fill in the form
    3. Attach these two zip files (drag into the text box):
         {news_zip.resolve()}
         {res_zip.resolve()}

Thank you for contributing to the Ghana LLM project! 🇬🇭
""")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
