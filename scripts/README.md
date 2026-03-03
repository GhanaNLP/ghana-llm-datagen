# Data Curator Reference — Ghana LLM Datagen

Everything you need to run this LLM data generation project from start to finish.

---

## Repo Structure

```
ghana-llm-datagen/
├── run.py                          ← volunteers run this (do not edit after sharing)
├── requirements.txt
├── ultrachat_sample.csv            ← style reference, commit this to the repo
├── .env                            ← your API keys, NEVER commit this
├── scripts/
│   ├── DATA_CURATOR_README.md      ← you are here
│   ├── generate_codes.py           ← generate volunteer codes
│   └── merge_results.py            ← auto-fetches submissions and merges them
└── .github/
    └── ISSUE_TEMPLATE/
        └── result_submission.md    ← pre-fills the GitHub submission form
```

---

## One-Time Setup

### 1. Fill in `.env` with your NVIDIA API keys
```
NVIDIA_KEY_1=nvapi-...
NVIDIA_KEY_2=nvapi-...
NVIDIA_KEY_3=nvapi-...
NVIDIA_KEY_4=nvapi-...
NVIDIA_KEY_5=nvapi-...
```

### 2. Fill in `generate_codes.py`
Open `scripts/generate_codes.py` and set:
```python
NEWS_CSV_PATH     = "/path/to/news_data.csv"
RESEARCH_CSV_PATH = "/path/to/research_data.csv"
NUM_VOLUNTEERS    = 5
```

### 3. Update `run.py` config block
Open `run.py` and set these four lines near the top:
```python
GITHUB_REPO       = "yourusername/ghana-llm-datagen"
RELEASE_TAG       = "v1.0-data"
NEWS_FILENAME     = "news_data.csv"
RESEARCH_FILENAME = "research_data.csv"
```

### 4. Update `merge_results.py` config line
Open `scripts/merge_results.py` and set:
```python
GITHUB_REPO = "yourusername/ghana-llm-datagen"
```

### 5. Upload data files to GitHub Releases
- Go to your repo → **Releases** → **Create a new release**
- Tag: `v1.0-data`
- Title: `Dataset Files`
- Attach both `news_data.csv` and `research_data.csv` to the same release
- Publish

### 6. Commit everything to GitHub
```bash
git add run.py requirements.txt ultrachat_sample.csv \
        scripts/ .github/ README.md .gitignore
git commit -m "Initial setup"
git push
```
Make sure `.env` and `volunteer_codes.json` are **never** committed — they are already in `.gitignore`.

---

## Generating Volunteer Codes

```bash
python scripts/generate_codes.py
```

The script reads your CSV files, counts the rows automatically, and prints a
table of 5 codes — one per volunteer. Each code covers both a news slice and
a research slice, so volunteers run it once and both datasets get processed.

```
#   NEWS ROWS              N.COUNT   RESEARCH ROWS          R.COUNT   CODE
1   0 – 2,400              2,400     0 – 1,500              1,500     eyJ0Ijo...
2   2,400 – 4,800          2,400     1,500 – 3,000          1,500     eyJ0Ijo...
...
```

A backup with all codes and keys is saved to `volunteer_codes.json`.
Send **one code per volunteer** — keep `volunteer_codes.json` private.

---

## Tracking Volunteer Progress

Submissions come in as GitHub issues tagged **results**. View them all at:
```
https://github.com/YOUR_USERNAME/ghana-llm-datagen/issues?q=label%3Aresults
```

Each submission has two `.xz` compressed result files attached — one for news,
one for research. You do not need to download these manually.

---

## Merging Results

Run this one command when you are ready to merge:

```bash
python scripts/merge_results.py
```

The script automatically:
1. Fetches all issues labelled `results` from your GitHub repo
2. Downloads every `.xz` attachment from each issue
3. Skips files already downloaded — safe to re-run as more submissions arrive
4. Merges everything into `final_dataset.jsonl`
5. Deduplicates by `chunk_id` across all files
6. Excludes any records that failed JSON parsing
7. Prints a per-file and overall summary

**Public repo** — no setup needed, works immediately.
**Private repo** — the script detects this, prompts for a GitHub token once,
then saves it to `.github_token` (gitignored) and never asks again.
To create a token: https://github.com/settings/tokens/new (tick **repo** scope).

Output is written to `final_dataset.jsonl` in the current directory.

---

## About the Output Files

Volunteers produce `.xz` compressed files (LZMA maximum compression).
These are typically **90–99% smaller** than the raw `.jsonl` — a 100MB result
file becomes 1–5MB, well within GitHub's 25MB attachment limit.
`merge_results.py` decompresses them automatically — no manual extraction needed.

---

## If a Volunteer Has Problems

**Auth error (401/403) during their run:**
Their API key may be exhausted. Generate a fresh code using a replacement key
and resend it. Their existing progress is saved — they resume from where they
left off with the new code.

**They need to restart from scratch:**
They delete their local `results/` folder and re-run with the same code.

**They processed incorrect rows:**
Not a problem — chunk IDs are unique per row range. Their output won't
overlap with other volunteers and `merge_results.py` deduplicates anyway.

**You need more volunteers mid-run:**
Increase `NUM_VOLUNTEERS` in `generate_codes.py` and regenerate. New codes
will cover only the rows not yet assigned. The final merge handles everything.

---

## Key Files — Quick Reference

| File | Who uses it | Purpose |
|---|---|---|
| `.env` | Data curator only | NVIDIA API keys |
| `scripts/generate_codes.py` | Data curator only | Generates volunteer codes |
| `volunteer_codes.json` | Data curator only | Backup of all codes + keys |
| `run.py` | Volunteers | The only script volunteers touch |
| `ultrachat_sample.csv` | Auto-loaded by `run.py` | UltraChat style reference for prompts |
| `scripts/merge_results.py` | Data curator only | Auto-fetches and merges all submissions |
| `.github_token` | Data curator only | Saved GitHub token (private repos only) |
| `results/*.xz` | Auto-managed | Downloaded automatically by merge script |
