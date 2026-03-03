"""
progress_logger.py — Ghana LLM Dataset Generator
=================================================
Logs volunteer run progress to a shared GitHub Gist.

The Gist contains a single file: "ghana_llm_progress.jsonl"
Each line is one progress event (JSON). New events are appended
by fetching the current content, appending, and PATCHing back.

The Gist ID and a fine-grained PAT (with Gist write scope) are
baked into the volunteer code by the coordinator.
"""

import json
import time
import hashlib
import urllib.request
import urllib.error
import urllib.parse
import random
import string
import re
from datetime import datetime, timezone


# ── Shadow name generation ────────────────────────────────────────────────────
# Deterministically derived from the volunteer API key — same key → same name
# always, so the leaderboard stays consistent across multiple runs.

ADJECTIVES = [
    "Amber", "Brave", "Calm", "Deft", "Eager", "Fierce", "Glad", "Hardy",
    "Idle", "Jolly", "Keen", "Lively", "Merry", "Noble", "Odd", "Proud",
    "Quick", "Rare", "Swift", "Tidy", "Utter", "Vivid", "Warm", "Exact",
    "Young", "Zeal", "Bold", "Crisp", "Dusty", "Elder", "Fresh", "Grand",
    "Hazy", "Inky", "Jade", "Kind", "Lunar", "Mossy", "Neon", "Olive",
    "Pale", "Quiet", "Rusty", "Salty", "Tawny", "Ultra", "Violet", "Windy",
]

ANIMALS = [
    "Adder", "Bison", "Cobra", "Dingo", "Eagle", "Finch", "Gecko", "Heron",
    "Ibis", "Jackal", "Kudu", "Lemur", "Mamba", "Nyala", "Oribi", "Panda",
    "Quail", "Raven", "Sable", "Tiger", "Urubu", "Viper", "Whale", "Xerus",
    "Yapok", "Zebra", "Crane", "Drake", "Egret", "Falcon", "Grebe", "Hyena",
    "Impala", "Kite", "Loris", "Moose", "Newt", "Otter", "Plover", "Robin",
    "Stoat", "Tapir", "Urial", "Vole", "Wren", "Xenops", "Yak", "Zorilla",
]

def derive_shadow_name(api_key: str) -> str:
    """Deterministic shadow name from API key hash."""
    h = hashlib.sha256(api_key.encode()).digest()
    adj    = ADJECTIVES[h[0] % len(ADJECTIVES)]
    animal = ANIMALS[h[1]   % len(ANIMALS)]
    number = ((h[2] << 8) | h[3]) % 900 + 100   # 100–999
    return f"{adj}{animal}{number}"


# ── Gist helpers ──────────────────────────────────────────────────────────────

GIST_API = "https://api.github.com/gists"
LOG_FILENAME = "ghana_llm_progress.jsonl"
MAX_RETRIES = 3


def _gist_request(method: str, url: str, token: str, body: dict = None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "GhanaLLMDatagen/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def fetch_gist_content(gist_id: str, token: str) -> str:
    """Return current JSONL content of the log file in the gist."""
    try:
        data = _gist_request("GET", f"{GIST_API}/{gist_id}", token)
        files = data.get("files", {})
        if LOG_FILENAME in files:
            raw_url = files[LOG_FILENAME]["raw_url"]
            # Fetch raw content (no auth needed for public gist)
            with urllib.request.urlopen(raw_url, timeout=15) as r:
                return r.read().decode("utf-8")
        return ""
    except Exception:
        return ""


def push_event(gist_id: str, token: str, event: dict) -> bool:
    """Append one event line to the gist. Returns True on success."""
    for attempt in range(MAX_RETRIES):
        try:
            existing = fetch_gist_content(gist_id, token)
            new_line  = json.dumps(event, ensure_ascii=False)
            updated   = (existing.rstrip("\n") + "\n" + new_line + "\n").lstrip("\n")
            _gist_request(
                "PATCH",
                f"{GIST_API}/{gist_id}",
                token,
                {"files": {LOG_FILENAME: {"content": updated}}},
            )
            return True
        except Exception as e:
            wait = 4 * (2 ** attempt)
            time.sleep(wait)
    return False


# ── Public API ────────────────────────────────────────────────────────────────

class ProgressLogger:
    """
    Attach to a run to log occasional progress events to the shared Gist.

    Usage:
        logger = ProgressLogger(gist_id, gist_token, api_key)
        logger.log_start("news", total_chunks=120)
        logger.log_progress("news", done=60, total=120, good=58)
        logger.log_done("news", total=120, good=115)
    """

    def __init__(self, gist_id: str, gist_token: str, api_key: str, silent: bool = False):
        self.gist_id     = gist_id
        self.gist_token  = gist_token
        self.shadow_name = derive_shadow_name(api_key)
        self.silent      = silent   # suppress console output if True
        self._last_push  = 0.0
        self._push_interval = 60    # push at most once per minute

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _push(self, event: dict):
        now = time.time()
        if now - self._last_push < self._push_interval:
            return  # rate-limit pushes
        self._last_push = now
        event.update({"volunteer": self.shadow_name, "ts": self._now()})
        ok = push_event(self.gist_id, self.gist_token, event)
        if not self.silent:
            status = "📡 Progress logged" if ok else "⚠️  Progress log failed (offline?)"
            print(f"  {status} [{self.shadow_name}]")

    def force_push(self, event: dict):
        """Push regardless of rate limit (used for start/done events)."""
        self._last_push = 0.0
        self._push(event)
        self._last_push = 0.0  # reset so next interval push still works

    def log_start(self, data_type: str, total_chunks: int):
        self.force_push({
            "event":      "start",
            "type":       data_type,
            "total":      total_chunks,
        })

    def log_progress(self, data_type: str, done: int, total: int, good: int):
        self._push({
            "event": "progress",
            "type":  data_type,
            "done":  done,
            "total": total,
            "good":  good,
        })

    def log_done(self, data_type: str, total: int, good: int):
        self.force_push({
            "event": "done",
            "type":  data_type,
            "total": total,
            "good":  good,
        })
