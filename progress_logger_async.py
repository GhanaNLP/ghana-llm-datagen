"""
progress_logger_async.py — Ghana LLM Dataset Generator
=================================================
Logs volunteer run progress to a shared GitHub Gist asynchronously.

Each volunteer writes to their OWN file inside the gist:
    {shadow_name}.jsonl   e.g.  MerryZorilla197.jsonl

This eliminates the race condition where two volunteers doing a
read-modify-write on the same file would overwrite each other's data.
The dashboard reads all *.jsonl files from the gist and merges them.
"""

import json
import time
import hashlib
import asyncio
import urllib.request
from datetime import datetime, timezone


# ── Shadow name generation ────────────────────────────────────────────────────

ADJECTIVES = [
    "Amber",
    "Brave",
    "Calm",
    "Deft",
    "Eager",
    "Fierce",
    "Glad",
    "Hardy",
    "Idle",
    "Jolly",
    "Keen",
    "Lively",
    "Merry",
    "Noble",
    "Odd",
    "Proud",
    "Quick",
    "Rare",
    "Swift",
    "Tidy",
    "Utter",
    "Vivid",
    "Warm",
    "Exact",
    "Young",
    "Zeal",
    "Bold",
    "Crisp",
    "Dusty",
    "Elder",
    "Fresh",
    "Grand",
    "Hazy",
    "Inky",
    "Jade",
    "Kind",
    "Lunar",
    "Mossy",
    "Neon",
    "Olive",
    "Pale",
    "Quiet",
    "Rusty",
    "Salty",
    "Tawny",
    "Ultra",
    "Violet",
    "Windy",
]

ANIMALS = [
    "Adder",
    "Bison",
    "Cobra",
    "Dingo",
    "Eagle",
    "Finch",
    "Gecko",
    "Heron",
    "Ibis",
    "Jackal",
    "Kudu",
    "Lemur",
    "Mamba",
    "Nyala",
    "Oribi",
    "Panda",
    "Quail",
    "Raven",
    "Sable",
    "Tiger",
    "Urubu",
    "Viper",
    "Whale",
    "Xerus",
    "Yapok",
    "Zebra",
    "Crane",
    "Drake",
    "Egret",
    "Falcon",
    "Grebe",
    "Hyena",
    "Impala",
    "Kite",
    "Loris",
    "Moose",
    "Newt",
    "Otter",
    "Plover",
    "Robin",
    "Stoat",
    "Tapir",
    "Urial",
    "Vole",
    "Wren",
    "Xenops",
    "Yak",
    "Zorilla",
]


def derive_shadow_name(api_key: str) -> str:
    h = hashlib.sha256(api_key.encode()).digest()
    adj = ADJECTIVES[h[0] % len(ADJECTIVES)]
    animal = ANIMALS[h[1] % len(ANIMALS)]
    number = ((h[2] << 8) | h[3]) % 900 + 100
    return f"{adj}{animal}{number}"


# ── Gist helpers ──────────────────────────────────────────────────────────────

GIST_API = "https://api.github.com/gists"
MAX_RETRIES = 3


async def _gist_request(method: str, url: str, token: str, body: dict = None):
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

    def _do_req():
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    return await asyncio.to_thread(_do_req)


async def _fetch_volunteer_log(gist_id: str, token: str, filename: str) -> str:
    """Fetch this volunteer's own log file. Uses the API (not raw URL) to avoid
    GitHub's CDN cache returning stale content."""
    try:
        data = await _gist_request("GET", f"{GIST_API}/{gist_id}", token)
        files = data.get("files", {})
        if filename in files:
            raw_url = files[filename]["raw_url"]

            def _do_fetch():
                with urllib.request.urlopen(raw_url, timeout=15) as r:
                    return r.read().decode("utf-8")

            return await asyncio.to_thread(_do_fetch)
        return ""  # first push — file doesn't exist yet
    except Exception:
        return ""


async def _push_event(gist_id: str, token: str, filename: str, event: dict) -> bool:
    """Append one event to THIS VOLUNTEER'S OWN file in the gist.
    No conflict with other volunteers since each writes only to their own file."""
    for attempt in range(MAX_RETRIES):
        try:
            existing = await _fetch_volunteer_log(gist_id, token, filename)
            new_line = json.dumps(event, ensure_ascii=False)
            updated = (existing.rstrip("\n") + "\n" + new_line + "\n").lstrip("\n")
            await _gist_request(
                "PATCH",
                f"{GIST_API}/{gist_id}",
                token,
                {"files": {filename: {"content": updated}}},
            )
            return True
        except Exception:
            await asyncio.sleep(4 * (2**attempt))
    return False


# ── Public API ────────────────────────────────────────────────────────────────


class ProgressLoggerAsync:
    def __init__(
        self, gist_id: str, gist_token: str, api_key: str, silent: bool = False
    ):
        self.gist_id = gist_id
        self.gist_token = gist_token
        self.shadow_name = derive_shadow_name(api_key)
        self.filename = f"{self.shadow_name}.jsonl"  # volunteer-private file
        self.silent = silent
        self._last_push = 0.0
        self._push_interval = 60

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    async def _push(self, event: dict):
        now = time.time()
        if now - self._last_push < self._push_interval:
            return
        self._last_push = now
        event.update({"volunteer": self.shadow_name, "ts": self._now()})
        ok = await _push_event(self.gist_id, self.gist_token, self.filename, event)
        if not self.silent:
            print(
                f"  {'📡 Progress logged' if ok else '⚠️  Progress log failed (offline?)'} [{self.shadow_name}]"
            )

    async def force_push(self, event: dict):
        self._last_push = 0.0
        await self._push(event)
        self._last_push = 0.0

    async def log_start(self, data_type: str, total_chunks: int):
        await self.force_push(
            {"event": "start", "type": data_type, "total": total_chunks}
        )

    async def log_progress(self, data_type: str, done: int, total: int, good: int):
        await self._push(
            {
                "event": "progress",
                "type": data_type,
                "done": done,
                "total": total,
                "good": good,
            }
        )

    async def log_done(self, data_type: str, total: int, good: int):
        await self.force_push(
            {"event": "done", "type": data_type, "total": total, "good": good}
        )
