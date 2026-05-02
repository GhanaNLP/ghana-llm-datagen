import base64
import json

def encode(ns, ne, rs, re, api_key):
    payload = json.dumps({
        "ns": ns, "ne": ne,
        "rs": rs, "re": re,
        "k": api_key,
    }, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")

def decode(code):
    # Add padding back
    padded = code + "=" * (4 - len(code) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))

# ─── CONFIGURE HERE ───────────────────────────────────────────
VOLUNTEER_CODE = "eyJucyI6NDY0MTIsIm5lIjo2OTYxOCwicnMiOjk1NzYwLCJyZSI6MTQzNjQwLCJrIjoibnZhcGktS2s5Ty1ZMTVlOGk2TzR1TXdSUjJyWWlMbXRHQWFab09pQVVCQzl0STg1RUNhSWJQRUdCY2owMW9kcnNVbW9sVCJ9"  # paste the old code here
NEW_API_KEY    = "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"   # paste the replacement key here
# ──────────────────────────────────────────────────────────────

old = decode(VOLUNTEER_CODE)
print(f"Original indices → news: {old['ns']}–{old['ne']}  |  research: {old['rs']}–{old['re']}")

new_code = encode(old["ns"], old["ne"], old["rs"], old["re"], NEW_API_KEY)
print(f"\nNew code:\n{new_code}")
