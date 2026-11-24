"""
Collect 100,000+ additional Korean Steam reviews across diverse popular games
using Steam's IUserReviewsService/GetAppReviews/v1 API and merge with existing
data/reviews_raw.csv (deduplicated by review_id).
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "reviews_raw.csv"

# (appid, game_name, target_new_reviews)
TARGET_GAMES = [
    (2246340, "Monster Hunter Wilds", 8000),
    (1091500, "Cyberpunk 2077", 8000),
    (1623730, "Palworld", 7000),
    (553850, "HELLDIVERS 2", 6000),
    (2358720, "Black Myth: Wukong", 6000),
    (578080, "PUBG: BATTLEGROUNDS", 8000),
    (1245620, "ELDEN RING", 7000),
    (582010, "Monster Hunter: World", 7000),
    (730, "Counter-Strike 2", 8000),
    (1172470, "Apex Legends", 7000),
    (1049590, "Eternal Return", 7000),
    (271590, "Grand Theft Auto V", 6000),
    (1174180, "Red Dead Redemption 2", 6000),
    (1086940, "Baldur's Gate 3", 6000),
    (108600, "Project Zomboid", 6000),
    (381210, "Dead by Daylight", 6000),
    (292030, "The Witcher 3: Wild Hunt", 5000),
    (252490, "Rust", 5000),
    (413150, "Stardew Valley", 5000),
    (1446780, "Monster Hunter Rise", 5000),
]

HANGUL_RE = re.compile(r"[가-힣]")
API_BASE = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/?input_json="


def fetch_page(appid: int, cursor: str, num_per_page: int = 100) -> dict | None:
    payload = {
        "appid": appid,
        "languages": ["koreana"],
        "filter": 1,  # recent
        "purchase_type": 1,
        "review_type": 0,  # all
        "num_per_page": num_per_page,
        "cursor": cursor,
    }
    url = API_BASE + urllib.parse.quote(json.dumps(payload, separators=(",", ":")))
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response")
        except Exception as exc:
            wait = 1.0 * (attempt + 1)
            if "429" in str(exc):
                wait = 3.0 * (attempt + 1)
            time.sleep(wait)
    return None


def collect_game(appid: int, game_name: str, target: int, existing_ids: set[str]) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    seen_local: set[str] = set()
    visited_cursors: set[str] = set()
    cursor = "*"
    max_pages = (target // 100) + 25

    for _ in range(max_pages):
        if len(collected) >= target or not cursor or cursor in visited_cursors:
            break
        visited_cursors.add(cursor)
        body = fetch_page(appid, cursor, 100)
        if not body:
            break
        batch = body.get("reviews") or []
        if not batch:
            break

        for r in batch:
            rid = str(r.get("recommendationid") or "").strip()
            if not rid or rid in existing_ids or rid in seen_local:
                continue
            if r.get("language") and r.get("language") != "koreana":
                continue
            text = (r.get("review") or "").strip()
            if not text or not HANGUL_RE.search(text):
                continue
            seen_local.add(rid)
            playtime_min = (r.get("author") or {}).get("playtime_at_review") or (r.get("author") or {}).get("playtime_forever") or 0
            collected.append(
                {
                    "app_id": str(appid),
                    "game_name": game_name,
                    "review_id": rid,
                    "language": "koreana",
                    "review_text": text,
                    "recommended": str(bool(r.get("voted_up"))),
                    "playtime_at_review_hours": f"{playtime_min / 60:.2f}",
                }
            )
            if len(collected) >= target:
                break

        next_cursor = body.get("cursor") or ""
        if not next_cursor or next_cursor in visited_cursors or len(batch) < 50:
            break
        cursor = next_cursor
        time.sleep(0.05)

    print(f"  [{game_name}] +{len(collected)} reviews", flush=True)
    return collected


def main() -> None:
    existing_rows: list[dict[str, str]] = []
    existing_ids: set[str] = set()

    if RAW_CSV.exists():
        with RAW_CSV.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rid = (r.get("review_id") or "").strip()
                txt = (r.get("review_text") or "").strip()
                if rid and rid not in existing_ids and txt and HANGUL_RE.search(txt):
                    existing_ids.add(rid)
                    existing_rows.append(
                        {
                            "app_id": r.get("app_id", ""),
                            "game_name": r.get("game_name", ""),
                            "review_id": rid,
                            "language": "koreana",
                            "review_text": txt,
                            "recommended": r.get("recommended", ""),
                            "playtime_at_review_hours": r.get("playtime_at_review_hours", ""),
                        }
                    )

    print(f"Existing Korean reviews: {len(existing_rows)}", flush=True)
    t0 = time.time()

    new_rows: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(collect_game, appid, name, target, existing_ids)
            for appid, name, target in TARGET_GAMES
        ]
        for fut in concurrent.futures.as_completed(futures):
            batch = fut.result()
            for row in batch:
                if row["review_id"] not in existing_ids:
                    existing_ids.add(row["review_id"])
                    new_rows.append(row)

    all_rows = existing_rows + new_rows
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "app_id",
                "game_name",
                "review_id",
                "language",
                "review_text",
                "recommended",
                "playtime_at_review_hours",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.time() - t0
    print(
        f"Done in {elapsed:.1f}s! Added {len(new_rows):,} new reviews -> Total {len(all_rows):,} reviews in {RAW_CSV}",
        flush=True,
    )


if __name__ == "__main__":
    main()
