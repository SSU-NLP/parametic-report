#!/usr/bin/env python3
"""Re-point the seed SQLite DB at THIS machine's artifact paths.

The platform stores an absolute `artifact_root` per analysis (the server path
where results were downloaded). On your laptop that path doesn't exist, so the
API would 404 on figures. This rewrites `artifact_root` / `manifest_path` for
every row to the bundled `seed/platform_artifacts/<cache_key>` next to this
script — wherever you extracted it. Safe to re-run (idempotent). Stdlib only.

Usage:
    python seed/seed_db.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent
DB_PATH = SEED_DIR / "parametic.db"
ARTIFACTS = SEED_DIR / "platform_artifacts"


def main() -> int:
    if not DB_PATH.exists():
        print(f"!! {DB_PATH} not found", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT cache_key, status FROM analysis_requests").fetchall()
        updated = 0
        for cache_key, status in rows:
            root = ARTIFACTS / cache_key
            if not root.is_dir():
                print(f"   skip {cache_key[:12]} ({status}) — no artifact dir bundled")
                continue
            manifest = root / "manifest.json"
            conn.execute(
                "UPDATE analysis_requests SET artifact_root = ?, manifest_path = ? "
                "WHERE cache_key = ?",
                (str(root), str(manifest) if manifest.exists() else None, cache_key),
            )
            updated += 1
            print(f"   set  {cache_key[:12]} ({status}) -> {root}")
        conn.commit()
        print(f"\nOK — re-pointed {updated}/{len(rows)} rows at {ARTIFACTS}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
