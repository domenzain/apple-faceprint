#!/usr/bin/env python3
"""List still-image faces for one Photos person from a *local* Photos.sqlite.

Never commit Photos.sqlite. Copy it out of the library first (Photos may have
it open), then point --db at the copy.

Default library path on macOS:

  ~/Pictures/Photos Library.photoslibrary/database/Photos.sqlite

Example:

  cp "$HOME/Pictures/Photos Library.photoslibrary/database/Photos.sqlite" /tmp/Photos.sqlite
  python tools/list_person_faces.py --db /tmp/Photos.sqlite --person 2309 -o faces.csv

Filters match the full-body mining pass used for LoRA crop sourcing:
still photos (ZKIND=0), not screenshot/screen recording, not hidden/trashed,
face size <= 0.2, at most 4 faces in the frame.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

DEFAULT_DB = Path.home() / "Pictures/Photos Library.photoslibrary/database/Photos.sqlite"

SQL = """
SELECT
  f.Z_PK AS face_pk,
  f.ZSIZE AS face_size,
  f.ZCENTERX AS center_x,
  f.ZCENTERY AS center_y,
  a.Z_PK AS asset_pk,
  a.ZUUID AS uuid,
  a.ZWIDTH AS width,
  a.ZHEIGHT AS height,
  datetime(a.ZDATECREATED + 978307200, 'unixepoch') AS date_created,
  a.ZDIRECTORY AS directory,
  a.ZFILENAME AS filename,
  (SELECT COUNT(*) FROM ZDETECTEDFACE f2 WHERE f2.ZASSETFORFACE = a.Z_PK) AS n_faces
FROM ZDETECTEDFACE f
JOIN ZASSET a ON a.Z_PK = f.ZASSETFORFACE
WHERE f.ZPERSONFORFACE = ?
  AND IFNULL(a.ZTRASHEDSTATE, 0) = 0
  AND IFNULL(a.ZHIDDEN, 0) = 0
  AND IFNULL(a.ZISDETECTEDSCREENSHOT, 0) = 0
  AND IFNULL(a.ZKINDSUBTYPE, 0) != 10
  AND IFNULL(a.ZKIND, 0) = 0
  AND f.ZSIZE <= ?
  AND (SELECT COUNT(*) FROM ZDETECTEDFACE f2 WHERE f2.ZASSETFORFACE = a.Z_PK) <= ?
ORDER BY a.ZDATECREATED, a.Z_PK
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Photos.sqlite path (default: {DEFAULT_DB})",
    )
    p.add_argument("--person", type=int, required=True, help="ZPERSON.Z_PK (Photos person id)")
    p.add_argument("--max-face-size", type=float, default=0.2)
    p.add_argument("--max-faces", type=int, default=4)
    p.add_argument("-o", "--output", type=Path, required=True, help="CSV output path")
    args = p.parse_args()

    if not args.db.is_file():
        raise SystemExit(
            f"Database not found: {args.db}\n"
            "Copy Photos.sqlite out of the .photoslibrary first; do not commit it."
        )

    # Read-only URI so a live library copy is safer.
    uri = f"file:{args.db.resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    rows = list(con.execute(SQL, (args.person, args.max_face_size, args.max_faces)))
    con.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "face_pk",
        "face_size",
        "center_x",
        "center_y",
        "asset_pk",
        "uuid",
        "width",
        "height",
        "date_created",
        "directory",
        "filename",
        "n_faces",
    ]
    with args.output.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Originals live under: <library>/originals/<directory>/<filename>")


if __name__ == "__main__":
    main()
