import sqlite3
import json

conn = sqlite3.connect("backend/data/meetings.db")
conn.row_factory = sqlite3.Row
latest = conn.execute("SELECT * FROM meetings ORDER BY upload_time DESC LIMIT 1").fetchone()
if latest:
    d = dict(latest)
    print("ID:", d["id"], "Filename:", d["filename"])
    t = json.loads(d["transcript_json"]) if d["transcript_json"] else []
    print("Total segments:", len(t))
    for idx, s in enumerate(t):
        txt = s.get("text", "")
        print(f"[{idx}] {s.get('start')}s - {s.get('end')}s ({s.get('speaker')}): {txt}")
conn.close()
