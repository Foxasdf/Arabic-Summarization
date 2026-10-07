import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp313-cp313-win_amd64.whl"
TARGET_FILE = "torch-2.6.0+cu124-cp313-cp313-win_amd64.whl"
TOTAL_SIZE = 2532308219
NUM_WORKERS = 12
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB per job

def download_range(start, end, job_id, total_jobs):
    req = urllib.request.Request(
        URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                expected = end - start + 1
                if len(data) != expected:
                    raise IOError(f"Job {job_id}: expected {expected} bytes, got {len(data)}")
                return start, data, job_id
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1 + attempt)

def main():
    if not os.path.exists(TARGET_FILE):
        print("Target file not found!")
        sys.exit(1)

    cur_size = os.path.getsize(TARGET_FILE)
    print(f"Current size: {cur_size:,} / {TOTAL_SIZE:,} bytes ({cur_size/TOTAL_SIZE*100:.1f}%)")
    if cur_size >= TOTAL_SIZE:
        print("File is already fully downloaded!")
        return

    # Generate ranges for remaining parts
    jobs = []
    offset = cur_size
    job_id = 0
    while offset < TOTAL_SIZE:
        end = min(offset + CHUNK_SIZE - 1, TOTAL_SIZE - 1)
        jobs.append((offset, end))
        offset = end + 1
        job_id += 1

    total_jobs = len(jobs)
    remaining_bytes = TOTAL_SIZE - cur_size
    print(f"Remaining: {remaining_bytes/1024/1024:.1f} MB across {total_jobs} chunks with {NUM_WORKERS} parallel workers.")

    # Open file for in-place writing
    t0 = time.time()
    downloaded_bytes = 0
    with open(TARGET_FILE, "r+b") as f:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_job = {
                executor.submit(download_range, start, end, idx + 1, total_jobs): (start, end)
                for idx, (start, end) in enumerate(jobs)
            }
            for future in as_completed(future_to_job):
                start, data, jid = future.result()
                f.seek(start)
                f.write(data)
                f.flush()
                downloaded_bytes += len(data)
                elapsed = time.time() - t0
                speed = (downloaded_bytes / 1024 / 1024) / max(0.1, elapsed)
                pct = (cur_size + downloaded_bytes) / TOTAL_SIZE * 100
                print(f"[{jid:3d}/{total_jobs}] {pct:.1f}% ({speed:.2f} MB/s) - offset {start:,}")

    final_size = os.path.getsize(TARGET_FILE)
    print(f"Done! Final file size: {final_size:,} bytes")

if __name__ == "__main__":
    main()
