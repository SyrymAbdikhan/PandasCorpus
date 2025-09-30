import os
import json
import csv
import time
import asyncio
from asyncio import Queue, QueueEmpty
from pathlib import Path
from typing import Optional, Tuple

# =========================
# CONSTANTS
# =========================
INPUT_DIR = "notebooks"  # directory with .ipynb files
OUTPUT_CODE_CSV = "source/code_cells.csv"
OUTPUT_MARKDOWN_CSV = "source/markdown_cells.csv"

# Number of worker tasks to parse ipynb files
NUM_WORKERS = max(4, os.cpu_count() or 4)
# Queue sizes to balance throughput and memory usage
FILES_QUEUE_MAXSIZE = 2000
ROWS_QUEUE_MAXSIZE = 20000
# Flush writers at this interval (seconds)
FLUSH_EVERY_SECS = 10
# Max rows to batch per dequeue cycle in writer
WRITER_BATCH_SIZE = 10000
# Whether to write notebook_name as relative path to INPUT_DIR
NOTEBOOK_NAME_RELATIVE = True


class Stats:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.start_monotonic = time.monotonic()
        self.files_discovered = 0
        self.files_processed = 0
        self.code_rows_written = 0
        self.md_rows_written = 0

    async def inc_files_discovered(self, n: int = 1) -> None:
        async with self._lock:
            self.files_discovered += n

    async def inc_files_processed(self, n: int = 1) -> None:
        async with self._lock:
            self.files_processed += n

    async def inc_code_rows(self, n: int = 1) -> None:
        async with self._lock:
            self.code_rows_written += n

    async def inc_md_rows(self, n: int = 1) -> None:
        async with self._lock:
            self.md_rows_written += n

    async def snapshot(self) -> Tuple[int, int, int, int, float]:
        async with self._lock:
            elapsed = time.monotonic() - self.start_monotonic
            return (
                self.files_discovered,
                self.files_processed,
                self.code_rows_written,
                self.md_rows_written,
                elapsed,
            )


async def scan_ipynb_files(root_dir: str, files_q: Queue, stats: Stats) -> None:
    root = Path(root_dir)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".ipynb"):
                continue
            await files_q.put(str(Path(dirpath) / name))
            await stats.inc_files_discovered(1)
        # Yield occasionally to keep the loop responsive
        await asyncio.sleep(0)
    # Signal end of scanning by putting one sentinel per worker
    for _ in range(NUM_WORKERS):
        await files_q.put(None)


async def read_notebook(filepath: str) -> Optional[dict]:
    def _read() -> Optional[dict]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    return await asyncio.to_thread(_read)


def cell_source_to_text(src) -> str:
    if src is None:
        return ""
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return str(src)


def make_notebook_name(nb_path: str) -> str:
    if NOTEBOOK_NAME_RELATIVE:
        try:
            return str(Path(nb_path).resolve().relative_to(Path(INPUT_DIR).resolve()))
        except Exception:
            return Path(nb_path).name
    return Path(nb_path).name


async def worker(files_q: Queue, code_q: Queue, md_q: Queue, stats: Stats) -> None:
    while True:
        nb_path = await files_q.get()
        if nb_path is None:
            files_q.task_done()
            break
        try:
            nb = await read_notebook(nb_path)
            if not nb:
                continue
            cells = nb.get("cells") or []
            nb_name = make_notebook_name(nb_path)
            for idx, cell in enumerate(cells):
                ctype = cell.get("cell_type")
                src = cell_source_to_text(cell.get("source"))
                cell_id = cell.get("id") or str(idx)
                row = (nb_name, cell_id, src)
                if ctype == "code":
                    await code_q.put(row)
                elif ctype == "markdown":
                    await md_q.put(row)
        finally:
            await stats.inc_files_processed(1)
            files_q.task_done()


async def writer(
    code_q: Queue,
    md_q: Queue,
    files_q: Queue,
    code_csv_path: str,
    md_csv_path: str,
    stats: Stats,
) -> None:
    headers = ["notebook_name", "cell_id", "source"]
    Path(code_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(code_csv_path, "w", encoding="utf-8", newline="") as fcode, open(
        md_csv_path, "w", encoding="utf-8", newline=""
    ) as fmd:
        code_writer = csv.writer(fcode)
        md_writer = csv.writer(fmd)
        code_writer.writerow(headers)
        md_writer.writerow(headers)
        fcode.flush()
        fmd.flush()

        last_flush = time.monotonic()
        code_done = False
        md_done = False

        while True:
            did_work = False

            # Drain code queue in batches
            code_batch = []
            for _ in range(WRITER_BATCH_SIZE):
                try:
                    row = code_q.get_nowait()
                except QueueEmpty:
                    break
                if row is None:
                    code_done = True
                else:
                    code_batch.append(row)
                code_q.task_done()
                did_work = True
            if code_batch:
                code_writer.writerows(code_batch)
                await stats.inc_code_rows(len(code_batch))

            # Drain markdown queue in batches
            md_batch = []
            for _ in range(WRITER_BATCH_SIZE):
                try:
                    row = md_q.get_nowait()
                except QueueEmpty:
                    break
                if row is None:
                    md_done = True
                else:
                    md_batch.append(row)
                md_q.task_done()
                did_work = True
            if md_batch:
                md_writer.writerows(md_batch)
                await stats.inc_md_rows(len(md_batch))

            # If nothing to do, brief wait to avoid busy-spin
            if not did_work:
                await asyncio.sleep(0.01)

            now = time.monotonic()
            if now - last_flush >= FLUSH_EVERY_SECS:
                try:
                    fcode.flush()
                    fmd.flush()
                finally:
                    last_flush = now
                # Emit periodic stats
                discovered, processed, code_rows, md_rows, elapsed = (
                    await stats.snapshot()
                )
                files_qsize = files_q.qsize()
                code_qsize = code_q.qsize()
                md_qsize = md_q.qsize()
                remaining = max(0, discovered - processed)
                rps = (processed / elapsed) if elapsed > 0 else 0.0
                print(
                    f"[flush] elapsed={elapsed:.1f}s files: discovered={discovered} processed={processed} remaining={remaining} q(files/code/md)={files_qsize}/{code_qsize}/{md_qsize} rows_written(code/md)={code_rows}/{md_rows} rps={rps:.1f}",
                    flush=True,
                )

            if code_done and md_done and code_q.empty() and md_q.empty():
                # Final flush before exit
                fcode.flush()
                fmd.flush()
                break


async def main_async() -> None:
    files_q = Queue(maxsize=FILES_QUEUE_MAXSIZE)
    code_q = Queue(maxsize=ROWS_QUEUE_MAXSIZE)
    md_q = Queue(maxsize=ROWS_QUEUE_MAXSIZE)
    stats = Stats()

    scan_task = asyncio.create_task(scan_ipynb_files(INPUT_DIR, files_q, stats))

    workers = [
        asyncio.create_task(worker(files_q, code_q, md_q, stats))
        for _ in range(NUM_WORKERS)
    ]

    writer_task = asyncio.create_task(
        writer(code_q, md_q, files_q, OUTPUT_CODE_CSV, OUTPUT_MARKDOWN_CSV, stats)
    )

    # Wait for scanning to finish and all files processed
    await scan_task
    await files_q.join()

    # Signal writer that queues are complete
    await code_q.put(None)
    await md_q.put(None)

    # Wait for workers to exit and queues to drain
    await asyncio.gather(*workers)
    await code_q.join()
    await md_q.join()

    # Wait for writer to finish flushing and exit
    await writer_task


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
