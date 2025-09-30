import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import aiohttp
import aiosqlite


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# Configuration constants
GITHUB_TOKEN = "your-token-here"
DB_PATH = "data/github_pandas.db"
# db
LIMIT = 100
OFFSET = 0
# output folders
RAW_DIR = "notebooks"
COMPACT_DIR = "compact_notebooks"
# workers
WORKERS_FETCH = 5
WORKERS_CONVERT = 5
MAX_RETRIES = 3
# output type
SAVE_RAW = True
SAVE_SLIM = False
OVERWRITE = True
VERBOSE = False

# mapping csv
MAPPING_CSV = "data/link_hash_map.csv"


# Derived path constants
RAW_DIR_PATH = Path(RAW_DIR)
COMPACT_DIR_PATH = Path(COMPACT_DIR)
MAPPING_CSV_PATH = Path(MAPPING_CSV)

GITHUB_BLOB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$")


def github_blob_to_owner_repo_ref_path(
    file_url: str,
) -> Optional[Tuple[str, str, str, str]]:
    fragment_pos = file_url.find("#")
    if fragment_pos != -1:
        file_url = file_url[:fragment_pos]
    query_pos = file_url.find("?")
    if query_pos != -1:
        file_url = file_url[:query_pos]

    m = GITHUB_BLOB_RE.match(file_url)
    if not m:
        return None
    owner, repo, ref, path = m.group(1), m.group(2), m.group(3), m.group(4)
    return owner, repo, ref, path


class RestRateLimiter:
    """Header-driven limiter using x-ratelimit-* and retry-after per GitHub docs.

    Docs: https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api?apiVersion=2022-11-28
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._remaining: Optional[int] = None
        self._reset_epoch: Optional[int] = None
        self._secondary_block_until: float = 0.0

    async def before_request(self) -> None:
        while True:
            async with self._lock:
                now = time.time()
                sleep_for = 0.0
                if now < self._secondary_block_until:
                    sleep_for = self._secondary_block_until - now
                elif self._remaining is not None and self._reset_epoch is not None:
                    if self._remaining <= 0 and now < self._reset_epoch:
                        sleep_for = max(0.0, self._reset_epoch - now + 0.5)
                if sleep_for <= 0.0:
                    return
            await asyncio.sleep(min(sleep_for, 1))

    async def after_response(self, resp: aiohttp.ClientResponse) -> None:
        remaining_hdr = resp.headers.get("x-ratelimit-remaining")
        reset_hdr = resp.headers.get("x-ratelimit-reset")
        try:
            remaining_val = int(remaining_hdr) if remaining_hdr is not None else None
        except Exception:
            remaining_val = None
        try:
            reset_val = int(reset_hdr) if reset_hdr is not None else None
        except Exception:
            reset_val = None
        async with self._lock:
            if remaining_val is not None:
                self._remaining = remaining_val
            if reset_val is not None:
                self._reset_epoch = reset_val
            if resp.status in (403, 429):
                now = time.time()
                if self._remaining == 0 and self._reset_epoch:
                    self._secondary_block_until = max(
                        self._secondary_block_until, float(self._reset_epoch)
                    )
                else:
                    self._secondary_block_until = max(
                        self._secondary_block_until, now + 5
                    )

    def get_status(self) -> Tuple[Optional[int], Optional[int], float]:
        return self._remaining, self._reset_epoch, self._secondary_block_until


@dataclass
class FetchJob:
    repo_name: str
    file_url: str
    keyword: Optional[str]
    attempts: int = 0
    owner: Optional[str] = None
    repo: Optional[str] = None
    ref: Optional[str] = None
    path: Optional[str] = None


@dataclass
class Stats:
    requests_used: int = 0
    processed_rows: int = 0
    skipped_existing: int = 0
    fetched_ok: int = 0
    failed_fetch: int = 0
    retries: int = 0
    converted_ok: int = 0
    saved_raw_fallback: int = 0
    convert_fail: int = 0
    discarded: int = 0


class CsvMappingWriter:
    def __init__(self, csv_path: Path, verbose: bool = False):
        self.csv_path = csv_path
        self.verbose = verbose
        self._lock = asyncio.Lock()
        self._seen_hashes: Set[str] = set()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists():
            try:
                with open(self.csv_path, "r", encoding="utf-8", newline="") as f_in:
                    reader = csv.DictReader(f_in)
                    for row in reader:
                        h = (row.get("hash") or "").strip()
                        if h:
                            self._seen_hashes.add(h)
            except Exception:
                # If reading fails, recreate with header below on first write
                pass
        else:
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f_out:
                writer = csv.DictWriter(
                    f_out, fieldnames=["repo_name", "repo_link", "file_link", "hash"]
                )
                writer.writeheader()

    async def write(
        self, repo_name: str, repo_link: str, file_link: str, hash_hex: str
    ) -> None:
        async with self._lock:
            if hash_hex in self._seen_hashes:
                return
            # Ensure header exists if file was corrupted/emptied
            need_header = (
                not self.csv_path.exists() or os.path.getsize(self.csv_path) == 0
            )
            with open(self.csv_path, "a", encoding="utf-8", newline="") as f_out:
                writer = csv.DictWriter(
                    f_out, fieldnames=["repo_name", "repo_link", "file_link", "hash"]
                )
                if need_header:
                    writer.writeheader()
                writer.writerow(
                    {
                        "repo_name": repo_name or "",
                        "repo_link": repo_link or "",
                        "file_link": file_link or "",
                        "hash": hash_hex,
                    }
                )
            self._seen_hashes.add(hash_hex)
            if self.verbose:
                logger.info(
                    f"Mapping written | repo_name={repo_name} hash={hash_hex} file_url={file_link}"
                )


async def db_reader(
    fetch_queue: asyncio.Queue,
    stats: Stats,
) -> None:
    query = "SELECT repo_name, file_url, keyword FROM code_snippets ORDER BY id"
    params: Tuple[Any, ...] = ()
    if LIMIT is not None and LIMIT > 0:
        query += " LIMIT ? OFFSET ?"
        params = (LIMIT, OFFSET)
    elif OFFSET:
        query += " LIMIT -1 OFFSET ?"
        params = (OFFSET,)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            async for row in cursor:
                repo_name = row["repo_name"] or ""
                file_url = row["file_url"] or ""
                keyword = row["keyword"] or ""
                stats.processed_rows += 1
                if not file_url:
                    continue
                job = FetchJob(repo_name=repo_name, file_url=file_url, keyword=keyword)
                await fetch_queue.put(job)


async def get_bytes(
    session: aiohttp.ClientSession,
    url: str,
    headers: Dict[str, str],
    timeout: aiohttp.ClientTimeout,
    limiter: RestRateLimiter,
    stats: Stats,
) -> Tuple[int, Optional[bytes], Dict[str, str]]:
    await limiter.before_request()
    async with session.get(url, headers=headers, timeout=timeout) as resp:
        stats.requests_used += 1
        await limiter.after_response(resp)
        status = resp.status
        body: Optional[bytes] = None
        try:
            if status == 200:
                body = await resp.read()
        except Exception:
            body = None
        return status, body, dict(resp.headers)


async def fetch_via_rest(
    name: str,
    session: aiohttp.ClientSession,
    limiter: RestRateLimiter,
    job: FetchJob,
    stats: Stats,
) -> Optional[bytes]:
    # Use a single media type per request as requested: application/vnd.github.raw+json
    headers_one = {
        "User-Agent": "github-notebook-downloader-rest/1.0",
        "Accept": "application/vnd.github.raw+json",
    }
    if GITHUB_TOKEN:
        headers_one["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    parsed = github_blob_to_owner_repo_ref_path(job.file_url)
    if not parsed:
        logger.warning(f"{name}: Skip non-blob URL: {job.file_url}")
        return None
    job.owner, job.repo, job.ref, job.path = parsed

    base_api = f"https://api.github.com/repos/{job.owner}/{job.repo}"
    contents_url = f"{base_api}/contents/{job.path}?ref={job.ref}"

    timeout = aiohttp.ClientTimeout(total=60)

    status_raw, body_raw, _ = await get_bytes(
        session, contents_url, headers_one, timeout, limiter, stats
    )
    if status_raw == 200 and body_raw is not None:
        return body_raw
    if status_raw == 404:
        logger.warning(f"{name}: Not found (404): {contents_url}")
        return None
    if status_raw in (403, 429):
        raise RuntimeError(f"Rate-limited or forbidden {status_raw} for {contents_url}")
    if 500 <= status_raw < 600:
        raise RuntimeError(f"Server error {status_raw} for {contents_url}")
    logger.warning(f"{name}: HTTP {status_raw} for {contents_url}")
    return None


def slim_notebook_bytes(data: bytes) -> Tuple[Optional[bytes], bool]:
    try:
        nb = json.loads(data.decode("utf-8"))
    except Exception:
        return None, False

    cells = nb.get("cells", [])
    slim_cells = []
    for cell in cells:
        cell_type = cell.get("cell_type")
        source_field = cell.get("source", [])
        if isinstance(source_field, list):
            source_value = source_field
        elif isinstance(source_field, str):
            source_value = source_field
        else:
            source_value = ""
        slim_cell = {
            "cell_type": cell_type,
            "source": source_value,
            "metadata": {},
        }
        if cell_type == "code":
            slim_cell["execution_count"] = None
            slim_cell["outputs"] = []
        slim_cells.append(slim_cell)

    meta = nb.get("metadata", {}) or {}
    kernelspec = (
        meta.get("kernelspec") if isinstance(meta.get("kernelspec"), dict) else None
    )
    language_info = (
        meta.get("language_info")
        if isinstance(meta.get("language_info"), dict)
        else None
    )
    slim_meta: Dict[str, Any] = {}
    if kernelspec:
        ks = {k: v for k, v in kernelspec.items() if k in ("name", "display_name")}
        if ks:
            slim_meta["kernelspec"] = ks
    if language_info:
        li = {k: v for k, v in language_info.items() if k in ("name", "version")}
        if li:
            slim_meta["language_info"] = li
    if "orig_nbformat" in nb:
        slim_meta["orig_nbformat"] = nb.get("orig_nbformat")

    slim_nb = {
        "nbformat": nb.get("nbformat", 4),
        "nbformat_minor": nb.get("nbformat_minor", 2),
        "metadata": slim_meta,
        "cells": [{k: v for k, v in c.items() if v is not None} for c in slim_cells],
    }
    try:
        slim_bytes = json.dumps(
            slim_nb, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return slim_bytes, False
    except Exception:
        return None, False


def analyze_notebook_bytes(data: bytes, op: Optional[str] = None) -> Dict[str, bool]:
    try:
        nb = json.loads(data.decode("utf-8"))
    except Exception:
        return {
            "import_pandas": False,
            "has_read_call": False,
            "has_op_call": False,
        }

    code_lines: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            code_lines.append("".join(src))
        elif isinstance(src, str):
            code_lines.append(src)

    code_text = "\n".join(code_lines)
    import_pandas = bool(
        re.search(r"(?m)^\s*(import\s+pandas\b|from\s+pandas\s+import\b)", code_text)
    )
    has_read_call = bool(re.search(r"\.read_[A-Za-z0-9_]*\s*\(", code_text))
    has_op_call = (
        bool(re.search(rf"\.{re.escape(op)}\s*\(", code_text)) if op else False
    )

    return {
        "import_pandas": import_pandas,
        "has_read_call": has_read_call,
        "has_op_call": has_op_call,
    }


def notebook_meets_required_filters(data: bytes, op: Optional[str]) -> bool:
    analysis = analyze_notebook_bytes(data, op)
    return (
        analysis.get("import_pandas", False)
        and analysis.get("has_read_call", False)
        and analysis.get("has_op_call", False)
    )


def atomic_write_bytes(target_path: Path, content: bytes) -> None:
    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, target_path)


async def fetch_worker(
    name: str,
    session: aiohttp.ClientSession,
    limiter: RestRateLimiter,
    fetch_queue: asyncio.Queue,
    process_queue: asyncio.Queue,
    stats: Stats,
) -> None:
    try:
        while True:
            job: FetchJob = await fetch_queue.get()
            try:
                data: Optional[bytes] = None
                attempts = 0
                while attempts <= MAX_RETRIES and data is None:
                    attempts += 1
                    try:
                        data = await fetch_via_rest(
                            name, session, limiter, job, stats
                        )
                    except RuntimeError as e:
                        if attempts <= MAX_RETRIES:
                            delay = 5.0
                            if VERBOSE:
                                logger.info(
                                    f"{name}: Retry in {delay:.1f}s for {job.file_url}: {e}"
                                )
                            await asyncio.sleep(delay)
                            stats.retries += 1
                            continue
                        else:
                            logger.error(
                                f"{name}: Max retries reached for {job.file_url}"
                            )
                            break

                if data is not None:
                    await process_queue.put((job, data))
                    stats.fetched_ok += 1
                    if VERBOSE:
                        ident = (
                            f"{job.owner}/{job.repo}/{job.path}"
                            if job.owner and job.repo and job.path
                            else job.file_url
                        )
                        logger.info(f"Fetched OK: {ident}")
                else:
                    stats.failed_fetch += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"{name}: Fetch error for {job.file_url}: {e}")
                stats.failed_fetch += 1
            finally:
                fetch_queue.task_done()
    except asyncio.CancelledError:
        if VERBOSE:
            logger.info(f"{name}: cancelled")


async def process_worker(
    name: str,
    mapping_writer: CsvMappingWriter,
    process_queue: asyncio.Queue,
    stats: Stats,
) -> None:
    try:
        while True:
            job, data = await process_queue.get()
            try:
                # Apply filter once for all save types
                if not notebook_meets_required_filters(data, job.keyword):
                    stats.discarded += 1
                    # do not write any flavor if the filter fails
                    process_queue.task_done()
                    continue
                
                file_hash = hashlib.sha1(job.file_url.encode("utf-8")).hexdigest()
                fname = f"{file_hash}.ipynb"

                wrote_any = False
                
                # RAW save
                if SAVE_RAW:
                    raw_path = RAW_DIR_PATH / fname
                    need_write_raw = OVERWRITE or not raw_path.exists()
                    if need_write_raw:
                        atomic_write_bytes(raw_path, data)
                        wrote_any = True

                # SLIM save
                if SAVE_SLIM:
                    compact_path = COMPACT_DIR_PATH / fname
                    need_write_slim = OVERWRITE or not compact_path.exists()
                    if need_write_slim:
                        slim_bytes, _ = slim_notebook_bytes(data)
                        if slim_bytes is None:
                            atomic_write_bytes(compact_path, data)
                            stats.saved_raw_fallback += 1
                        else:
                            atomic_write_bytes(compact_path, slim_bytes)
                            stats.converted_ok += 1
                        wrote_any = True

                if not wrote_any and (SAVE_RAW or SAVE_SLIM):
                    # Nothing written because files existed and overwrite=False
                    stats.skipped_existing += 1

                if wrote_any:
                    repo_link = (
                        f"https://github.com/{job.owner}/{job.repo}"
                        if job.owner and job.repo
                        else ""
                    )
                    await mapping_writer.write(
                        repo_name=job.repo_name,
                        repo_link=repo_link,
                        file_link=job.file_url,
                        hash_hex=file_hash,
                    )

                if VERBOSE:
                    logger.info(
                        f"{name}: Processed {job.file_url} | wrote_any={wrote_any}"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"{name}: Convert error for {job.file_url}: {e}")
                stats.convert_fail += 1
            finally:
                process_queue.task_done()
    except asyncio.CancelledError:
        if VERBOSE:
            logger.info(f"{name}: cancelled")


async def progress(stats: Stats, limiter: RestRateLimiter) -> None:
    try:
        noted_zero = False
        while True:
            await asyncio.sleep(10)
            rem, reset_epoch, sec_until = limiter.get_status()

            remaining_val = rem if isinstance(rem, int) else None
            if remaining_val == 0 and not noted_zero:
                noted_zero = True
            elif remaining_val == 0 and noted_zero:
                continue
            else:
                noted_zero = False

            logger.info(
                "Progress | rows=%d used=%d fetched=%d failed=%d skipped=%d converted=%d fallback=%d discarded=%d",
                stats.processed_rows,
                stats.requests_used,
                stats.fetched_ok,
                stats.failed_fetch,
                stats.skipped_existing,
                stats.converted_ok,
                stats.saved_raw_fallback,
                stats.discarded,
            )
            logger.info(
                f"Rate limit status | rate_remaining=%s reset=%s sec_block=%.1f",
                rem if rem is not None else "?",
                reset_epoch if reset_epoch is not None else "?",
                max(0.0, (sec_until - time.time())) if sec_until else 0.0,
            )
    except asyncio.CancelledError:
        return


async def run() -> int:
    if SAVE_RAW:
        RAW_DIR_PATH.mkdir(parents=True, exist_ok=True)
    if SAVE_SLIM:
        COMPACT_DIR_PATH.mkdir(parents=True, exist_ok=True)

    # Ensure mapping CSV dir exists and warm cache
    mapping_writer = CsvMappingWriter(MAPPING_CSV_PATH, verbose=VERBOSE)

    logger.info(
        "Starting REST downloader v2 | fetchers=%d, converters=%d, retries=%d, raw=%s, slim=%s",
        WORKERS_FETCH,
        WORKERS_CONVERT,
        MAX_RETRIES,
        SAVE_RAW,
        SAVE_SLIM,
    )

    fetch_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    process_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    stats = Stats()

    limiter = RestRateLimiter()

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        # Graceful shutdown primitives
        stop_event = asyncio.Event()
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, stop_event.set)
            loop.add_signal_handler(signal.SIGTERM, stop_event.set)
        except NotImplementedError:
            # Signals may not be supported (e.g., on Windows)
            pass

        fetchers = [
            asyncio.create_task(
                fetch_worker(
                    name=f"fetcher-{i+1}",
                    session=session,
                    limiter=limiter,
                    fetch_queue=fetch_queue,
                    process_queue=process_queue,
                    stats=stats,
                )
            )
            for i in range(WORKERS_FETCH)
        ]

        converters = [
            asyncio.create_task(
                process_worker(
                    name=f"converter-{i+1}",
                    mapping_writer=mapping_writer,
                    process_queue=process_queue,
                    stats=stats,
                )
            )
            for i in range(WORKERS_CONVERT)
        ]

        reader_task = asyncio.create_task(db_reader(fetch_queue, stats))
        stopper_task = asyncio.create_task(stop_event.wait())
        progress_task = asyncio.create_task(progress(stats, limiter))

        done, pending = await asyncio.wait(
            {reader_task, stopper_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if stopper_task in done and not reader_task.done():
            logger.info(
                "Received keyboard interrupt - stopping fetchers and letting converters finish..."
            )
            # Ctrl+C before reader finished: stop reader and fetchers immediately
            reader_task.cancel()
            for t in fetchers:
                t.cancel()
            await asyncio.gather(reader_task, *fetchers, return_exceptions=True)

            # Let converters finish remaining items
            await process_queue.join()
            for t in converters:
                t.cancel()
            await asyncio.gather(*converters, return_exceptions=True)
        else:
            # Reader finished. Now race fetch drain with a late Ctrl+C.
            await reader_task
            fetch_join = asyncio.create_task(fetch_queue.join())
            done2, _ = await asyncio.wait(
                {fetch_join, stopper_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if stopper_task in done2 and not fetch_join.done():
                logger.info(
                    "Received keyboard interrupt - stopping fetchers and letting converters finish..."
                )
                # Stop fetchers immediately on late Ctrl+C
                for t in fetchers:
                    t.cancel()
                await asyncio.gather(*fetchers, return_exceptions=True)

                # Then let converters finish
                await process_queue.join()
                for t in converters:
                    t.cancel()
                await asyncio.gather(*converters, return_exceptions=True)
            else:
                # Fetchers drained normally
                for t in fetchers:
                    t.cancel()
                await asyncio.gather(*fetchers, return_exceptions=True)

                await process_queue.join()
                for t in converters:
                    t.cancel()
                await asyncio.gather(*converters, return_exceptions=True)

        progress_task.cancel()
        await asyncio.gather(progress_task, return_exceptions=True)

    logger.info(
        "Done | rows=%d requests=%d fetched=%d failed=%d retries=%d skipped=%d converted=%d fallback=%d discarded=%d",
        stats.processed_rows,
        stats.requests_used,
        stats.fetched_ok,
        stats.failed_fetch,
        stats.retries,
        stats.skipped_existing,
        stats.converted_ok,
        stats.saved_raw_fallback,
        stats.discarded,
    )
    return 0


def main() -> int:
    if not SAVE_RAW and not SAVE_SLIM:
        logger.error("Set at least one of SAVE_RAW or SAVE_SLIM")
        return 1
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
