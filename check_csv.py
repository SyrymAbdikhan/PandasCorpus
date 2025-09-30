import asyncio
import csv
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import requests

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------- Constants ----------------
GITHUB_TOKEN = "your-token-here"

INPUT_CSV_PATH = "data/link_hash_map.csv"
OUTPUT_CSV_PATH = "data/csv_counts.csv"
OUTPUT_FILES_CSV_PATH = "source/csv_files.csv"

WORKER_CONCURRENCY = 10
API_TIMEOUT_SECONDS = 30
MAX_RETRIES_PER_REQUEST = 3
STATS_INTERVAL_SECONDS = 10

OUTPUT_COLS = [
    "repo_name",
    "repo_url",
    "csv_count",
    "found_at",
]

FILES_OUTPUT_COLS = [
    "repo_name",
    "file_url",
]

GITHUB_REPO_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")


# ---------------- Shared Stats/State ----------------
@dataclass
class Stats:
    requests_used: int = 0
    repos_total: int = 0
    repos_done: int = 0
    repos_failed: int = 0
    retries: int = 0
    waits: int = 0


@dataclass
class WaitState:
    until_epoch: float = 0.0
    in_wait: bool = False
    wait_logged: bool = False


stats = Stats()
wait_state = WaitState()


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_repo_from_url(repo_url: str) -> Optional[Tuple[str, str, str]]:
    m = GITHUB_REPO_RE.match(repo_url.strip())
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    repo_name = f"{owner}/{repo}"
    return owner, repo, repo_name


# ---------------- Async GitHub client (requests via asyncio.to_thread) ----------------
class AsyncGitHubClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    async def _get_json(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        attempts = 0
        while attempts < MAX_RETRIES_PER_REQUEST:
            attempts += 1
            try:
                resp = await asyncio.to_thread(
                    self.session.get,
                    url,
                    params=params or {},
                    timeout=API_TIMEOUT_SECONDS,
                )
                stats.requests_used += 1
                if resp.status_code == 403 or resp.status_code == 429:
                    reset_header = resp.headers.get("X-RateLimit-Reset")
                    try:
                        reset_epoch = (
                            float(reset_header) if reset_header is not None else 0.0
                        )
                    except ValueError:
                        reset_epoch = 0.0
                    wait_seconds = max(1.0, reset_epoch - time.time() + 1.0)
                    stats.waits += 1
                    wait_state.in_wait = True
                    wait_state.wait_logged = False
                    wait_state.until_epoch = time.time() + wait_seconds
                    # Single wait per limit event
                    await asyncio.sleep(wait_seconds)
                    wait_state.in_wait = False
                    # retry after wait
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                # Non-rate-limit HTTP errors: retry a couple times
                if attempts >= MAX_RETRIES_PER_REQUEST:
                    logger.warning(f"HTTP error after retries: {e}")
                    return None
                stats.retries += 1
                await asyncio.sleep(min(2 * attempts, 6))
            except requests.RequestException as e:
                if attempts >= MAX_RETRIES_PER_REQUEST:
                    logger.warning(f"Request exception after retries: {e}")
                    return None
                stats.retries += 1
                await asyncio.sleep(min(2 * attempts, 6))
        return None

    async def get_default_branch(self, owner: str, repo: str) -> Optional[str]:
        url = f"https://api.github.com/repos/{owner}/{repo}"
        data = await self._get_json(url)
        if not isinstance(data, dict):
            return None
        return data.get("default_branch")

    async def count_csvs_in_tree(
        self, owner: str, repo: str, ref: str
    ) -> Optional[int]:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}"
        data = await self._get_json(url, params={"recursive": "1"})
        if not isinstance(data, dict):
            return None
        if data.get("message") == "Git Repository is empty.":
            return 0
        count = 0
        for entry in data.get("tree", []) or []:
            if entry.get("type") == "blob" and str(
                entry.get("path", "")
            ).lower().endswith(".csv"):
                count += 1
        return count

    async def count_repo_csvs(self, owner: str, repo: str) -> int:
        # Try default branch, then common fallbacks
        refs: List[str] = []
        default_branch = await self.get_default_branch(owner, repo)
        if default_branch:
            refs.append(default_branch)
        for fb in ("main", "master"):
            if fb not in refs:
                refs.append(fb)
        for ref in refs:
            cnt = await self.count_csvs_in_tree(owner, repo, ref)
            if cnt is not None:
                return cnt
        return 0

    async def list_csvs_in_tree(
        self, owner: str, repo: str, ref: str
    ) -> Optional[List[str]]:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}"
        data = await self._get_json(url, params={"recursive": "1"})
        if not isinstance(data, dict):
            return None
        if data.get("message") == "Git Repository is empty.":
            return []
        csv_paths: List[str] = []
        for entry in data.get("tree", []) or []:
            if entry.get("type") == "blob":
                path = str(entry.get("path", ""))
                if path.lower().endswith(".csv"):
                    csv_paths.append(path)
        return csv_paths

    async def list_repo_csvs(
        self, owner: str, repo: str
    ) -> Tuple[List[str], Optional[str]]:
        # Try default branch, then common fallbacks
        refs: List[str] = []
        default_branch = await self.get_default_branch(owner, repo)
        if default_branch:
            refs.append(default_branch)
        for fb in ("main", "master"):
            if fb not in refs:
                refs.append(fb)
        for ref in refs:
            paths = await self.list_csvs_in_tree(owner, repo, ref)
            if paths is not None:
                return paths, ref
        return [], None


# ---------------- Workers ----------------
async def worker_process_repos(
    name: str,
    client: AsyncGitHubClient,
    repo_queue: "asyncio.Queue[Tuple[str,str,str]]",
    counts_result_queue: "asyncio.Queue[Dict[str,str]]",
    files_result_queue: "asyncio.Queue[Dict[str,str]]",
) -> None:
    while True:
        item = await repo_queue.get()
        if item is None:  # sentinel
            repo_queue.task_done()
            break
        owner, repo, repo_name = item
        repo_url = f"https://github.com/{owner}/{repo}"
        try:
            csv_paths, ref = await client.list_repo_csvs(owner, repo)
            count = len(csv_paths)
            count_row = {
                "repo_name": repo_name,
                "repo_url": repo_url,
                "csv_count": str(count),
                "found_at": now_utc_iso(),
            }
            await counts_result_queue.put(count_row)
            if ref:
                # Emit one row per CSV file with a single HTML URL
                for path in csv_paths:
                    file_html_url = (
                        f"https://github.com/{owner}/{repo}/blob/{ref}/{path}"
                    )
                    file_row = {
                        "repo_name": repo_name,
                        "file_url": file_html_url,
                    }
                    await files_result_queue.put(file_row)
        except Exception as e:
            stats.repos_failed += 1
            logger.warning(f"{repo_name} -> ERROR: {e}")
        finally:
            stats.repos_done += 1
            repo_queue.task_done()


async def writer_worker(
    out_path: str,
    result_queue: "asyncio.Queue[Optional[Dict[str,str]]]",
    fieldnames: List[str],
) -> None:
    f = open(out_path, "w", encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            row = await result_queue.get()
            try:
                if row is None:
                    break
                writer.writerow(row)
            finally:
                result_queue.task_done()
    finally:
        f.close()


async def stats_worker(start_time: float) -> None:
    while True:
        if wait_state.in_wait:
            # Sleep straight until wait is expected to finish
            sleep_for = max(0.0, wait_state.until_epoch - time.time())
            if not wait_state.wait_logged:
                elapsed = time.time() - start_time
                rate = (stats.repos_done / elapsed) if elapsed > 0 else 0.0
                logger.info(
                    "Sleep for %.1fs: %d/%d repos (%.2f repos/s) | req=%d failed=%d waits=%d",
                    sleep_for,
                    stats.repos_done,
                    stats.repos_total,
                    rate,
                    stats.requests_used,
                    stats.repos_failed,
                    stats.waits,
                )
                wait_state.wait_logged = True
            await asyncio.sleep(sleep_for)
            continue
        await asyncio.sleep(STATS_INTERVAL_SECONDS)
        elapsed = time.time() - start_time
        rate = (stats.repos_done / elapsed) if elapsed > 0 else 0.0
        logger.info(
            "Progress: %d/%d repos (%.2f repos/s) | req=%d failed=%d waits=%d",
            stats.repos_done,
            stats.repos_total,
            rate,
            stats.requests_used,
            stats.repos_failed,
            stats.waits,
        )


# ---------------- Orchestration ----------------
async def run() -> int:
    if not os.path.exists(INPUT_CSV_PATH):
        logger.error(f"Input CSV not found: {INPUT_CSV_PATH}")
        return 1

    # Load repos and drop duplicates by repo name
    unique: Dict[str, Tuple[str, str, str]] = {}
    with open(INPUT_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_url = row.get("repo_url").strip()
            parsed = parse_repo_from_url(repo_url)
            if not parsed:
                continue
            owner, repo, repo_name = parsed
            if repo_name not in unique:
                unique[repo_name] = (owner, repo, repo_name)

    repos = list(unique.values())
    stats.repos_total = len(repos)
    if stats.repos_total == 0:
        logger.info("No valid repositories found. Wrote empty output.")
        return 0

    logger.info("Unique GitHub repos to scan: %d", stats.repos_total)

    client = AsyncGitHubClient(GITHUB_TOKEN)
    repo_queue: asyncio.Queue = asyncio.Queue()
    counts_result_queue: asyncio.Queue = asyncio.Queue()
    files_result_queue: asyncio.Queue = asyncio.Queue()

    for item in repos:
        await repo_queue.put(item)

    # Start workers
    workers = [
        asyncio.create_task(
            worker_process_repos(
                f"worker-{i+1}",
                client,
                repo_queue,
                counts_result_queue,
                files_result_queue,
            )
        )
        for i in range(WORKER_CONCURRENCY)
    ]

    counts_writer = asyncio.create_task(
        writer_worker(OUTPUT_CSV_PATH, counts_result_queue, OUTPUT_COLS)
    )
    files_writer = asyncio.create_task(
        writer_worker(OUTPUT_FILES_CSV_PATH, files_result_queue, FILES_OUTPUT_COLS)
    )
    start_time = time.time()
    reporter = asyncio.create_task(stats_worker(start_time))

    # Wait for repos to finish
    await repo_queue.join()
    # Stop workers
    for _ in workers:
        await repo_queue.put(None)
    await asyncio.gather(*workers)

    # Signal writers with None and wait for them
    await counts_result_queue.put(None)
    await files_result_queue.put(None)
    await counts_result_queue.join()
    await files_result_queue.join()
    await asyncio.gather(counts_writer, files_writer)

    # Cancel reporter
    reporter.cancel()
    try:
        await reporter
    except asyncio.CancelledError:
        pass

    elapsed = time.time() - start_time
    rate = (stats.repos_done / elapsed) if elapsed > 0 else 0.0
    logger.info(
        "Done. Repos: %d, failed: %d, requests: %d, waits: %d, elapsed: %.1fs, rate: %.2f repos/s",
        stats.repos_total,
        stats.repos_failed,
        stats.requests_used,
        stats.waits,
        elapsed,
        rate,
    )
    return 0


def main() -> int:
    if not GITHUB_TOKEN:
        logger.warning("No GITHUB_TOKEN set — unauthenticated limit is ~60 req/hour.")
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
