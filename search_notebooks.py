import logging
import time
import sqlite3
import signal
import csv
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    api_timeout: int = 30
    db_path: str = "data/github_pandas.db"
    queries_path: str = "static/pandas_ops.csv"


class Database:
    """Database handler for raw repositories found by search"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def init_tables(self):
        """Create database tables for repositories and state"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_snippets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_name TEXT NOT NULL,
                    repo_url TEXT NOT NULL,
                    file_url TEXT NOT NULL,
                    search_query TEXT,
                    keyword TEXT,
                    found_at TEXT,
                    UNIQUE(file_url)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_state (
                    id INTEGER PRIMARY KEY,
                    search_index INTEGER,
                    snippets_found INTEGER
                )
            """
            )

            # Create indexes (match simplified schema)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_repo_name ON code_snippets(repo_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_keyword ON code_snippets(keyword)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_query ON code_snippets(search_query)"
            )

            conn.commit()
        logger.info(f"Database initialized: {self.db_path}")

    def save_repos(self, repos: List[Dict]) -> int:
        """Save repositories to database"""
        if not repos:
            return 0

        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.executemany(
                    """
                    INSERT OR IGNORE INTO code_snippets 
                    (repo_name, repo_url, file_url, search_query, keyword, found_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            repo.get("repo_name"),
                            repo.get("repo_url"),
                            repo.get("file_url"),
                            repo.get("search_query"),
                            repo.get("keyword"),
                            repo.get("found_at"),
                        )
                        for repo in repos
                    ],
                )
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                logger.error(f"Error saving repos: {e}")
                return 0

    def load_state(self) -> tuple:
        """Load scraper state"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM code_snippets")
            total_snippets = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT search_index, snippets_found FROM search_state WHERE id = 1"
            )
            result = cursor.fetchone()
            if result:
                search_index, last_count = result
            else:
                search_index, last_count = 0, total_snippets

            return search_index, total_snippets, last_count

    def save_state(self, search_index: int, snippets_found: int):
        """Save scraper state"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO search_state (id, search_index, snippets_found)
                VALUES (1, ?, ?)
            """,
                (search_index, snippets_found),
            )
            conn.commit()

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM code_snippets")
            total = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(DISTINCT repo_name) FROM code_snippets")
            unique_repos = cursor.fetchone()[0]
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT keyword) FROM code_snippets WHERE keyword IS NOT NULL"
            )
            unique_keywords = cursor.fetchone()[0]
            return {
                "total": total,
                "unique_repos": unique_repos,
                "unique_keywords": unique_keywords,
            }


class CodeScraper:
    """GitHub repository-first scraper to find pandas Jupyter notebooks (mirrors repo_search)."""

    def __init__(self, token: str, config: Config = None):
        self.token = token
        self.config = config or Config()
        self.running = True

        # Components
        self.database = Database(self.config.db_path)

        # Setup session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json, application/vnd.github.text-match+json",
            }
        )

        # State
        self.search_index = 0
        self.snippets_found = 0

        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received...")
        self.running = False

    def _sleep_check(self, total_seconds: float, interval_seconds: float = 5.0) -> bool:
        """Sleep up to total_seconds in chunks, checking self.running between chunks.

        Returns True if full sleep completed, or False if interrupted (self.running became False).
        """
        remaining = max(0.0, float(total_seconds))
        end_time = time.time() + remaining
        while self.running and remaining > 0:
            sleep_for = min(interval_seconds, remaining)
            time.sleep(sleep_for)
            remaining = end_time - time.time()
        return self.running

    def _has_target_operation(self, text_matches: List[Dict], keyword: str) -> bool:
        for tm in text_matches:
            fragment = tm.get("fragment", "")
            if f".{keyword}(" in fragment:
                return True
        return False

    def _search_code(self, query: dict, order: str) -> List[dict]:
        logger.info(f"Searching code: '{query.get('query')}' (order={order})")
        all_records: List[Dict[str, Any]] = []
        page = 1
        while self.running:
            if not self.running:
                break

            logger.info(f"Fetching code page {page}...")
            params = {
                "q": query.get("query"),
                "per_page": 100,
                "page": page,
                "sort": "indexed",
                "order": order,
            }
            data = self._make_request("https://api.github.com/search/code", params)
            if data is None:
                logger.error(
                    "No response received; stopping code search for this order."
                )
                break

            items = data.get("items", [])
            if not items:
                logger.info(f"No more code results on page {page}")
                break

            for item in items:
                if not self._has_target_operation(
                    item.get("text_matches", []), query.get("keyword")
                ):
                    continue
                repository = item.get("repository", {})
                record = {
                    "repo_name": repository.get("full_name"),
                    "repo_url": repository.get("html_url"),
                    "file_url": item.get("html_url"),
                    "search_query": query.get("query"),
                    "keyword": query.get("keyword"),
                    "found_at": datetime.now().isoformat(),
                }
                all_records.append(record)

            logger.info(f"Code page {page}: accumulated {len(all_records)} results")
            page += 1

        logger.info(
            f"Code search complete: {len(all_records)} results found (order={order})"
        )
        return all_records

    def _make_request(self, url: str, params: dict = None) -> Optional[Dict[str, Any]]:
        """GET with up to 3 retries; raise for status; delegate HTTP error handling."""
        if not self.running:
            return None

        attempts = 0
        while self.running and attempts < 3:
            attempts += 1
            try:
                response = self.session.get(
                    url, params=params or {}, timeout=self.config.api_timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as http_err:
                retry = self._handle_error_status(response)
                if not retry:
                    return None
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")

            logger.warning(f"Attempts left {3-attempts}/3")
            time.sleep(2)

        return None

    def _handle_error_status(self, response: requests.Response) -> bool:
        """Handle HTTP error responses: True to retry, False to stop."""
        status = response.status_code
        if status == 429 or status == 403:
            reset_header = response.headers.get("X-RateLimit-Reset")
            reset_epoch = float(reset_header) if reset_header is not None else 0.0
            wait_seconds = max(1.0, reset_epoch - time.time() + 1)
            logger.warning(
                f"Too Many Requests. Waiting {wait_seconds:.1f}s before retry."
            )
            return self._sleep_check(wait_seconds, interval_seconds=5.0)

        if status == 422:
            logger.error("Endpoint has been spammed. Stopping...")
            return False

        logger.error(f"Unhandled HTTP error {status}.")
        return True

    def _process_query(self, query: dict) -> int:
        """Process a single query"""
        logger.info(f"Processing query: '{query.get('query')}'")

        try:
            results_asc = self._search_code(query, order="asc")
            results_desc = self._search_code(query, order="desc")

            # Deduplicate by file_url
            combined = {}
            for rec in results_asc + results_desc:
                file_url = rec.get("file_url")
                if not file_url:
                    continue
                combined[file_url] = rec

            results = list(combined.values())
            logger.info(f"Total {len(results)} results found")
            if results:
                saved_count = self.database.save_repos(results)
                self.snippets_found += saved_count
                logger.info(
                    f"Saved {saved_count} new records (duplicates: {len(results) - saved_count})"
                )
                return saved_count
        except Exception as e:
            logger.error(f"Error processing query '{query.get('query')}': {e}")

        return 0

    def run(self):
        """Main scraper loop"""
        logger.info("=" * 60)
        logger.info("Starting Synchronous GitHub Jupyter Pandas Code Scraper")
        logger.info("Searching for essential pandas functions in Jupyter notebooks")
        logger.info("=" * 60)

        try:
            # Initialize
            self.database.init_tables()
            # Load state
            self.search_index, self.snippets_found, last_count = (
                self.database.load_state()
            )
            logger.info(
                f"Loaded state: {self.snippets_found:_} snippets, search index {self.search_index}"
            )
            # Show initial stats
            stats = self.database.get_stats()
            logger.info(
                f"Database stats: {stats['total']:_} rows, {stats['unique_repos']:_} repos, "
                f"{stats.get('unique_keywords', 0):_} unique keywords"
            )

            queries = self._load_queries_from_csv()
            start_time = time.time()

            # Main loop
            while self.running and self.search_index < len(queries):
                query = queries[self.search_index]

                logger.info("=" * 60)
                logger.info(
                    f"Query {self.search_index + 1}/{len(queries)}: '{query.get('query')}'"
                )
                logger.info(f"Progress: {self.snippets_found:_} code snippets")

                try:
                    new_snippets = self._process_query(query)
                    logger.info(f"Query result: {new_snippets} new repos added")

                except Exception as e:
                    logger.error(f"Error processing query '{query}': {e}")

                # Save state and move to next query
                self.database.save_state(self.search_index, self.snippets_found)
                self.search_index += 1

                # Show progress
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = self.snippets_found / elapsed * 3600
                    logger.info(f"Rate: {rate:.1f} snippets/hour")

            # Final stats
            logger.info("=" * 60)
            if self.search_index >= len(queries):
                logger.info("All queries completed!")
            else:
                logger.info("Scraping stopped by user")

            stats = self.database.get_stats()
            elapsed = time.time() - start_time
            logger.info(f"Final stats: {stats['total']:_} total rows")
            logger.info(f"Unique repositories: {stats['unique_repos']:_}")
            if "unique_keywords" in stats:
                logger.info(f"Unique keywords: {stats['unique_keywords']:_}")
            logger.info(f"Session time: {elapsed/3600:.1f} hours")
            if elapsed > 0:
                logger.info(
                    f"Average rate: {self.snippets_found / elapsed * 3600:.1f} snippets/hour"
                )
            logger.info("=" * 60)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.database.save_state(self.search_index, self.snippets_found)
            self.session.close()

    def _load_queries_from_csv(self) -> List[Dict[str, str]]:
        queries: List[Dict[str, str]] = []
        try:
            with open(
                self.config.queries_path, newline="", encoding="utf-8"
            ) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    keyword = row.get("function", "").strip()
                    if not keyword:
                        continue
                    for query in self._construct_repo_queries(keyword):
                        queries.append(
                            {
                                "keyword": keyword,
                                "query": query,
                            }
                        )
        except FileNotFoundError:
            logger.error(f"Queries CSV not found at: {self.config.queries_path}")
        except Exception as e:
            logger.error(f"Failed to load queries from CSV: {e}")
        logger.info(f"Loaded {len(queries)} repo queries from CSV")
        return queries

    def _construct_repo_queries(self, func_name: str) -> List[str]:
        base = f'".{func_name}("'
        ext = "extension:ipynb"
        return [
            f"{base} {ext}",
            f'{base} "import pandas" {ext}',
            f'{base} ".read_" {ext}',
            f'{base} "import pandas" ".read_" {ext}',
        ]


def main():
    """Main entry point"""
    token = "your-token-here"
    if not token:
        logger.error("GitHub token not found!")
        return 1

    config = Config()
    scraper = CodeScraper(token, config)

    try:
        scraper.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
