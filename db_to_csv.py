import csv
import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ---- Configuration (edit these constants) ----
# Path to the SQLite database file to export from
SRC_DB_PATH: Path = Path("data") / "github_pandas.db"

# Destination directory to put CSV files into
DST_DIR: Path = Path("data")

# Tables to export. Examples:
#   None -> export all tables (and optionally views if INCLUDE_VIEWS=True)
#   ["table_a", "table_b"] -> export only these tables
#   ["table_a, table_b"] -> comma-separated also supported
TABLES: Optional[List[str]] = ["code_snippets"]

# Whether to include views when TABLES is not specified
INCLUDE_VIEWS: bool = False

# Overwrite existing CSV files
OVERWRITE: bool = False

# Number of rows fetched per batch when writing CSV
CHUNK_SIZE: int = 5000
# ---------------------------------------------


def ensure_output_directory(directory_path: Path) -> None:
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    connection = sqlite3.connect(str(db_path))
    # Return rows as tuples (default). For named access use Row factory, but
    # we only need column names once for the header.
    return connection


def discover_tables_and_views(
    connection: sqlite3.Connection, include_views: bool
) -> List[Tuple[str, str]]:
    """Return a list of (name, type) for user tables (and optionally views).

    Excludes internal sqlite_ entities.
    """
    cursor = connection.cursor()
    entity_types = ("table", "view") if include_views else ("table",)
    placeholders = ",".join(["?"] * len(entity_types))
    query = (
        f"SELECT name, type FROM sqlite_master "
        f"WHERE type IN ({placeholders}) AND name NOT LIKE 'sqlite_%' "
        f"ORDER BY name"
    )
    cursor.execute(query, entity_types)
    results = [(row[0], row[1]) for row in cursor.fetchall()]
    cursor.close()
    return results


def normalize_table_list(raw_tables: Optional[Iterable[str]]) -> Optional[List[str]]:
    if raw_tables is None:
        return None
    table_names: List[str] = []
    for entry in raw_tables:
        for name in str(entry).split(","):
            cleaned = name.strip()
            if cleaned:
                table_names.append(cleaned)
    # Deduplicate while preserving order
    seen = set()
    unique_tables = []
    for name in table_names:
        if name not in seen:
            seen.add(name)
            unique_tables.append(name)
    return unique_tables


def export_table_to_csv(
    connection: sqlite3.Connection,
    table_name: str,
    output_file: Path,
    chunk_size: int,
    overwrite: bool,
) -> None:
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. Use --overwrite to replace it."
        )

    cursor = connection.cursor()
    # Use double quotes to quote identifiers that may include reserved words
    select_sql = f'SELECT * FROM "{table_name}"'
    cursor.execute(select_sql)

    # Retrieve column names for header
    description = cursor.description
    if description is None:
        cursor.close()
        raise RuntimeError(f"Failed to retrieve columns for table: {table_name}")
    column_names = [col[0] for col in description]

    ensure_output_directory(output_file.parent)
    with output_file.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(column_names)
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            writer.writerows(rows)

    cursor.close()


def main() -> int:
    db_path = Path(os.path.expanduser(str(SRC_DB_PATH))).resolve()
    output_dir = Path(os.path.expanduser(str(DST_DIR))).resolve()

    try:
        connection = connect_sqlite(db_path)
    except Exception as exc:
        print(f"Error opening database: {exc}", file=sys.stderr)
        return 2

    try:
        requested_tables = normalize_table_list(TABLES)
        if requested_tables:
            entities = [(name, "table") for name in requested_tables]
        else:
            entities = discover_tables_and_views(connection, include_views=INCLUDE_VIEWS)

        if not entities:
            print("No tables (or views) found to export.", file=sys.stderr)
            return 1

        # When user provided tables, verify existence and filter by actual presence
        if requested_tables:
            available = {name for name, _ in discover_tables_and_views(connection, include_views=True)}
            missing = [name for name in requested_tables if name not in available]
            if missing:
                print(
                    "Warning: some requested tables/views were not found: "
                    + ", ".join(missing),
                    file=sys.stderr,
                )
            entities = [(name, "table") for name in requested_tables if name in available]
            if not entities:
                print("None of the requested tables/views exist. Nothing to export.", file=sys.stderr)
                return 1

        print(
            f"Exporting {len(entities)} {'entities' if INCLUDE_VIEWS else 'tables'} "
            f"from {db_path} to {output_dir}..."
        )

        failures: List[Tuple[str, str]] = []
        for name, entity_type in entities:
            # Use .csv even for views; name collisions are handled by overwrite flag
            output_file = output_dir / f"{name}.csv"
            try:
                export_table_to_csv(
                    connection=connection,
                    table_name=name,
                    output_file=output_file,
                    chunk_size=max(1, int(CHUNK_SIZE)),
                    overwrite=bool(OVERWRITE),
                )
                print(f"✔ Exported {entity_type} '{name}' -> {output_file}")
            except Exception as exc:
                failures.append((name, str(exc)))
                print(f"✘ Failed to export '{name}': {exc}", file=sys.stderr)

        if failures:
            print(
                f"Completed with {len(failures)} failures. See messages above.",
                file=sys.stderr,
            )
            return 1

        print("All exports completed successfully.")
        return 0
    finally:
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())


