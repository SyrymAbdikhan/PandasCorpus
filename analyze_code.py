import os
import sys
import time
import ast
import keyword
import logging
import csv
import builtins
import warnings
from logging.handlers import RotatingFileHandler
from collections import Counter
from typing import Dict, Set, Tuple, Optional, List


# =========================
# CONFIG
# =========================
CELLS_PATH = "source/code_cells.csv"
PANDAS_OPS_PATH = "static/pandas_ops.csv"
MPL_FUNCS_PATH = "static/mpl_ops.csv"
SEABORN_FUNCS_PATH = "static/seaborn_ops.csv"
OUT_METRICS_PATH = "data/code_cell_metrics.csv"
OUT_SUMMARY_PATH = "data/pandas_ops_count.csv"

FLUSH_SECS = 10
STATS_SECS = 10

LOG_PATH: Optional[str] = None
LOG_LEVEL = "INFO"

# Recursion settings
RECURSION_LIMIT = 20000
sys.setrecursionlimit(RECURSION_LIMIT)

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


logger: Optional[logging.Logger] = None


# =========================
# CONSTANTS
# =========================
MAGIC_PREFIXES = ("%%", "%", "!")
TRAIN_FUNCS: Set[str] = {
    "fit",
    "fit_transform",
    "partial_fit",
    "train",
    "fit_generator",
}
BUILTIN_NAMES = set(dir(builtins))

# Pandas accessor attribute names that do not break method-call chains
PANDAS_ACCESSOR_ATTRS: Set[str] = {"str", "dt", "cat"}

# Indexing bridge attributes that we consider transparent for LCI chains
INDEX_BRIDGE_ATTRS: Set[str] = {"loc", "iloc", "at", "iat"}


# =========================
# LOGGING
# =========================
def setup_logger(log_path: Optional[str] = None, level: str = "INFO") -> None:
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path:
        fh = RotatingFileHandler(
            log_path, maxBytes=10_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False


# =========================
# HELPERS
# =========================
def is_magic_or_shell(line: str) -> bool:
    s = line.lstrip()
    return any(s.startswith(p) for p in MAGIC_PREFIXES)


def strip_magics_shells(code: str) -> str:
    return "\n".join([ln for ln in code.splitlines() if not is_magic_or_shell(ln)])


def _py2_print_fix(code: str) -> str:
    out_lines: List[str] = []
    for ln in code.splitlines():
        s = ln.lstrip()
        if s.startswith("print") and not s.startswith("print("):
            prefix = ln[: len(ln) - len(s)]
            body = s[5:].strip()
            if body.startswith(">>"):
                out_lines.append(ln)
            elif body == "":
                out_lines.append(prefix + "print()")
            else:
                out_lines.append(f"{prefix}print({body})")
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


def _safe_parse(code: str) -> Tuple[Optional[ast.AST], Optional[str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            return ast.parse(code), None
        except SyntaxError as e:
            if "Missing parentheses in call to 'print'" in str(e):
                fixed = _py2_print_fix(code)
                try:
                    return ast.parse(fixed), None
                except Exception:
                    pass
            return None, f"{type(e).__name__}: {e}"


def call_name(node: ast.Call) -> Optional[str]:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_all_literal_container(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(
            isinstance(e, ast.Constant) or _is_all_literal_container(e)
            for e in node.elts
        )
    if isinstance(node, ast.Dict):
        return all(
            ((k is None) or isinstance(k, ast.Constant) or _is_all_literal_container(k))
            and (isinstance(v, ast.Constant) or _is_all_literal_container(v))
            for k, v in zip(node.keys, node.values)
        )
    return False


def operands_with_containers(tree: ast.AST) -> Tuple[int, int]:
    tags: List[str] = []

    def visit(n: ast.AST) -> None:
        if _is_all_literal_container(n):
            if isinstance(n, ast.List):
                tags.append("LISTLIT")
            elif isinstance(n, ast.Tuple):
                tags.append("TUPLELIT")
            elif isinstance(n, ast.Set):
                tags.append("SETLIT")
            elif isinstance(n, ast.Dict):
                tags.append("DICTLIT")
            return
        if isinstance(n, ast.Constant):
            tags.append(f"CONST:{repr(n.value)}")
            return
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(tree)
    return len(tags), len(set(tags))


def load_functions(path: str) -> Set[str]:
    """Load function names from a CSV with a 'function' header."""
    names: Set[str] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "function" not in (r.fieldnames or []):
            raise ValueError(f"missing 'function' column in {path}")
        for row in r:
            fn = (row.get("function") or "").strip()
            if fn:
                names.add(fn)
    return names


def load_functions_with_categories(path: str) -> Tuple[Set[str], Dict[str, str]]:
    """Load function names and categories from a CSV with 'function' and 'category' headers."""
    names: Set[str] = set()
    func_to_cat: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        missing = {h for h in ("function", "category") if h not in (r.fieldnames or [])}
        if missing:
            raise ValueError(f"missing columns {sorted(missing)} in {path}")
        for row in r:
            fn = (row.get("function") or "").strip()
            cat = (row.get("category") or "").strip()
            if fn:
                names.add(fn)
                func_to_cat[fn] = cat
    return names, func_to_cat


# =========================
# RECURSIVE COLLECTOR
# =========================
class NodeMetricsCollector(ast.NodeVisitor):
    def __init__(
        self, pandas_funcs: Set[str], mpl_funcs: Set[str], seaborn_funcs: Set[str]
    ) -> None:
        self.pandas_funcs = pandas_funcs
        self.mpl_funcs = mpl_funcs
        self.seaborn_funcs = seaborn_funcs

        # Core metrics accumulators
        self.statements = 0
        self.imports = 0
        self.user_defined_functions = 0
        self.total_params = 0
        self.operator_kinds: List[str] = []
        self.identifiers: Set[str] = set()

        # Visualization
        self.pyplot_aliases: Set[str] = {"plt"}
        self.seaborn_aliases: Set[str] = {"sns"}
        self.visual_calls = 0

        # Training calls
        self.train_calls = 0

        # Pandas call chain metrics
        self.pandas_po_counter: Counter = Counter()
        self.pandas_chain_lengths: List[int] = []

        # Index chains (collect subscripts, evaluate after traversal)
        self.subscripts: List[ast.Subscript] = []

        # Nested block depth (computed externally)
        self.max_block_depth = 0

    # ---------- Utility ----------
    def _add_identifier(self, name: Optional[str]) -> None:
        if not name:
            return
        if name not in keyword.kwlist and name not in BUILTIN_NAMES:
            self.identifiers.add(name)

    # ---------- Overrides ----------
    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.stmt):
            self.statements += 1
        super().visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        # Record parent pointers for children to enable upward chain analysis
        for child in ast.iter_child_nodes(node):
            try:
                setattr(child, "_parent", node)
            except Exception:
                pass
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports += 1
        for n in node.names:
            name_for_id = n.asname or n.name.split(".")[-1]
            self._add_identifier(name_for_id)
            # alias detection for pyplot
            if n.name == "matplotlib.pyplot":
                self.pyplot_aliases.add(n.asname or "matplotlib.pyplot")
            elif n.name.startswith("matplotlib.pyplot"):
                self.pyplot_aliases.add(n.asname or n.name)
            elif n.name == "pyplot":
                self.pyplot_aliases.add(n.asname or "pyplot")
            elif n.name == "seaborn":
                self.seaborn_aliases.add(n.asname or "seaborn")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports += 1
        module = node.module or ""
        for n in node.names:
            name_for_id = n.asname or n.name.split(".")[-1]
            self._add_identifier(name_for_id)
        if module == "matplotlib":
            for n in node.names:
                if n.name == "pyplot":
                    self.pyplot_aliases.add(n.asname or "pyplot")
        if module == "seaborn":
            self.seaborn_aliases.add("seaborn")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.user_defined_functions += 1
        self._add_identifier(node.name)
        args = node.args
        self.total_params += len(getattr(args, "posonlyargs", []))
        self.total_params += len(getattr(args, "args", []))
        self.total_params += len(getattr(args, "kwonlyargs", []))
        if getattr(args, "vararg", None):
            self.total_params += 1
        if getattr(args, "kwarg", None):
            self.total_params += 1
        # identifiers for parameters
        for a in getattr(args, "posonlyargs", []):
            self._add_identifier(a.arg)
        for a in getattr(args, "args", []):
            self._add_identifier(a.arg)
        for a in getattr(args, "kwonlyargs", []):
            self._add_identifier(a.arg)
        if getattr(args, "vararg", None):
            self._add_identifier(args.vararg.arg)
        if getattr(args, "kwarg", None):
            self._add_identifier(args.kwarg.arg)
        # Visit body
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # mirror FunctionDef handling
        self.user_defined_functions += 1
        self._add_identifier(node.name)
        args = node.args
        self.total_params += len(getattr(args, "posonlyargs", []))
        self.total_params += len(getattr(args, "args", []))
        self.total_params += len(getattr(args, "kwonlyargs", []))
        if getattr(args, "vararg", None):
            self.total_params += 1
        if getattr(args, "kwarg", None):
            self.total_params += 1
        for a in getattr(args, "posonlyargs", []):
            self._add_identifier(a.arg)
        for a in getattr(args, "args", []):
            self._add_identifier(a.arg)
        for a in getattr(args, "kwonlyargs", []):
            self._add_identifier(a.arg)
        if getattr(args, "vararg", None):
            self._add_identifier(args.vararg.arg)
        if getattr(args, "kwarg", None):
            self._add_identifier(args.kwarg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_identifier(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._add_identifier(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add_identifier(node.attr)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.operator_kinds.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # counts as len(values) - 1
        self.operator_kinds.extend(
            [type(node.op).__name__] * max(0, len(node.values) - 1)
        )
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.operator_kinds.append(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            self.operator_kinds.append(type(op).__name__)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.operator_kinds.extend(["Assign"] * max(1, len(node.targets)))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.operator_kinds.append(f"AugAssign_{type(node.op).__name__}")
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.operator_kinds.append("NamedExpr")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Train calls
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in TRAIN_FUNCS:
            self.train_calls += 1
        elif isinstance(f, ast.Name) and f.id in TRAIN_FUNCS:
            self.train_calls += 1

        # Visualization calls
        if isinstance(f, ast.Attribute):
            base = f.value
            # seaborn and pyplot detection
            if (
                isinstance(base, ast.Name)
                and base.id in self.pyplot_aliases
                and f.attr in self.mpl_funcs
            ):
                self.visual_calls += 1
            elif (
                isinstance(base, ast.Name)
                and base.id in {"ax", "axs", "axes", "axis"}
                and f.attr in self.mpl_funcs
            ):
                self.visual_calls += 1
            elif (
                isinstance(base, ast.Name)
                and base.id in self.seaborn_aliases
                and f.attr in self.seaborn_funcs
            ):
                self.visual_calls += 1
        elif isinstance(f, ast.Name):
            if f.id in self.seaborn_funcs or f.id in self.mpl_funcs:
                self.visual_calls += 1

        # Pandas ops and method-call chains
        nm = call_name(node)
        if nm and nm in self.pandas_funcs:
            self.pandas_po_counter[nm] += 1
            if self._is_pandas_call_tail(node):
                length = self._walk_pandas_call_chain_length(node)
                if length > 0:
                    self.pandas_chain_lengths.append(length)

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.subscripts.append(node)
        self.generic_visit(node)

    # ---------- Chains ----------
    def _is_pandas_call_tail(self, node: ast.Call) -> bool:
        nm = call_name(node)
        if nm not in self.pandas_funcs:
            return False

        # Climb parents to see if this node is used as base for a pandas Call.
        parent = getattr(node, "_parent", None)

        # Skip through accessor attributes like .str/.dt/.cat above the node
        while isinstance(parent, ast.Attribute) and getattr(parent, "value", None) is node and getattr(parent, "attr", None) in PANDAS_ACCESSOR_ATTRS:
            node = parent  # treat the accessor attribute as part of the same object chain
            parent = getattr(node, "_parent", None)

        # Now, if parent is Attribute with value == node, check if its parent is a Call
        if isinstance(parent, ast.Attribute) and getattr(parent, "value", None) is node:
            gp = getattr(parent, "_parent", None)
            if isinstance(gp, ast.Call):
                gp_name = call_name(gp)
                # If the parent call is a pandas function, then current is not tail
                if gp_name in self.pandas_funcs:
                    return False
            # parent exists but is not a pandas Call; current node is tail for pandas chain
            return True

        # Parent is not an Attribute that forms another Call; current node is outermost
        return True

    def _walk_pandas_call_chain_length(self, node: ast.Call) -> int:
        length = 0
        cur: ast.AST = node
        while isinstance(cur, ast.Call):
            nm = call_name(cur)
            if nm in self.pandas_funcs:
                length += 1
                f = cur.func
                if isinstance(f, ast.Attribute):
                    cur = f.value
                    while (
                        isinstance(cur, ast.Attribute)
                        and getattr(cur, "attr", None) in PANDAS_ACCESSOR_ATTRS
                    ):
                        cur = cur.value
                else:
                    break
            else:
                break
        return length

    def enumerate_index_chain_lengths(self) -> List[int]:
        # Find heads: a Subscript whose base is not another Subscript (ignoring bridge attributes)
        non_heads: Set[ast.Subscript] = set()
        for parent in self.subscripts:
            base: ast.AST = parent.value
            while (
                isinstance(base, ast.Attribute)
                and getattr(base, "attr", None) in INDEX_BRIDGE_ATTRS
            ):
                base = base.value
            if isinstance(base, ast.Subscript):
                non_heads.add(base)

        lengths: List[int] = []
        for node in self.subscripts:
            if node in non_heads:
                continue
            length = 1
            base: ast.AST = node.value
            while True:
                if (
                    isinstance(base, ast.Attribute)
                    and getattr(base, "attr", None) in INDEX_BRIDGE_ATTRS
                ):
                    base = base.value
                    continue
                if isinstance(base, ast.Subscript):
                    length += 1
                    base = base.value
                    continue
                break
            lengths.append(length)
        return lengths


# =========================
# METRICS API
# =========================
def analyze_cell(
    source: str, pandas_funcs: Set[str], mpl_funcs: Set[str], seaborn_funcs: Set[str]
) -> Tuple[Dict[str, int], Counter]:
    metrics: Dict[str, int] = {
        "LOC": 0,
        "BLC": 0,
        "S": 0,
        "P": 0,
        "UDF": 0,
        "NBD": 0,
        "OPRND": 0,
        "OPRAT": 0,
        "UOPRND": 0,
        "UOPRAT": 0,
        "ID": 0,
        "I": 0,
        "PO": 0,
        "LCO": 0,
        "CO": 0,
        "TCO": 0,
        "LCI": 0,
        "CI": 0,
        "TCI": 0,
        "V": 0,
        "MSC": 0,
        "TC": 0,
        "parse_error": "",
    }

    lines = source.splitlines()
    metrics["LOC"] = len(lines)
    metrics["BLC"] = sum(1 for ln in lines if ln.strip() == "")
    metrics["MSC"] = sum(1 for ln in lines if is_magic_or_shell(ln))

    cleaned = strip_magics_shells(source)
    tree, err = _safe_parse(cleaned)
    if tree is None:
        metrics["parse_error"] = err or "SyntaxError"
        return metrics, Counter()

    # Collector
    collector = NodeMetricsCollector(pandas_funcs, mpl_funcs, seaborn_funcs)

    # Visit the tree ONCE to collect all node-level metrics.
    # Depth is computed in a separate pass below to avoid double-counting.
    collector.visit(tree)

    # Compute nested block depth (separate pass; do not visit nodes again)
    def walk_stmt_list(stmts: List[ast.stmt], depth: int) -> None:
        for s in stmts:
            walk_node(s, depth)

    def walk_node(node: ast.AST, depth: int) -> None:
        adds_depth = False
        is_match = hasattr(ast, "Match") and isinstance(node, ast.Match)
        if (
            isinstance(
                node,
                (
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            )
            or is_match
        ):
            adds_depth = True
        new_depth = depth + (1 if adds_depth else 0)
        if new_depth > collector.max_block_depth:
            collector.max_block_depth = new_depth

        if isinstance(node, ast.If):
            walk_stmt_list(node.body, new_depth)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                walk_node(node.orelse[0], depth)
            else:
                walk_stmt_list(node.orelse, new_depth)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            walk_stmt_list(node.body, new_depth)
            walk_stmt_list(node.orelse, new_depth)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            walk_stmt_list(node.body, new_depth)
        elif isinstance(node, ast.Try):
            walk_stmt_list(node.body, new_depth)
            for h in node.handlers:
                walk_stmt_list(h.body, new_depth)
            walk_stmt_list(node.orelse, new_depth)
            walk_stmt_list(node.finalbody, new_depth)
        elif is_match:
            for case in node.cases:  # type: ignore[attr-defined]
                walk_stmt_list(case.body, new_depth)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            walk_stmt_list(node.body, new_depth)
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    walk_node(child, depth)

    if isinstance(tree, ast.Module):
        walk_stmt_list(tree.body, depth=0)
    else:
        walk_node(tree, depth=0)

    # Operands and operators
    OPRND, UOPRND = operands_with_containers(tree)
    metrics["OPRND"] = OPRND
    metrics["UOPRND"] = UOPRND
    metrics["OPRAT"] = len(collector.operator_kinds)
    metrics["UOPRAT"] = len(set(collector.operator_kinds))

    # Core counts
    metrics["S"] = collector.statements
    metrics["I"] = collector.imports
    metrics["UDF"] = collector.user_defined_functions
    metrics["P"] = collector.total_params
    metrics["ID"] = len(collector.identifiers)
    metrics["NBD"] = collector.max_block_depth

    # Pandas metrics
    po_counter: Counter = collector.pandas_po_counter
    metrics["PO"] = int(sum(po_counter.values()))
    pandas_chain_lengths = collector.pandas_chain_lengths
    metrics["LCO"] = max(pandas_chain_lengths) if pandas_chain_lengths else 0
    metrics["CO"] = len(pandas_chain_lengths)
    metrics["TCO"] = sum(pandas_chain_lengths)

    # Index metrics
    index_chain_lengths = collector.enumerate_index_chain_lengths()
    metrics["LCI"] = max(index_chain_lengths) if index_chain_lengths else 0
    metrics["CI"] = len(index_chain_lengths)
    metrics["TCI"] = sum(index_chain_lengths)

    # Visual and training
    metrics["V"] = collector.visual_calls
    metrics["TC"] = collector.train_calls

    return metrics, po_counter


def ensure_header(path: str, header: List[str], mode: str = "a"):
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0 or mode == "w"
    f = open(path, mode, encoding="utf-8", newline="")
    writer = csv.writer(f)
    if need_header:
        writer.writerow(header)
        f.flush()
    return f, writer


def run_streaming(
    input_cells_csv: str,
    pandas_ops_csv: str,
    out_metrics_csv: str,
    out_summary_csv: str,
    flush_secs: int = 10,
    stats_secs: int = 10,
    log_path: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    setup_logger(log_path=log_path, level=log_level)
    logger.info("Starting streaming analysis (recursive)")
    logger.info(f"cells={input_cells_csv} pandas_ops={pandas_ops_csv}")
    logger.info(f"flush_secs={flush_secs:_} stats_secs={stats_secs:_}")

    # Load pandas ops (small)
    pandas_funcs, func_to_cat = load_functions_with_categories(pandas_ops_csv)
    logger.info(f"Loaded pandas ops: {len(pandas_funcs):_} functions")

    # Load visualization function names
    mpl_funcs = load_functions(MPL_FUNCS_PATH)
    seaborn_funcs = load_functions(SEABORN_FUNCS_PATH)

    header = [
        "notebook_hash",
        "cell_id",
        "LOC",
        "BLC",
        "S",
        "P",
        "UDF",
        "NBD",
        "OPRND",
        "OPRAT",
        "UOPRND",
        "UOPRAT",
        "ID",
        "I",
        "PO",
        "LCO",
        "CO",
        "TCO",
        "LCI",
        "CI",
        "TCI",
        "V",
        "MSC",
        "TC",
    ]

    out_f, out_w = ensure_header(out_metrics_csv, header, mode="w")

    global_po: Counter = Counter()
    parse_errors = 0
    total_rows = 0
    start = time.time()
    last_flush = start
    last_stats = start

    try:
        with open(input_cells_csv, "r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            for req in ("notebook_hash", "cell_id", "source"):
                if req not in reader.fieldnames:
                    raise ValueError(f"code_cells.csv missing column: {req}")

            for row in reader:
                total_rows += 1
                src = row["source"]
                metrics, po_counter = analyze_cell(
                    src, pandas_funcs, mpl_funcs, seaborn_funcs
                )

                # Aggregate global pandas op counts from per-cell result
                for fn, cnt in po_counter.items():
                    global_po[fn] += cnt

                if metrics.get("parse_error"):
                    parse_errors += 1

                out_row = [
                    row["notebook_hash"],
                    row["cell_id"],
                    metrics["LOC"],
                    metrics["BLC"],
                    metrics["S"],
                    metrics["P"],
                    metrics["UDF"],
                    metrics["NBD"],
                    metrics["OPRND"],
                    metrics["OPRAT"],
                    metrics["UOPRND"],
                    metrics["UOPRAT"],
                    metrics["ID"],
                    metrics["I"],
                    metrics["PO"],
                    metrics["LCO"],
                    metrics["CO"],
                    metrics["TCO"],
                    metrics["LCI"],
                    metrics["CI"],
                    metrics["TCI"],
                    metrics["V"],
                    metrics["MSC"],
                    metrics["TC"],
                ]
                out_w.writerow(out_row)

                now = time.time()
                if now - last_flush >= flush_secs:
                    try:
                        out_f.flush()
                    except Exception as e:
                        logger.warning(f"Flush failed: {e}")
                    last_flush = now

                if now - last_stats >= stats_secs:
                    elapsed = now - start
                    rps = total_rows / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        f"total_rows={total_rows:_} rps={rps:_.1f} parse_errors={parse_errors:_}"
                    )
                    last_stats = now
    except KeyboardInterrupt:
        logger.warning("Interrupted by Ctrl+C. Finalizing partial outputs...")
    finally:
        try:
            out_f.flush()
        except Exception:
            pass
        try:
            out_f.close()
        except Exception:
            pass

    # Write pandas ops summary
    with open(out_summary_csv, "w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["category", "function", "count"])
        all_functions = sorted(set(func_to_cat.keys()) | set(global_po.keys()))
        for fn in all_functions:
            w.writerow([func_to_cat.get(fn, ""), fn, int(global_po.get(fn, 0))])

    end = time.time()
    elapsed = end - start
    rows_per_second = (total_rows / elapsed) if elapsed > 0 else 0.0

    logger.info(
        f"Done. rows={total_rows:_} elapsed={elapsed:_.1f}s rps={rows_per_second:_.1f} parse_errors={parse_errors:_}"
    )
    logger.info(f"Pandas ops summary -> {out_summary_csv}")
    logger.info(f"Metrics CSV -> {out_metrics_csv}")


if __name__ == "__main__":
    run_streaming(
        input_cells_csv=CELLS_PATH,
        pandas_ops_csv=PANDAS_OPS_PATH,
        out_metrics_csv=OUT_METRICS_PATH,
        out_summary_csv=OUT_SUMMARY_PATH,
        flush_secs=FLUSH_SECS,
        stats_secs=STATS_SECS,
        log_path=LOG_PATH,
        log_level=LOG_LEVEL,
    )


