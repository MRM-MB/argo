"""Best-effort parse-only static context for validation.

This module deliberately provides syntactic evidence, not semantic proof. It never builds or
executes the target. Tree-sitter is used to recover an enclosing symbol, local call expressions,
possible name-matched callers, and a small local definition/use slice around a cited line.

The output must not be interpreted as proving reachability, control flow, data flow,
exploitability, or vulnerability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .ranking import split_ref

_SUPPORTED = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".cs": "c_sharp",
}

_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "vendor", "dist", "build",
    "__pycache__", ".pytest_cache",
}

_FUNCTION_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "function_expression", "arrow_function", "method_definition",
                   "generator_function_declaration", "generator_function"},
    "typescript": {"function_declaration", "function_expression", "arrow_function", "method_definition",
                   "generator_function_declaration", "generator_function", "method_signature"},
    "tsx": {"function_declaration", "function_expression", "arrow_function", "method_definition",
            "generator_function_declaration", "generator_function", "method_signature"},
    "c_sharp": {"method_declaration", "constructor_declaration", "local_function_statement",
                "lambda_expression", "anonymous_method_expression", "accessor_declaration"},
}

_CALL_TYPES = {
    "python": {"call"},
    "javascript": {"call_expression", "new_expression"},
    "typescript": {"call_expression", "new_expression"},
    "tsx": {"call_expression", "new_expression"},
    "c_sharp": {"invocation_expression", "object_creation_expression"},
}

_IDENTIFIER_TYPES = {"identifier", "type_identifier", "property_identifier", "field_identifier"}


@dataclass(frozen=True)
class StaticSymbol:
    name: str
    kind: str
    file: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CallReference:
    name: str
    file: str
    line: int
    caller: str | None = None


@dataclass(frozen=True)
class DefUseStep:
    name: str
    defined_at_line: int
    used_at_line: int
    depends_on: tuple[str, ...] = ()


@dataclass
class StaticContext:
    location: str
    language: str | None = None
    symbol: StaticSymbol | None = None
    calls: list[CallReference] = field(default_factory=list)
    possible_incoming: list[CallReference] = field(default_factory=list)
    local_def_use: list[DefUseStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _point_row(point: Any) -> int:
    row = getattr(point, "row", None)
    return int(row if row is not None else point[0])


def _line_span(node: Any) -> tuple[int, int]:
    return _point_row(node.start_point) + 1, _point_row(node.end_point) + 1


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _named_children(node: Any) -> Iterable[Any]:
    return getattr(node, "named_children", ())


@lru_cache(maxsize=8)
def _parser(language: str) -> Any:
    """Create a local tree-sitter parser. No target build and no target execution."""
    from tree_sitter import Language, Parser

    if language == "python":
        import tree_sitter_python as grammar
        lang = Language(grammar.language())
    elif language == "javascript":
        import tree_sitter_javascript as grammar
        lang = Language(grammar.language())
    elif language == "typescript":
        import tree_sitter_typescript as grammar
        lang = Language(grammar.language_typescript())
    elif language == "tsx":
        import tree_sitter_typescript as grammar
        lang = Language(grammar.language_tsx())
    elif language == "c_sharp":
        import tree_sitter_c_sharp as grammar
        lang = Language(grammar.language())
    else:
        raise ValueError(f"unsupported language: {language}")
    return Parser(lang)


def _language_for(path: Path) -> str | None:
    return _SUPPORTED.get(path.suffix.lower())


def _walk(node: Any) -> Iterable[Any]:
    yield node
    for child in _named_children(node):
        yield from _walk(child)


def _contains_line(node: Any, line: int) -> bool:
    start, end = _line_span(node)
    return start <= line <= end


def _smallest_node_at_line(node: Any, line: int) -> Any:
    current = node
    while True:
        matches = [c for c in _named_children(current) if _contains_line(c, line)]
        if not matches:
            return current
        current = min(matches, key=lambda n: n.end_byte - n.start_byte)


def _symbol_name(node: Any, source: bytes) -> str:
    name = node.child_by_field_name("name")
    if name is not None:
        return _node_text(name, source).strip()
    parent = getattr(node, "parent", None)
    if parent is not None:
        for field_name in ("name", "left"):
            cand = parent.child_by_field_name(field_name)
            if cand is not None:
                text = _node_text(cand, source).strip()
                if text:
                    return text
    if node.type == "accessor_declaration":
        return _node_text(node, source).split("{", 1)[0].strip() or "<accessor>"
    return "<anonymous>"


def _symbol_kind(node: Any) -> str:
    if node.type in {"method_definition", "method_declaration", "constructor_declaration",
                     "accessor_declaration", "method_signature"}:
        return "method"
    if node.type in {"lambda_expression", "anonymous_method_expression", "arrow_function"}:
        return "lambda"
    return "function"


def _enclosing_symbol(root: Any, source: bytes, language: str, file: str,
                      line: int) -> StaticSymbol | None:
    candidates = [
        n for n in _walk(root)
        if n.type in _FUNCTION_TYPES[language] and _contains_line(n, line)
    ]
    if not candidates:
        return None
    node = min(candidates, key=lambda n: n.end_byte - n.start_byte)
    start, end = _line_span(node)
    return StaticSymbol(
        name=_symbol_name(node, source), kind=_symbol_kind(node), file=file,
        start_line=start, end_line=end,
    )


def _find_symbol_node(root: Any, language: str, line: int) -> Any | None:
    nodes = [n for n in _walk(root)
             if n.type in _FUNCTION_TYPES[language] and _contains_line(n, line)]
    return min(nodes, key=lambda n: n.end_byte - n.start_byte) if nodes else None


def _callee_node(node: Any) -> Any | None:
    for field_name in ("function", "expression", "type"):
        child = node.child_by_field_name(field_name)
        if child is not None:
            return child
    children = list(_named_children(node))
    return children[0] if children else None


def _terminal_name(text: str) -> str:
    text = text.strip()
    for sep in ("::", "->", "."):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text.strip()


def _calls_within(node: Any, source: bytes, language: str, file: str) -> list[CallReference]:
    calls: list[CallReference] = []
    seen: set[tuple[str, int]] = set()

    def visit(cur: Any, top: bool = False) -> None:
        if not top and cur.type in _FUNCTION_TYPES[language]:
            return
        if cur.type in _CALL_TYPES[language]:
            callee = _callee_node(cur)
            if callee is not None:
                name = _node_text(callee, source).strip()
                line = _point_row(cur.start_point) + 1
                key = (name, line)
                if name and key not in seen:
                    seen.add(key)
                    calls.append(CallReference(name=name, file=file, line=line))
        for child in _named_children(cur):
            visit(child)

    visit(node, top=True)
    return calls


def _identifiers(node: Any, source: bytes) -> set[str]:
    out: set[str] = set()
    for cur in _walk(node):
        if cur.type in _IDENTIFIER_TYPES:
            text = _node_text(cur, source).strip()
            if text and not text[0].isdigit():
                out.add(text)
    return out


def _definition_parts(node: Any) -> tuple[Any | None, Any | None]:
    if node.type in {"assignment", "assignment_expression", "augmented_assignment"}:
        return node.child_by_field_name("left"), node.child_by_field_name("right")
    if node.type == "variable_declarator":
        return node.child_by_field_name("name"), node.child_by_field_name("value")
    return None, None


def _local_def_use(symbol_node: Any, source: bytes, line: int,
                   *, max_steps: int = 12) -> list[DefUseStep]:
    defs: dict[str, list[tuple[int, tuple[str, ...]]]] = {}
    for node in _walk(symbol_node):
        lhs, rhs = _definition_parts(node)
        if lhs is None:
            continue
        def_line = _point_row(node.start_point) + 1
        if def_line > line:
            continue
        deps = tuple(sorted(_identifiers(rhs, source))) if rhs is not None else ()
        for name in sorted(_identifiers(lhs, source)):
            defs.setdefault(name, []).append((def_line, deps))

    # Seed the backwards slice from every identifier on the cited source line.  Using the
    # single smallest AST node is too narrow: for `sink(clean)` it can select the callee
    # identifier `sink` and miss the argument `clean`, which is the value whose prior
    # definition we actually want to trace.
    line_identifiers: set[str] = set()
    for node in _walk(symbol_node):
        if node.type not in _IDENTIFIER_TYPES:
            continue
        start, end = _line_span(node)
        if start <= line <= end:
            text = _node_text(node, source).strip()
            if text and not text[0].isdigit():
                line_identifiers.add(text)

    if not line_identifiers:
        target = _smallest_node_at_line(symbol_node, line)
        line_identifiers = _identifiers(target, source)

    queue: list[tuple[str, int]] = [(name, line) for name in sorted(line_identifiers)]
    seen: set[tuple[str, int]] = set()
    steps: list[DefUseStep] = []
    while queue and len(steps) < max_steps:
        name, used_at = queue.pop(0)
        candidates = [d for d in defs.get(name, ()) if d[0] <= used_at]
        if not candidates:
            continue
        defined_at, deps = max(candidates, key=lambda item: item[0])
        key = (name, defined_at)
        if key in seen:
            continue
        seen.add(key)
        steps.append(DefUseStep(name=name, defined_at_line=defined_at,
                                used_at_line=used_at, depends_on=deps))
        queue.extend((dep, defined_at) for dep in deps)
    return steps


def _parse_file(path: Path) -> tuple[str, bytes, Any]:
    language = _language_for(path)
    if language is None:
        raise ValueError("unsupported source extension")
    source = path.read_bytes()
    tree = _parser(language).parse(source)
    return language, source, tree.root_node


def _iter_source_files(root: Path, *, max_files: int = 1000) -> Iterable[Path]:
    count = 0
    for path in root.rglob("*"):
        if count >= max_files:
            return
        if not path.is_file() or _language_for(path) is None:
            continue
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        count += 1
        yield path


@lru_cache(maxsize=4)
def _repo_call_index(repo_root: str) -> dict[str, tuple[CallReference, ...]]:
    """Build a bounded name-only call index once per read-only repo copy."""
    repo_dir = Path(repo_root)
    index: dict[str, list[CallReference]] = {}
    for path in _iter_source_files(repo_dir):
        try:
            language, source, root = _parse_file(path)
        except (OSError, ValueError, ImportError):
            continue
        rel = path.relative_to(repo_dir).as_posix()
        for node in _walk(root):
            if node.type not in _CALL_TYPES[language]:
                continue
            callee = _callee_node(node)
            if callee is None:
                continue
            call_name = _node_text(callee, source).strip()
            terminal = _terminal_name(call_name)
            if not terminal:
                continue
            line = _point_row(node.start_point) + 1
            caller = _enclosing_symbol(root, source, language, rel, line)
            index.setdefault(terminal, []).append(
                CallReference(name=call_name, file=rel, line=line,
                              caller=caller.name if caller else None)
            )
    return {name: tuple(refs) for name, refs in index.items()}


def _possible_incoming(repo_dir: Path, target_name: str,
                       *, max_refs: int = 8) -> list[CallReference]:
    terminal = _terminal_name(target_name)
    if not terminal or terminal.startswith("<"):
        return []
    try:
        refs = _repo_call_index(str(repo_dir.resolve())).get(terminal, ())
    except Exception:  # noqa: BLE001 - optional enrichment must remain best-effort
        return []
    return list(refs[:max_refs])


def analyze_location(repo_dir: Path, ref: str, *, max_incoming: int = 8) -> StaticContext:
    """Return bounded parse-only context for one file:line citation."""
    file, raw_line = split_ref(ref)
    ctx = StaticContext(location=ref)
    root = repo_dir.resolve()
    path = (repo_dir / file).resolve()
    if not path.is_relative_to(root):
        ctx.notes.append("citation resolves outside the repository; static context withheld")
        return ctx
    language = _language_for(path)
    ctx.language = language
    if language is None:
        ctx.notes.append("unsupported source extension")
        return ctx
    if not raw_line:
        ctx.notes.append("citation has no line number")
        return ctx
    try:
        line = int(raw_line)
    except ValueError:
        ctx.notes.append("citation line is not an integer")
        return ctx
    try:
        language, source, tree_root = _parse_file(path)
    except ImportError:
        ctx.notes.append("tree-sitter parser dependencies are unavailable")
        return ctx
    except (OSError, ValueError) as exc:
        ctx.notes.append(f"static parse unavailable: {exc.__class__.__name__}")
        return ctx

    rel = path.relative_to(root).as_posix()
    symbol = _enclosing_symbol(tree_root, source, language, rel, line)
    ctx.symbol = symbol
    symbol_node = _find_symbol_node(tree_root, language, line)
    if symbol_node is None:
        ctx.notes.append("no enclosing function/method found for cited line")
        return ctx

    ctx.calls = _calls_within(symbol_node, source, language, rel)
    ctx.local_def_use = _local_def_use(symbol_node, source, line)
    if symbol is not None:
        ctx.possible_incoming = _possible_incoming(root, symbol.name, max_refs=max_incoming)
    return ctx


def build_static_context(repo_dir: Path, affected: list[str], *, max_locations: int = 3) -> str:
    """Format deterministic static context for validation prompts.

    Unsupported or malformed locations never break validation and never alter a verdict by
    themselves; they simply produce an unavailable-context note.
    """
    contexts: list[dict[str, Any]] = []
    for ref in affected[:max_locations]:
        try:
            contexts.append(analyze_location(repo_dir, ref).to_dict())
        except Exception as exc:  # noqa: BLE001 - enrichment must never break validation
            contexts.append({
                "location": ref,
                "notes": [f"static analysis unavailable: {exc.__class__.__name__}"],
            })
    if not contexts:
        return "(no deterministic static context available)"

    import json
    return (
        "SYNTACTIC EVIDENCE ONLY - this does NOT prove runtime reachability, data flow, control "
        "flow, exploitability, or vulnerability. Name-matched incoming references are possible "
        "callers, not a resolved call graph. Source code remains authoritative.\n"
        + json.dumps(contexts, indent=2)
    )
