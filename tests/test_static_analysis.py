from __future__ import annotations

import json

import pytest

from argo.static_analysis import analyze_location, build_static_context


@pytest.mark.parametrize(
    ("name", "source", "line", "language", "symbol", "expected_call"),
    [
        (
            "sample.py",
            "def normalize(v):\n    return v\n\ndef handler(request):\n"
            "    data = request.value\n    clean = normalize(data)\n    sink(clean)\n",
            7,
            "python",
            "handler",
            "sink",
        ),
        (
            "sample.js",
            "function handler(input) {\n  const clean = normalize(input);\n  sink(clean);\n}\n",
            3,
            "javascript",
            "handler",
            "sink",
        ),
        (
            "sample.ts",
            "function handler(input: string) {\n  const clean = normalize(input);\n  sink(clean);\n}\n",
            3,
            "typescript",
            "handler",
            "sink",
        ),
        (
            "sample.cs",
            "class Example {\n  void Handler(string input) {\n"
            "    var clean = Normalize(input);\n    Sink(clean);\n  }\n}\n",
            4,
            "c_sharp",
            "Handler",
            "Sink",
        ),
    ],
)
def test_analyze_location_supported_languages(
    tmp_path, name, source, line, language, symbol, expected_call
):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")

    ctx = analyze_location(tmp_path, f"{name}:{line}")

    assert ctx.language == language
    assert ctx.symbol is not None
    assert ctx.symbol.name == symbol
    assert any(expected_call in call.name for call in ctx.calls)


def test_local_def_use_and_possible_incoming_are_syntactic(tmp_path):
    (tmp_path / "target.py").write_text(
        "def normalize(v):\n    return v\n\n"
        "def handler(request):\n"
        "    raw = request.value\n"
        "    clean = normalize(raw)\n"
        "    sink(clean)\n",
        encoding="utf-8",
    )
    (tmp_path / "routes.py").write_text(
        "from target import handler\n\n"
        "def route(req):\n"
        "    return handler(req)\n",
        encoding="utf-8",
    )

    ctx = analyze_location(tmp_path, "target.py:7")

    assert ctx.symbol is not None and ctx.symbol.name == "handler"
    assert any(step.name == "clean" and step.defined_at_line == 6 for step in ctx.local_def_use)
    assert any(ref.file == "routes.py" and ref.caller == "route" for ref in ctx.possible_incoming)


def test_path_escape_is_rejected(tmp_path):
    ctx = analyze_location(tmp_path, "../outside.py:1")

    assert ctx.symbol is None
    assert any("outside the repository" in note for note in ctx.notes)


def test_static_context_is_explicitly_non_semantic(tmp_path):
    (tmp_path / "sample.py").write_text(
        "def f(x):\n    y = normalize(x)\n    sink(y)\n",
        encoding="utf-8",
    )

    rendered = build_static_context(tmp_path, ["sample.py:3"])

    first, payload = rendered.split("\n", 1)
    assert "does NOT prove runtime reachability" in first
    data = json.loads(payload)
    assert data[0]["symbol"]["name"] == "f"
