"""Fallback YAML parser supporting safe_load and dump when PyYAML is not installed."""

from __future__ import annotations

import io
import re
from typing import Any


def _parse_scalar(val: str) -> Any:
    val = val.strip()
    if not val:
        return ""
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in ("null", "none", "~"):
        return None
    # Inline list [a, b, c]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        items = [x.strip() for x in inner.split(",")]
        return [_parse_scalar(x) for x in items]
    # Inline dict {0: a, 1: b}
    if val.startswith("{") and val.endswith("}"):
        inner = val[1:-1].strip()
        if not inner:
            return {}
        items = [x.strip() for x in inner.split(",")]
        res = {}
        for item in items:
            if ":" in item:
                k, v = item.split(":", 1)
                res[_parse_scalar(k)] = _parse_scalar(v)
        return res
    # Numbers
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def _tokenize_lines(text: str) -> list[tuple[int, str]]:
    lines = []
    for raw in text.splitlines():
        # Strip comments unless in quotes
        clean = ""
        in_quote = False
        quote_char = ""
        for i, char in enumerate(raw):
            if char in ('"', "'"):
                if not in_quote:
                    in_quote = True
                    quote_char = char
                elif quote_char == char:
                    in_quote = False
            elif char == "#" and not in_quote:
                break
            clean += char
        stripped = clean.rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    return lines


def _parse_block(lines: list[tuple[int, str]], start: int = 0, current_indent: int = 0) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start

    first_indent, first_line = lines[start]
    # Check if this block is a list
    if first_line.startswith("- ") or first_line == "-":
        result_list = []
        i = start
        while i < len(lines):
            indent, line = lines[i]
            if indent < current_indent:
                break
            if indent == current_indent and (line.startswith("- ") or line == "-"):
                item_content = line[2:].strip()
                if item_content:
                    if ":" in item_content and not (item_content.startswith("[") or item_content.startswith("{")):
                        # It's a dict item in a list e.g. - name: foo
                        k, v = item_content.split(":", 1)
                        sub_dict = {_parse_scalar(k): _parse_scalar(v) if v.strip() else {}}
                        i += 1
                        # check if next lines continue this dict
                        while i < len(lines) and lines[i][0] > indent:
                            sub_val, i = _parse_block(lines, i, lines[i][0])
                            if isinstance(sub_val, dict):
                                sub_dict.update(sub_val)
                        result_list.append(sub_dict)
                        continue
                    else:
                        result_list.append(_parse_scalar(item_content))
                        i += 1
                else:
                    # Next lines are the list item content
                    sub_val, i = _parse_block(lines, i + 1, indent + 2)
                    result_list.append(sub_val)
            elif indent > current_indent:
                # continuation
                sub_val, i = _parse_block(lines, i, indent)
                if result_list:
                    result_list[-1] = sub_val
            else:
                break
        return result_list, i

    # Otherwise it's a mapping / dict
    result_dict: dict[Any, Any] = {}
    i = start
    while i < len(lines):
        indent, line = lines[i]
        if indent < current_indent:
            break
        if ":" in line:
            parts = line.split(":", 1)
            raw_key = parts[0].strip()
            raw_val = parts[1].strip()
            key = _parse_scalar(raw_key)
            if raw_val:
                val = _parse_scalar(raw_val)
                result_dict[key] = val
                i += 1
            else:
                # Value is in subsequent indented lines
                if i + 1 < len(lines) and lines[i + 1][0] > indent:
                    sub_val, next_i = _parse_block(lines, i + 1, lines[i + 1][0])
                    result_dict[key] = sub_val
                    i = next_i
                else:
                    result_dict[key] = None
                    i += 1
        else:
            i += 1
    return result_dict, i


def safe_load(stream: Any) -> Any:
    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = str(stream)
    lines = _tokenize_lines(text)
    if not lines:
        return {}
    res, _ = _parse_block(lines, 0, lines[0][0])
    return res


def load(stream: Any, Loader: Any = None) -> Any:
    return safe_load(stream)


def dump(data: Any, stream: Any = None, **kwargs: Any) -> str | None:
    import json
    # Simple JSON-compatible YAML dump fallback
    output = json.dumps(data, indent=2)
    if stream is not None:
        stream.write(output)
        return None
    return output
