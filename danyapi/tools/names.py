from __future__ import annotations

import json
import re
from difflib import get_close_matches
from functools import lru_cache
from typing import Any

from .common import ToolCall, _tool_function
from .jsonfix import _coerce_scalar, _loads_lenient

_FUZZY_NAME_CUTOFF = 0.8
_FUZZY_PARAM_CUTOFF = 0.85


def _length_reachable(needle: str, candidate: str, cutoff: float) -> bool:
    short = len(needle)
    long = len(candidate)
    if short > long:
        short, long = long, short
    if not short:
        return long == 0
    return (2.0 * short) / (short + long) >= cutoff


@lru_cache(maxsize=8192)
def _casefold(text: str) -> str:
    return text.casefold()


@lru_cache(maxsize=8192)
def _name_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", _casefold(name))


@lru_cache(maxsize=256)
def _folded_keys(keys: tuple[Any, ...]) -> dict[str, str]:
    folded: dict[str, str] = {}
    for key in keys:
        if isinstance(key, str):
            folded.setdefault(_casefold(key), key)
    return folded


@lru_cache(maxsize=256)
def _folded_names(keys: tuple[Any, ...]) -> dict[str, str]:
    return {_casefold(key): key for key in keys}


@lru_cache(maxsize=256)
def _compact_names(keys: tuple[Any, ...]) -> dict[str, str]:
    return {_name_key(key): key for key in keys}


def _schema_for_name(tool_schemas: dict[str, dict[str, Any]] | None, name: str) -> dict[str, Any] | None:
    if not tool_schemas or not name:
        return None
    if not isinstance(tool_schemas, dict):
        return None
    if name in tool_schemas:
        spec = tool_schemas[name]
        return spec if isinstance(spec, dict) else None
    keys, seed = _schema_scope(tool_schemas)
    key = _folded_keys(keys).get(_casefold(name))
    if key is not None:
        spec = tool_schemas[key]
        return spec if isinstance(spec, dict) else None
    resolved = _resolve_alias(name, seed)
    if resolved is not None and resolved in tool_schemas:
        spec = tool_schemas[resolved]
        return spec if isinstance(spec, dict) else None
    return None


@lru_cache(maxsize=256)
def _alias_rows(seed: tuple[tuple[str, tuple[Any, ...]], ...]) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for known, aliases in seed:
        for alias in aliases:
            if not isinstance(alias, str) or not alias:
                continue
            rows.append((_name_key(alias), _casefold(alias), known))
    return tuple(rows)


_scope_cache: list[tuple[dict[Any, Any], tuple[Any, ...], tuple[tuple[str, tuple[Any, ...]], ...]]] = []


def _schema_scope(tool_schemas: dict[str, dict[str, Any]]) -> tuple[tuple[Any, ...], tuple[tuple[str, tuple[Any, ...]], ...]]:
    entry = None
    if _scope_cache:
        entry = _scope_cache[0]
    if entry is not None and entry[0] is tool_schemas:
        return entry[1], entry[2]
    keys = tuple(tool_schemas)
    seed = tuple((known, tuple((spec if isinstance(spec, dict) else {}).get("_aliases") or ())) for known, spec in tool_schemas.items())
    if entry is None:
        _scope_cache.append((tool_schemas, keys, seed))
    else:
        _scope_cache[0] = (tool_schemas, keys, seed)
    return keys, seed


@lru_cache(maxsize=256)
def _schema_name_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_name_key(key) for key in keys)


def _resolve_alias(name: str, seed: tuple[tuple[str, tuple[Any, ...]], ...] | dict[str, dict[str, Any]] | None) -> str | None:
    if seed is None:
        return None
    if isinstance(seed, dict):
        if not seed:
            return None
        seed = _schema_scope(seed)[1]
    folded = _casefold(name)
    key = _name_key(name)
    if not key:
        return None
    for alias_key, alias_folded, known in _alias_rows(seed):
        if alias_key == key or alias_folded == folded:
            return known
    return None


@lru_cache(maxsize=2048)
def _fuzzy_tool_name(key: str, keys: tuple[str, ...]) -> str | None:
    if len(key) < 4:
        return None
    compacted = _schema_name_keys(keys)
    candidates = [candidate for candidate in compacted if _length_reachable(key, candidate, _FUZZY_NAME_CUTOFF)]
    if not candidates:
        return None
    matches = get_close_matches(key, candidates, n=2, cutoff=_FUZZY_NAME_CUTOFF)
    if len(matches) != 1:
        return None
    hit = matches[0]
    return next(known for known in keys if _name_key(known) == hit)


def _fuzzy_known_name(name: str, tool_schemas: dict[str, dict[str, Any]]) -> str | None:
    key = _name_key(name)
    keys, _seed = _schema_scope(tool_schemas)
    if not keys:
        return None
    return _fuzzy_tool_name(key, keys)


def _normalize_call_name(name: str, tool_schemas: dict[str, dict[str, Any]] | None) -> str:
    if not tool_schemas or name in tool_schemas:
        return name
    keys, seed = _schema_scope(tool_schemas)
    hit = _folded_names(keys).get(_casefold(name))
    if hit is not None:
        return hit
    hit = _compact_names(keys).get(_name_key(name))
    if hit is not None:
        return hit
    return _resolve_alias(name, seed) or _fuzzy_known_name(name, tool_schemas) or name


_tool_schema_map_cache: dict[int, tuple[tuple[int, ...], dict[str, dict[str, Any]]]] = {}
_TOOL_SCHEMA_MAP_CACHE_MAX = 256


def _resolved_prop_type(spec: Any) -> Any:
    if not isinstance(spec, dict):
        return None
    typ = spec.get("type")
    if isinstance(typ, list):
        for candidate in ("integer", "number", "boolean", "null", "string"):
            if candidate in typ:
                return candidate
    return typ


def _schema_params(fn: dict) -> Any:
    params = fn.get("parameters")
    if isinstance(params, str):
        try:
            return json.loads(params)
        except (ValueError, TypeError, AttributeError):
            return None
    return params


def tool_schema_map(tools: list[Any] | None) -> dict[str, dict[str, Any]]:
    if not tools or not isinstance(tools, list):
        return {}
    key = id(tools)
    fingerprint = tuple(id(item) for item in tools)
    cached = _tool_schema_map_cache.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = _tool_function(tool)
        if not fn or not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        params = _schema_params(fn)
        prop_types: dict[str, Any] = {}
        properties = params.get("properties") if isinstance(params, dict) else None
        if isinstance(properties, dict):
            for prop, spec in properties.items():
                if not isinstance(prop, str) or not isinstance(spec, dict):
                    continue
                typ = _resolved_prop_type(spec)
                if typ:
                    prop_types[prop] = typ
        aliases = fn.get("aliases")
        if isinstance(aliases, (list, tuple)):
            cleaned = [a for a in aliases if isinstance(a, str) and a.strip()]
            if cleaned:
                prop_types["_aliases"] = cleaned
        result[name] = prop_types
    while len(_tool_schema_map_cache) >= _TOOL_SCHEMA_MAP_CACHE_MAX:
        _tool_schema_map_cache.pop(next(iter(_tool_schema_map_cache)))
    _tool_schema_map_cache[key] = (fingerprint, result)
    return result


_FIX_MODES = frozenset({"report", "safe", "full"})
_FIX_FUZZY_PARAM_MIN = 4
_FIX_FUZZY_PARAM_CUTOFF = 0.85
_FIX_FUZZY_ENUM_MIN = 2
_FIX_FUZZY_ENUM_CUTOFF = 0.85


def tool_schema_detail(tools: list[Any] | None) -> dict[str, dict[str, Any]]:
    if not tools or not isinstance(tools, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = _tool_function(tool)
        if not fn or not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        params = _schema_params(fn)
        types: dict[str, Any] = {}
        enums: dict[str, list[Any]] = {}
        defaults: dict[str, Any] = {}
        bounds: dict[str, dict[str, Any]] = {}
        param_aliases: dict[str, list[str]] = {}
        required: list[str] = []
        if isinstance(params, dict):
            req = params.get("required")
            if isinstance(req, list):
                required = [str(item) for item in req if isinstance(item, str) and item]
            properties = params.get("properties")
            if isinstance(properties, dict):
                for prop, spec in properties.items():
                    if not isinstance(prop, str) or not isinstance(spec, dict):
                        continue
                    typ = _resolved_prop_type(spec)
                    if typ:
                        types[prop] = typ
                    enum_values = spec.get("enum")
                    if isinstance(enum_values, list) and enum_values:
                        enums[prop] = list(enum_values)
                    if "default" in spec:
                        defaults[prop] = spec["default"]
                    entry: dict[str, Any] = {}
                    low = spec.get("minimum")
                    high = spec.get("maximum")
                    if isinstance(low, (int, float)) and not isinstance(low, bool):
                        entry["minimum"] = low
                    if isinstance(high, (int, float)) and not isinstance(high, bool):
                        entry["maximum"] = high
                    if entry:
                        bounds[prop] = entry
                    aliases = spec.get("aliases")
                    if isinstance(aliases, (list, tuple)):
                        cleaned = [a for a in aliases if isinstance(a, str) and a.strip()]
                        if cleaned:
                            param_aliases[prop] = cleaned
        name_aliases: list[str] = []
        raw_aliases = fn.get("aliases")
        if isinstance(raw_aliases, (list, tuple)):
            name_aliases = [a for a in raw_aliases if isinstance(a, str) and a.strip()]
        result[name] = {
            "types": types,
            "required": required,
            "enums": enums,
            "defaults": defaults,
            "bounds": bounds,
            "param_aliases": param_aliases,
            "name_aliases": name_aliases,
        }
    return result


@lru_cache(maxsize=2048)
def _fuzzy_arg_key(compact: str, known_props: tuple[str, ...]) -> str | None:
    compact_map = _compact_names(known_props)
    candidates = [key for key in compact_map if _length_reachable(compact, key, _FIX_FUZZY_PARAM_CUTOFF)]
    if not candidates:
        return None
    matches = get_close_matches(compact, candidates, n=2, cutoff=_FIX_FUZZY_PARAM_CUTOFF)
    if len(matches) == 1:
        return compact_map[matches[0]]
    return None


def _resolve_arg_key(
    key: str,
    known_props: tuple[str, ...],
    param_aliases: dict[str, Any],
) -> tuple[str | None, str]:
    if key in known_props:
        return key, "exact"
    if not known_props:
        return None, "unknown"
    hit = _folded_names(known_props).get(_casefold(key))
    if hit is not None:
        return hit, "casefold"
    hit = _compact_names(known_props).get(_name_key(key))
    if hit is not None:
        return hit, "compact"
    folded = _casefold(key)
    compact = _name_key(key)
    for prop, aliases in param_aliases.items():
        for alias in aliases:
            if _casefold(alias) == folded or _name_key(alias) == compact:
                return prop, "alias"
    if len(compact) >= _FIX_FUZZY_PARAM_MIN:
        hit = _fuzzy_arg_key(compact, known_props)
        if hit is not None:
            return hit, "fuzzy"
    return None, "unknown"


def _match_enum(value: Any, enum_values: list[Any]) -> tuple[Any, str] | None:
    try:
        if value in enum_values:
            return value, "exact"
    except TypeError:
        pass
    if isinstance(value, str):
        folded = _casefold(value)
        for item in enum_values:
            if isinstance(item, str) and _casefold(item) == folded:
                return item, "casefold"
        str_values = [item for item in enum_values if isinstance(item, str)]
        if str_values and len(value) >= _FIX_FUZZY_ENUM_MIN:
            matches = get_close_matches(value, str_values, n=2, cutoff=_FIX_FUZZY_ENUM_CUTOFF)
            if len(matches) == 1:
                return matches[0], "fuzzy"
    return None


def _coerce_by_type(value: Any, json_type: Any) -> tuple[Any, bool]:
    if json_type is None:
        return value, False
    if json_type == "string":
        if isinstance(value, str):
            return value, False
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False), True
        if value is None:
            return "", True
        return json.dumps(value, ensure_ascii=False), True
    if json_type in ("integer", "number"):
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            if json_type == "integer" and isinstance(value, float) and value.is_integer():
                return int(value), True
            return value, False
        if isinstance(value, str):
            coerced = _coerce_scalar(value, json_type)
            if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
                return coerced, True
        return value, False
    if json_type == "boolean":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, str):
            low = value.strip().lower()
            if low == "true":
                return True, True
            if low == "false":
                return False, True
        return value, False
    if json_type == "null":
        if value is None:
            return value, False
        if isinstance(value, str) and value.strip().lower() in ("null", "none", "~"):
            return None, True
        return value, False
    return value, False


def fix_tool_calls(
    calls: list[ToolCall],
    tool_schemas: dict[str, dict[str, Any]] | None,
    tool_details: dict[str, dict[str, Any]] | None = None,
    mode: str = "report",
    report: dict[str, Any] | None = None,
) -> list[ToolCall]:
    if not calls:
        return calls
    if mode not in _FIX_MODES:
        mode = "report"
    schemas = tool_schemas if isinstance(tool_schemas, dict) else {}
    details = tool_details if isinstance(tool_details, dict) else {}
    fixes: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    result: list[ToolCall] = []
    apply_fixes = mode in ("safe", "full")
    drop_unknown = mode == "full"
    for call in calls:
        spec = details.get(call.name)
        if not isinstance(spec, dict):
            resolved = _schema_for_name(schemas, call.name) if schemas else None
            if schemas and resolved is None:
                warnings.append({"call_id": call.id, "kind": "unknown_tool", "name": call.name})
                result.append(call)
                continue
            spec = {"types": {key: value for key, value in (resolved or {}).items() if key != "_aliases"}}
        raw_types = spec.get("types")
        types: dict[str, Any] = raw_types if isinstance(raw_types, dict) else {}
        raw_enums = spec.get("enums")
        enums: dict[str, Any] = raw_enums if isinstance(raw_enums, dict) else {}
        raw_defaults = spec.get("defaults")
        defaults: dict[str, Any] = raw_defaults if isinstance(raw_defaults, dict) else {}
        raw_bounds = spec.get("bounds")
        bounds: dict[str, Any] = raw_bounds if isinstance(raw_bounds, dict) else {}
        raw_param_aliases = spec.get("param_aliases")
        param_aliases: dict[str, Any] = raw_param_aliases if isinstance(raw_param_aliases, dict) else {}
        raw_required = spec.get("required")
        required: list[str] = [name for name in raw_required if isinstance(name, str)] if isinstance(raw_required, list) else []
        try:
            parsed = _loads_lenient(call.arguments) if call.arguments and call.arguments.strip() else {}
        except ValueError:
            result.append(call)
            continue
        if not isinstance(parsed, dict):
            result.append(call)
            continue
        known_props = tuple(types)
        rebuilt: dict[str, Any] = {}
        unknowns: list[str] = []
        for key, value in parsed.items():
            resolved_key, how = _resolve_arg_key(key, known_props, param_aliases)
            if resolved_key is None:
                unknowns.append(key)
                rebuilt[key] = value
                continue
            if resolved_key in rebuilt:
                continue
            if resolved_key != key:
                fixes.append({"call_id": call.id, "kind": "rename", "from": key, "to": resolved_key, "confidence": how})
            rebuilt[resolved_key] = value
        coerced: dict[str, Any] = {}
        for key, value in rebuilt.items():
            new_value, changed = _coerce_by_type(value, types.get(key))
            if changed:
                fixes.append({"call_id": call.id, "kind": "coerce", "param": key, "from": value, "to": new_value})
            enum_values = enums.get(key)
            if isinstance(enum_values, list) and enum_values:
                match = _match_enum(new_value, enum_values)
                if match is not None:
                    if match[0] != new_value:
                        fixes.append({"call_id": call.id, "kind": "enum", "param": key, "from": new_value, "to": match[0]})
                        new_value = match[0]
                else:
                    warnings.append({"call_id": call.id, "kind": "enum_mismatch", "param": key, "value": new_value})
            bound = bounds.get(key)
            if isinstance(bound, dict) and isinstance(new_value, (int, float)) and not isinstance(new_value, bool):
                low = bound.get("minimum")
                high = bound.get("maximum")
                if isinstance(low, (int, float)) and new_value < low:
                    warnings.append({"call_id": call.id, "kind": "out_of_range", "param": key, "value": new_value, "minimum": low})
                if isinstance(high, (int, float)) and new_value > high:
                    warnings.append({"call_id": call.id, "kind": "out_of_range", "param": key, "value": new_value, "maximum": high})
            coerced[key] = new_value
        for key, default_value in defaults.items():
            if key not in coerced:
                coerced[key] = default_value
                fixes.append({"call_id": call.id, "kind": "default", "param": key, "to": default_value})
        for name in required:
            if name not in coerced:
                warnings.append({"call_id": call.id, "kind": "missing_required", "param": name})
        for key in unknowns:
            warnings.append({"call_id": call.id, "kind": "unknown_param", "param": key})
        if apply_fixes:
            if drop_unknown:
                for key in unknowns:
                    coerced.pop(key, None)
            new_arguments = json.dumps(coerced, ensure_ascii=False)
        else:
            new_arguments = call.arguments
        result.append(ToolCall(call.id, call.name, new_arguments))
    if report is not None:
        existing_fixes = report.get("fixes")
        if isinstance(existing_fixes, list):
            existing_fixes.extend(fixes)
        else:
            report["fixes"] = fixes
        existing_warnings = report.get("warnings")
        if isinstance(existing_warnings, list):
            existing_warnings.extend(warnings)
        else:
            report["warnings"] = warnings
    return result
