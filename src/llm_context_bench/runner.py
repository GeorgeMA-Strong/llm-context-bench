#!/usr/bin/env python3
"""Run the locked regular-usage and coding suites against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import multiprocessing
import os
import platform
from pathlib import Path
import re
import shlex
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
import zlib

from . import __version__


ROOT = Path(__file__).resolve().parent
SUITES = {
    "regular": ROOT / "suites" / "regular-v2.json",
    "coding": ROOT / "suites" / "coding-v2.json",
}
SIZE_TOKENS = {
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
}
MODEL_LOAD_WARMUP_NOMINAL_INPUT_TOKENS = 1024
MODEL_LOAD_WARMUP_INPUT_CHARS = 3500
MODEL_LOAD_WARMUP_OUTPUT_TOKENS = 512
MODEL_LOAD_WARMUP_SOURCE = ROOT / "prompts" / "regular-16k.txt"
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "vars",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_suite(name: str) -> tuple[dict, str]:
    raw = SUITES[name].read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def load_model_warmup_prompt() -> str:
    """Return a frozen heterogeneous prompt sized for roughly 1K model tokens."""
    prompt = MODEL_LOAD_WARMUP_SOURCE.read_text(encoding="utf-8")[
        :MODEL_LOAD_WARMUP_INPUT_CHARS
    ]
    if len(prompt) != MODEL_LOAD_WARMUP_INPUT_CHARS:
        raise ValueError("model-load warm-up source is unexpectedly short")
    return prompt


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path: Path, payload: dict) -> None:
    """Durably replace a result JSON without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def redacted_benchmark_argv(argv: list[str]) -> list[str]:
    """Return a shareable argv copy without exposing a CLI API key."""
    redacted = []
    redact_next = False
    for item in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
        elif item == "--api-key":
            redacted.append(item)
            redact_next = True
        elif item.startswith("--api-key="):
            redacted.append("--api-key=<redacted>")
        else:
            redacted.append(item)
    return redacted


def build_run_parameters(args: argparse.Namespace, argv: list[str]) -> dict:
    """Build the repeatability metadata persisted from the first checkpoint."""
    safe_argv = redacted_benchmark_argv(argv)
    return {
        "benchmark_command": shlex.join([Path(sys.executable).name, *safe_argv]),
        "benchmark_argv": safe_argv,
        "command": args.command,
        "system": args.system,
        "model_load_warmup": not args.no_warmup,
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": Path(sys.executable).name,
            "platform": platform.platform(),
        },
    }


def request_json(
    base_url: str,
    path: str,
    payload: dict | None,
    api_key: str,
    timeout: int,
) -> tuple[float, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method="GET" if data is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            body = json.loads(raw) if raw else {}
            body.setdefault("http_status", response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": raw or str(exc)}
        body.setdefault("http_status", exc.code)
    except Exception as exc:  # Network errors are data, not harness crashes.
        body = {"error": str(exc)}
    return time.perf_counter() - started, body


def render_prompt(case: dict) -> str:
    if "prompt" in case:
        return case["prompt"]
    if "prompt_file" in case:
        prompt_path = (ROOT / case["prompt_file"]).resolve()
        if ROOT not in prompt_path.parents:
            raise ValueError(f"prompt path leaves benchmark directory: {prompt_path}")
        raw = prompt_path.read_bytes()
        actual_hash = sha256_bytes(raw)
        if actual_hash != case["prompt_sha256"]:
            raise ValueError(
                f"prompt hash mismatch for {case.get('id')}: "
                f"expected {case['prompt_sha256']}, got {actual_hash}"
            )
        return raw.decode("utf-8")
    raise ValueError(f"case {case.get('id')} has no prompt or prompt_file")


def render_performance_prompt(prompt: str, instruction: str) -> str:
    """Replace the quality task at the end of a long fixture with a TG task."""
    marker = "\nTASK:"
    document = prompt.rsplit(marker, 1)[0] if marker in prompt else prompt
    return document.rstrip() + "\n\n" + instruction


def extract_json(text: str):
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
    return json.loads(candidate)


def extract_artifact(text: str, language: str) -> str:
    candidate = text.strip()
    fence = re.fullmatch(
        rf"```(?:{language})?\s*(.*?)\s*```",
        candidate,
        re.DOTALL | re.IGNORECASE,
    )
    return fence.group(1).strip() if fence else candidate


def validate_python_ast(code: str, function_name: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"
    if not any(isinstance(node, ast.FunctionDef) and node.name == function_name for node in tree.body):
        return False, f"missing function {function_name}"
    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.With,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            return False, f"forbidden syntax: {type(node).__name__}"
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            return False, f"forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"forbidden attribute: {node.attr}"
    return True, "safe subset accepted"


def python_test_worker(code: str, function_name: str, tests: list[dict], queue) -> None:
    try:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        except (ImportError, OSError, ValueError):
            pass
        namespace = {"__builtins__": SAFE_BUILTINS}
        exec(compile(code, "<candidate>", "exec"), namespace, namespace)
        function = namespace[function_name]
        for index, test in enumerate(tests, start=1):
            args = copy.deepcopy(test["args"])
            original = copy.deepcopy(args)
            actual = function(*args)
            if args != original:
                queue.put((False, f"test {index}: input mutated"))
                return
            if actual != test["expected"]:
                queue.put((False, f"test {index}: expected {test['expected']!r}, got {actual!r}"))
                return
        queue.put((True, f"passed {len(tests)} tests"))
    except BaseException as exc:
        queue.put((False, f"execution error: {type(exc).__name__}: {exc}"))


def check_python(text: str, checker: dict) -> tuple[bool, str]:
    code = extract_artifact(text, "python")
    valid, reason = validate_python_ast(code, checker["function"])
    if not valid:
        return False, reason
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(
        target=python_test_worker,
        args=(code, checker["function"], checker["tests"], queue),
    )
    process.start()
    process.join(4)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return False, "execution timed out"
    if queue.empty():
        return False, f"test process exited with code {process.exitcode}"
    return queue.get()


def check_sqlite(text: str, checker: dict) -> tuple[bool, str]:
    query = extract_artifact(text, "sql").strip().rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        return False, "query must start with SELECT or WITH"
    if ";" in query:
        return False, "multiple SQL statements are not allowed"
    forbidden = re.search(
        r"\b(ALTER|ATTACH|CREATE|DELETE|DETACH|DROP|INSERT|PRAGMA|REPLACE|UPDATE|VACUUM)\b",
        query,
        re.IGNORECASE,
    )
    if forbidden:
        return False, f"forbidden SQL keyword: {forbidden.group(1).upper()}"
    connection = sqlite3.connect(":memory:")
    try:
        for statement in checker["setup"]:
            connection.execute(statement)
        cursor = connection.execute(query)
        columns = [column[0] for column in cursor.description or []]
        rows = [list(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        return False, f"SQLite error: {exc}"
    finally:
        connection.close()
    if columns != checker["expected_columns"]:
        return False, f"expected columns {checker['expected_columns']!r}, got {columns!r}"
    if rows != checker["expected_rows"]:
        return False, f"expected rows {checker['expected_rows']!r}, got {rows!r}"
    return True, f"returned {len(rows)} expected rows"


def score_response(text: str, checker: dict) -> tuple[bool, str]:
    checker_type = checker["type"]
    if checker_type == "exact_json":
        try:
            actual = extract_json(text)
        except (json.JSONDecodeError, TypeError) as exc:
            return False, f"invalid JSON: {exc}"
        if actual != checker["expected"]:
            return False, f"JSON mismatch: got {actual!r}"
        return True, "exact JSON match"
    if checker_type == "json_fields":
        try:
            actual = extract_json(text)
        except (json.JSONDecodeError, TypeError) as exc:
            return False, f"invalid JSON: {exc}"
        if list(actual) != checker["exact_keys"]:
            return False, f"expected keys {checker['exact_keys']!r} in order"
        for key, expected in checker["expected_fields"].items():
            if actual.get(key) != expected:
                return False, f"field {key} mismatch"
        for key, required in checker.get("required_text", {}).items():
            value = actual.get(key)
            if not isinstance(value, str) or not all(word.lower() in value.lower() for word in required):
                return False, f"field {key} is missing required terms {required!r}"
        return True, "required JSON fields match"
    if checker_type == "json_rules":
        try:
            actual = extract_json(text)
        except (json.JSONDecodeError, TypeError) as exc:
            return False, f"invalid JSON: {exc}"
        if not isinstance(actual, dict) or list(actual) != checker["exact_keys"]:
            return False, f"expected keys {checker['exact_keys']!r} in order"
        for key, expected in checker.get("scalar_exact", {}).items():
            if actual.get(key) != expected:
                return False, f"field {key} mismatch"
        for key, terms in checker.get("string_terms", {}).items():
            value = actual.get(key)
            if not isinstance(value, str) or not all(term.lower() in value.lower() for term in terms):
                return False, f"field {key} is missing required terms {terms!r}"
        for key, item_rules in checker.get("array_item_terms", {}).items():
            value = actual.get(key)
            if not isinstance(value, list) or len(value) != len(item_rules):
                return False, f"field {key} must contain {len(item_rules)} items"
            for index, (item, terms) in enumerate(zip(value, item_rules), start=1):
                if not isinstance(item, str) or not all(term.lower() in item.lower() for term in terms):
                    return False, f"field {key} item {index} is missing required terms {terms!r}"
        return True, "structured JSON rules match"
    if checker_type == "python_tests":
        return check_python(text, checker)
    if checker_type == "sqlite_query":
        return check_sqlite(text, checker)
    raise ValueError(f"unknown checker type: {checker_type}")


def chat_once(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompt: str,
    max_tokens: int,
    sampling: dict,
    timeout: int,
    ignore_eos: bool = False,
    request_tag: str | None = None,
) -> dict:
    tagged_system_prompt = system_prompt
    if request_tag:
        # Put a deterministic, per-request value before the shared content so
        # prefix/KV caches cannot turn a measured cold-prefill run into a hit.
        tagged_system_prompt = f"[BENCHMARK REQUEST TAG: {request_tag}]\n{system_prompt}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": tagged_system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "seed": sampling["seed"],
        "cache_prompt": False,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False,
    }
    for optional_sampling_key in ("top_k", "min_p"):
        if optional_sampling_key in sampling:
            payload[optional_sampling_key] = sampling[optional_sampling_key]
    if ignore_eos:
        payload["ignore_eos"] = True
    elapsed, body = request_json(base_url, "/v1/chat/completions", payload, api_key, timeout)
    try:
        message = body["choices"][0]["message"]
        text = message.get("content") or ""
        reasoning_text = message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        text = ""
        reasoning_text = ""
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    timings = body.get("timings", {}) if isinstance(body, dict) else {}
    completion_tokens = usage.get("completion_tokens")
    produced_tokens = completion_tokens or timings.get("predicted_n")
    return {
        "ok": bool(text or reasoning_text) and "error" not in body,
        "elapsed_s": round(elapsed, 6),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "output_tps_whole_request": completion_tokens / elapsed if completion_tokens else None,
        "prompt_tps_engine": timings.get("prompt_per_second"),
        "generation_tps_engine": timings.get("predicted_per_second"),
        "draft_tokens": timings.get("draft_n"),
        "accepted_draft_tokens": timings.get("draft_n_accepted"),
        "produced_tokens": produced_tokens,
        "ignore_eos": ignore_eos,
        "request_tag": request_tag,
        "timings": timings,
        "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
        "response": text,
        "reasoning_response": reasoning_text,
        "error": body.get("error") if isinstance(body, dict) else "invalid response",
        "http_status": body.get("http_status") if isinstance(body, dict) else None,
    }


def analyze_output_content(text: str) -> dict:
    """Reject low-entropy forced output such as hundreds of identical dots."""
    raw = text.encode("utf-8")
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
    token_count = len(tokens)
    counts = {token: tokens.count(token) for token in set(tokens)} if tokens else {}
    unique_ratio = len(counts) / token_count if token_count else 0.0
    dominant_ratio = max(counts.values()) / token_count if counts else 1.0
    compression_ratio = len(zlib.compress(raw, 9)) / len(raw) if raw else 0.0
    reasons = []
    if token_count < 128:
        reasons.append("too_few_text_tokens")
    if unique_ratio < 0.08:
        reasons.append("low_unique_token_ratio")
    if dominant_ratio > 0.20:
        reasons.append("dominant_repeated_token")
    if compression_ratio < 0.15:
        reasons.append("highly_compressible_repetition")
    return {
        "valid": not reasons,
        "text_token_count": token_count,
        "unique_token_ratio": round(unique_ratio, 6),
        "dominant_token_ratio": round(dominant_ratio, 6),
        "compression_ratio": round(compression_ratio, 6),
        "reasons": reasons,
    }


def validate_engine_timings(trial: dict) -> dict:
    """Cross-check llama.cpp timing rates and cache/token counters."""
    timings = trial.get("timings") or {}
    prompt_n = timings.get("prompt_n")
    prompt_ms = timings.get("prompt_ms")
    prompt_reported = timings.get("prompt_per_second")
    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    predicted_reported = timings.get("predicted_per_second")

    def relative_error(actual, expected):
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return None
        if expected == 0:
            return 0.0 if actual == 0 else None
        return abs(actual - expected) / abs(expected)

    prompt_calculated = (
        prompt_n * 1000.0 / prompt_ms
        if isinstance(prompt_n, (int, float))
        and isinstance(prompt_ms, (int, float))
        and prompt_ms > 0
        else None
    )
    # llama.cpp excludes the first generated token from decode throughput
    # because that token is sampled from the final prompt-evaluation logits.
    generation_calculated = (
        max(predicted_n - 1, 0) * 1000.0 / predicted_ms
        if isinstance(predicted_n, (int, float))
        and isinstance(predicted_ms, (int, float))
        and predicted_ms > 0
        else None
    )
    prompt_error = relative_error(prompt_reported, prompt_calculated)
    generation_error = relative_error(predicted_reported, generation_calculated)
    cache_n = timings.get("cache_n")
    prompt_usage = trial.get("prompt_tokens")
    completion_usage = trial.get("completion_tokens")
    token_counts_consistent = (
        isinstance(prompt_usage, (int, float))
        and isinstance(completion_usage, (int, float))
        and isinstance(prompt_n, (int, float))
        and isinstance(cache_n, (int, float))
        and isinstance(predicted_n, (int, float))
        and prompt_usage == prompt_n + cache_n
        and completion_usage == predicted_n
    )
    return {
        "valid": (
            prompt_error is not None
            and generation_error is not None
            and prompt_error <= 0.01
            and generation_error <= 0.01
            and token_counts_consistent
        ),
        "prompt_tps_calculated": prompt_calculated,
        "generation_tps_calculated": generation_calculated,
        "prompt_tps_relative_error": prompt_error,
        "generation_tps_relative_error": generation_error,
        "token_counts_consistent": token_counts_consistent,
        "cache_n": cache_n,
        "no_prompt_cache": cache_n == 0,
    }


def summarize_trials(trials: list[dict]) -> dict:
    elapsed = [trial["elapsed_s"] for trial in trials if trial["ok"]]
    prompt_tokens = [trial["prompt_tokens"] for trial in trials if trial["prompt_tokens"] is not None]
    output_tps_whole_request = [
        trial["output_tps_whole_request"]
        for trial in trials
        if trial["output_tps_whole_request"] is not None
    ]
    prompt_tps = [trial["prompt_tps_engine"] for trial in trials if trial["prompt_tps_engine"] is not None]
    generation_tps = [
        trial["generation_tps_engine"]
        for trial in trials
        if trial["generation_tps_engine"] is not None
    ]
    return {
        "repetitions": len(trials),
        "successful_requests": sum(bool(trial["ok"]) for trial in trials),
        "passed_trials": sum(bool(trial.get("passed")) for trial in trials),
        "pass_rate": (
            sum(bool(trial.get("passed")) for trial in trials) / len(trials)
            if trials
            else None
        ),
        "median_elapsed_s": statistics.median(elapsed) if elapsed else None,
        "median_prompt_tokens": statistics.median(prompt_tokens) if prompt_tokens else None,
        "median_output_tps_whole_request": (
            statistics.median(output_tps_whole_request)
            if output_tps_whole_request
            else None
        ),
        "median_prompt_tps_engine": statistics.median(prompt_tps) if prompt_tps else None,
        "median_generation_tps_engine": statistics.median(generation_tps) if generation_tps else None,
    }


def summarize_performance_trials(
    trials: list[dict], required_output_tokens: int, nominal_input_tokens: int
) -> dict:
    valid_trials = [trial for trial in trials if trial.get("performance_valid", False)]
    summary = summarize_trials(valid_trials)
    summary["repetitions"] = len(trials)
    summary["successful_requests"] = sum(bool(trial["ok"]) for trial in trials)
    summary.pop("passed_trials")
    summary.pop("pass_rate")
    summary["required_output_tokens"] = required_output_tokens
    summary["fixed_length_valid_trials"] = sum(
        trial.get("fixed_length_valid", False) for trial in trials
    )
    summary["all_trials_fixed_length"] = bool(trials) and all(
        trial.get("fixed_length_valid", False) for trial in trials
    )
    summary["nominal_input_tokens"] = nominal_input_tokens
    summary["input_size_tolerance_percent"] = 2.0
    summary["all_trials_input_size_valid"] = bool(trials) and all(
        trial.get("input_size_valid", False)
        for trial in trials
    )
    summary["all_trials_output_content_valid"] = bool(trials) and all(
        trial.get("output_content", {}).get("valid", False) for trial in trials
    )
    summary["all_trials_engine_timings_valid"] = bool(trials) and all(
        trial.get("engine_timing_validation", {}).get("valid", False)
        for trial in trials
    )
    summary["all_trials_no_prompt_cache"] = bool(trials) and all(
        trial.get("engine_timing_validation", {}).get("no_prompt_cache", False)
        for trial in trials
    )
    summary["valid_performance_trials"] = len(valid_trials)
    summary["performance_valid"] = bool(trials) and len(valid_trials) == len(trials)
    acceptance_rates = [
        trial["draft_acceptance_rate"]
        for trial in valid_trials
        if trial.get("draft_acceptance_rate") is not None
    ]
    summary["median_draft_acceptance_rate"] = (
        statistics.median(acceptance_rates) if acceptance_rates else None
    )
    return summary


def summarize_suite_cases(cases: list[dict]) -> dict:
    quality_cases = [case["quality"] for case in cases if "quality" in case]
    performance_cases = [case["performance"] for case in cases if "performance" in case]
    total_trials = sum(len(case["trials"]) for case in quality_cases)
    passed_trials = sum(case["summary"]["passed_trials"] for case in quality_cases)
    return {
        "quality_cases": len(quality_cases),
        "quality_trials": total_trials,
        "passed_quality_trials": passed_trials,
        "quality_score": passed_trials / total_trials if total_trials else None,
        "fully_passing_quality_cases": sum(
            case["summary"]["pass_rate"] == 1.0 for case in quality_cases
        ),
        "performance_cases": len(performance_cases),
        "valid_performance_cases": sum(
            case["summary"]["performance_valid"] for case in performance_cases
        ),
    }


def run_suite(
    suite_name: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    repetitions_override: int | None,
    lane: str,
    sizes: list[str],
    checkpoint_callback=None,
) -> dict:
    suite, suite_hash = load_suite(suite_name)
    repetitions = repetitions_override or int(suite["repetitions"])
    cases = []
    suite_result = {
        "suite_id": suite["suite_id"],
        "suite_file": f"llm_context_bench/suites/{SUITES[suite_name].name}",
        "suite_sha256": suite_hash,
        "quality_sampling": suite["sampling"],
        "performance_sampling": suite["performance_sampling"],
        "repetitions": repetitions,
        "lane": lane,
        "sizes": sizes,
        "status": "running",
        "cases": cases,
        "summary": summarize_suite_cases(cases),
    }

    def checkpoint(event: str) -> None:
        suite_result["last_event"] = event
        suite_result["summary"] = summarize_suite_cases(cases)
        if checkpoint_callback is not None:
            checkpoint_callback(suite_result, event)

    selected_tokens = {SIZE_TOKENS[value] for value in sizes if value != "all"}
    for case in suite["cases"]:
        if "all" not in sizes and case.get("nominal_input_tokens") not in selected_tokens:
            continue
        if lane == "performance" and "performance_max_tokens" not in case:
            continue
        prompt = render_prompt(case)
        case_result = {
            "id": case["id"],
            "nominal_input_tokens": case.get("nominal_input_tokens"),
            "input_chars": len(prompt),
            "input_sha256": sha256_bytes(prompt.encode("utf-8")),
            "prompt_file": case.get("prompt_file"),
        }
        cases.append(case_result)
        checkpoint(f"{case['id']}:started")
        if lane in ("quality", "all"):
            case_result["quality"] = {
                "max_tokens": case["max_tokens"],
                "trials": [],
                "summary": summarize_trials([]),
            }
            trials = []
            case_result["quality"]["trials"] = trials
            for index in range(1, repetitions + 1):
                trial = chat_once(
                    base_url,
                    api_key,
                    model,
                    suite["system_prompt"],
                    prompt,
                    int(case["max_tokens"]),
                    suite["sampling"],
                    timeout,
                    request_tag=f"{case['id']}-quality-{index:02d}",
                )
                if trial["ok"]:
                    passed, detail = score_response(trial["response"], case["checker"])
                else:
                    passed, detail = False, f"request failed: {trial['error']}"
                trial["passed"] = passed
                trial["score_detail"] = detail
                trials.append(trial)
                case_result["quality"]["summary"] = summarize_trials(trials)
                checkpoint(f"{case['id']}:quality:trial-{index:02d}")
        if lane in ("performance", "all") and "performance_max_tokens" in case:
            required_tokens = int(case["performance_max_tokens"])
            performance_prompt = render_performance_prompt(
                prompt, suite["performance_instruction"]
            )
            case_result["performance"] = {
                "max_tokens": required_tokens,
                "ignore_eos": False,
                "cold_cache_via_unique_request_prefix": True,
                "input_chars": len(performance_prompt),
                "input_sha256": sha256_bytes(performance_prompt.encode("utf-8")),
                "trials": [],
                "summary": summarize_performance_trials(
                    [], required_tokens, int(case["nominal_input_tokens"])
                ),
            }
            performance_trials = []
            case_result["performance"]["trials"] = performance_trials
            for index in range(1, repetitions + 1):
                trial = chat_once(
                    base_url,
                    api_key,
                    model,
                    suite["performance_system_prompt"],
                    performance_prompt,
                    required_tokens,
                    suite["performance_sampling"],
                    timeout,
                    request_tag=f"{case['id']}-performance-{index:02d}",
                )
                minimum_input = int(case["nominal_input_tokens"]) * 0.98
                maximum_input = int(case["nominal_input_tokens"]) * 1.02
                trial["fixed_length_valid"] = (
                    trial["ok"] and trial["produced_tokens"] == required_tokens
                )
                trial["input_size_valid"] = (
                    trial["prompt_tokens"] is not None
                    and minimum_input <= trial["prompt_tokens"] <= maximum_input
                )
                generated_text = "\n".join(
                    part
                    for part in (
                        trial.get("reasoning_response", ""),
                        trial.get("response", ""),
                    )
                    if part
                )
                trial["output_content"] = analyze_output_content(generated_text)
                trial["engine_timing_validation"] = validate_engine_timings(trial)
                drafted = trial.get("draft_tokens")
                accepted = trial.get("accepted_draft_tokens")
                trial["draft_acceptance_rate"] = (
                    accepted / drafted
                    if isinstance(drafted, (int, float))
                    and drafted > 0
                    and isinstance(accepted, (int, float))
                    else None
                )
                trial["performance_valid"] = (
                    trial["fixed_length_valid"]
                    and trial["input_size_valid"]
                    and trial["output_content"]["valid"]
                    and trial["engine_timing_validation"]["valid"]
                    and trial["engine_timing_validation"]["no_prompt_cache"]
                )
                performance_trials.append(trial)
                case_result["performance"]["summary"] = summarize_performance_trials(
                    performance_trials,
                    required_tokens,
                    int(case["nominal_input_tokens"]),
                )
                checkpoint(f"{case['id']}:performance:trial-{index:02d}")
        checkpoint(f"{case['id']}:complete")
    suite_result["status"] = "complete"
    checkpoint("suite:complete")
    return suite_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible server, e.g. http://127.0.0.1:8080")
    parser.add_argument("--model", required=True, help="Exact API model identifier")
    parser.add_argument("--profile", required=True, help="Unique model/build/hardware profile name")
    parser.add_argument("--output", required=True, type=Path, help="Result JSON path")
    parser.add_argument("--suite", choices=("regular", "coding", "all"), default="all")
    parser.add_argument("--lane", choices=("quality", "performance", "all"), default="all")
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=("16k", "32k", "64k", "128k", "all"),
        default=["all"],
        help="Run a list of long-input sizes, or 'all'",
    )
    parser.add_argument("--api-key", default=os.environ.get("LLM_BENCH_API_KEY", ""))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--command",
        required=True,
        help="Full llama.cpp server command including global/environment variables; saved unchanged",
    )
    parser.add_argument(
        "--system",
        required=True,
        help="Free-form system description: cards, OS, ROCm/CUDA, llama.cpp version, etc.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Override the suite default of one measured repetition (use an explicit value for repeats)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the single 1K-input/512-output model-load warm-up",
    )
    args = parser.parse_args(argv)
    if args.repetitions is not None and args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if "all" in args.sizes and len(args.sizes) > 1:
        parser.error("--sizes all cannot be combined with individual sizes")
    args.sizes = list(dict.fromkeys(args.sizes))
    return args


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    suite_names = list(SUITES) if args.suite == "all" else [args.suite]
    result = {
        "benchmark_schema_version": 6,
        "tool": {"name": "llm-context-bench", "version": __version__},
        "status": "running",
        "profile": args.profile,
        "model": args.model,
        "base_url": base_url,
        "started_at": utc_now(),
        "lane": args.lane,
        "sizes": args.sizes,
        "canonical": args.repetitions is None,
        "harness_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "run_parameters": build_run_parameters(args, sys.argv),
        "environment": {},
        "suites": {},
        "checkpoint_count": 0,
    }

    def checkpoint(event: str) -> None:
        result["checkpoint_count"] += 1
        result["last_checkpoint"] = {"event": event, "at": utc_now()}
        write_json_atomic(args.output, result)

    checkpoint("benchmark:initialized")
    try:
        health_elapsed, health = request_json(base_url, "/health", None, args.api_key, 20)
        result["environment"]["health_elapsed_s"] = round(health_elapsed, 6)
        result["environment"]["health"] = health
        checkpoint("environment:health")

        if not args.no_warmup:
            warmup_prompt = load_model_warmup_prompt()
            warmup = chat_once(
                base_url,
                args.api_key,
                args.model,
                "Model initialization warm-up. Continue the supplied document.",
                warmup_prompt,
                MODEL_LOAD_WARMUP_OUTPUT_TOKENS,
                {"temperature": 0.0, "top_p": 1.0, "seed": 3407},
                args.timeout,
                ignore_eos=True,
                request_tag="model-load-warmup",
            )
            warmup["nominal_input_tokens"] = MODEL_LOAD_WARMUP_NOMINAL_INPUT_TOKENS
            warmup["input_chars"] = len(warmup_prompt)
            warmup["input_sha256"] = sha256_bytes(warmup_prompt.encode("utf-8"))
            warmup["required_output_tokens"] = MODEL_LOAD_WARMUP_OUTPUT_TOKENS
            warmup["fixed_length_valid"] = (
                warmup["ok"]
                and warmup["produced_tokens"] == MODEL_LOAD_WARMUP_OUTPUT_TOKENS
            )
            result["environment"]["warmup"] = warmup
            checkpoint("environment:model-load-warmup")

        for suite_name in suite_names:
            result["suites"][suite_name] = {"status": "starting"}
            checkpoint(f"{suite_name}:starting")

            def suite_checkpoint(partial_suite: dict, event: str, name=suite_name) -> None:
                result["suites"][name] = partial_suite
                checkpoint(f"{name}:{event}")

            result["suites"][suite_name] = run_suite(
                suite_name,
                base_url,
                args.api_key,
                args.model,
                args.timeout,
                args.repetitions,
                args.lane,
                args.sizes,
                checkpoint_callback=suite_checkpoint,
            )
        result["status"] = "complete"
        result["finished_at"] = utc_now()
        checkpoint("benchmark:complete")
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["finished_at"] = utc_now()
        result["error"] = "KeyboardInterrupt"
        checkpoint("benchmark:interrupted")
        raise
    except Exception as exc:
        result["status"] = "failed"
        result["finished_at"] = utc_now()
        result["error"] = f"{type(exc).__name__}: {exc}"
        checkpoint("benchmark:failed")
        raise

    compact = {
        name: suite["summary"] for name, suite in result["suites"].items()
    }
    print(json.dumps({"profile": args.profile, "canonical": result["canonical"], "suites": compact}, indent=2))


if __name__ == "__main__":
    main()
