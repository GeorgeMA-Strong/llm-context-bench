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
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

from . import __version__
from . import engine as engine_api


ROOT = Path(__file__).resolve().parent
SUITES = {
    "regular": ROOT / "suites" / "regular-v2.json",
    "coding": ROOT / "suites" / "coding-v2.json",
}
SIZE_TOKENS = {
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
}
INPUT_SIZE_TOLERANCE_PERCENT = 2.0
# Above this the measured engine is usually the bottleneck being the harness,
# not the model, and one lost request starts to dominate the group aggregate.
MAX_CONCURRENCY = 8
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


def harness_fingerprint() -> str:
    """Fingerprint every module that can influence a measurement."""
    digest = hashlib.sha256()
    for name in ("engine.py", "runner.py"):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


# The engine module owns this predicate so request parsing and metric maths
# agree on what counts as a usable number.
numeric = engine_api.numeric


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


def is_canonical(args: argparse.Namespace) -> bool:
    """True only when the run matches the locked suite definition.

    More repetitions or several requests in flight both change what is being
    measured, so those runs are recorded as diagnostics and never as a
    canonical leaderboard entry.
    """
    return args.repetitions is None and args.concurrency == 1


def build_run_parameters(args: argparse.Namespace, argv: list[str]) -> dict:
    """Build the repeatability metadata persisted from the first checkpoint."""
    safe_argv = redacted_benchmark_argv(argv)
    return {
        "benchmark_command": shlex.join([Path(sys.executable).name, *safe_argv]),
        "benchmark_argv": safe_argv,
        "command": args.command,
        "system": args.system,
        "concurrency": args.concurrency,
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
    if os.environ.get("LLM_CONTEXT_BENCH_HTTP_CLIENT") == "curl":
        # Useful on hosts where the bundled Python HTTP client does not
        # interoperate with a particular OpenAI-compatible server.  curl is
        # intentionally opt-in so the normal dependency-free client remains
        # the default.
        marker = b"\n__LLM_CONTEXT_BENCH_STATUS__:"
        command = [
            "curl",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout),
            "--connect-timeout",
            str(min(timeout, 15)),
            "-H",
            "Content-Type: application/json",
            "-w",
            "\n__LLM_CONTEXT_BENCH_STATUS__:%{http_code}",
        ]
        if api_key:
            command.extend(["-H", f"Authorization: Bearer {api_key}"])
        if data is not None:
            command.extend(["--data-binary", "@-"])
        command.append(base_url.rstrip("/") + path)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout + 5,
                check=False,
            )
            raw, separator, status = completed.stdout.rpartition(marker)
            if not separator:
                body = {"error": completed.stderr.decode("utf-8", "replace") or "curl returned no status"}
            else:
                try:
                    body = json.loads(raw.decode("utf-8", "replace")) if raw else {}
                except json.JSONDecodeError:
                    body = {"error": raw.decode("utf-8", "replace") or "invalid JSON response"}
                body.setdefault("http_status", int(status.decode("ascii", "replace") or 0))
                if completed.returncode and "error" not in body:
                    body["error"] = completed.stderr.decode("utf-8", "replace") or f"curl exited {completed.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            body = {"error": str(exc)}
        return time.perf_counter() - started, body
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method="GET" if data is None else "POST",
        # Some OpenAI-compatible servers leave an idle keep-alive socket open
        # after completing a response.  Bench requests are independent, so
        # close each connection explicitly rather than waiting for that socket.
        headers={"Content-Type": "application/json", "Connection": "close"},
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


def request_sse(
    base_url: str,
    path: str,
    payload: dict,
    api_key: str,
    timeout: int,
    client: str = "urllib",
) -> dict:
    """Streaming transport seam, substitutable in tests like ``request_json``."""
    return engine_api.request_sse(base_url, path, payload, api_key, timeout, client=client)


def non_stream_measured(elapsed: float, body: dict) -> dict:
    """Present a non-streaming response in the same shape as a captured stream."""
    text = ""
    reasoning_text = ""
    finish_reason = None
    try:
        choice = body["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        reasoning_text = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return {
        "ok": bool(text or reasoning_text) and "error" not in body,
        "error": body.get("error") if isinstance(body, dict) else "invalid response",
        "http_status": body.get("http_status") if isinstance(body, dict) else None,
        "elapsed_s": elapsed,
        "response": text,
        "reasoning_response": reasoning_text,
        "usage": body.get("usage", {}) if isinstance(body, dict) else {},
        "timings": body.get("timings", {}) if isinstance(body, dict) else {},
        "finish_reason": finish_reason,
        "first_token_at": None,
        "last_token_at": None,
        "stream_chunks": 0,
        "content_chars": len(text),
        "done_at": None,
        "parse_errors": 0,
        "stream_truncated": False,
    }


def build_trial(
    measured: dict,
    *,
    engine: str,
    stream: bool,
    dropped_params: list[str],
    ignore_eos: bool,
    request_tag: str | None,
    attempts: int,
    retry_statuses: list,
) -> dict:
    """Turn one captured response into a trial with harness-calculated rates."""
    timings = measured.get("timings") or {}
    counts = engine_api.token_counts(measured)
    trial = {
        "ok": measured["ok"],
        "elapsed_s": round(measured["elapsed_s"], 6),
        "prompt_tokens": counts["prompt_tokens"],
        "completion_tokens": counts["completion_tokens"],
        "produced_tokens": counts["completion_tokens"],
        "token_count_source": counts["token_count_source"],
        "engine": engine,
        "stream": stream,
        "stream_chunks": measured["stream_chunks"],
        "content_chars": measured["content_chars"],
        "engine_only_params_dropped": dropped_params,
        "ignore_eos": ignore_eos,
        "ignore_eos_applied": bool(ignore_eos and "ignore_eos" not in dropped_params),
        "request_tag": request_tag,
        "attempts": attempts,
        "retry_statuses": retry_statuses,
        "timings": timings,
        # Engine-reported rates are retained only to cross-check the measured
        # ones; they are never the source of a reported throughput number.
        "prompt_tps_engine": timings.get("prompt_per_second"),
        "generation_tps_engine": timings.get("predicted_per_second"),
        "draft_tokens": timings.get("draft_n"),
        "accepted_draft_tokens": timings.get("draft_n_accepted"),
        "prompt_cache": engine_api.detect_cached_tokens(measured),
        "finish_reason": measured["finish_reason"],
        "response": measured["response"],
        "reasoning_response": measured["reasoning_response"],
        "error": measured["error"],
        "http_status": measured["http_status"],
    }
    trial.update(
        engine_api.compute_timing_metrics(
            elapsed_s=measured["elapsed_s"],
            prompt_tokens=counts["prompt_tokens"],
            completion_tokens=counts["completion_tokens"],
            first_token_at=measured["first_token_at"],
            last_token_at=measured["last_token_at"],
            stream_chunks=measured["stream_chunks"],
            streamed=stream,
        )
    )
    trial["engine_cross_check"] = engine_api.cross_check_engine_timings(trial)
    drafted = trial["draft_tokens"]
    accepted = trial["accepted_draft_tokens"]
    trial["draft_acceptance_rate"] = (
        accepted / drafted
        if numeric(drafted) and drafted > 0 and numeric(accepted)
        else None
    )
    return trial


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
    chat_template_kwargs: dict | None = None,
    engine: str = "llama.cpp",
    stream: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 1.0,
) -> dict:
    tagged_system_prompt = system_prompt
    if request_tag:
        # Put a deterministic, per-request value before the shared content so
        # prefix/KV caches cannot turn a measured cold-prefill run into a hit.
        tagged_system_prompt = f"[BENCHMARK REQUEST TAG: {request_tag}]\n{system_prompt}"
    payload, dropped_params = engine_api.build_chat_payload(
        engine=engine,
        model=model,
        messages=[
            {"role": "system", "content": tagged_system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        sampling=sampling,
        stream=stream,
        ignore_eos=ignore_eos,
        chat_template_kwargs=chat_template_kwargs,
    )
    client = "curl" if os.environ.get("LLM_CONTEXT_BENCH_HTTP_CLIENT") == "curl" else "urllib"
    attempts = 0
    retry_statuses: list = []
    measured = {}
    for attempt in range(max_retries + 1):
        attempts += 1
        if stream:
            measured = request_sse(
                base_url, "/v1/chat/completions", payload, api_key, timeout, client=client
            )
        else:
            elapsed, body = request_json(
                base_url, "/v1/chat/completions", payload, api_key, timeout
            )
            measured = non_stream_measured(
                elapsed, body if isinstance(body, dict) else {"error": "invalid response"}
            )
        status = measured["http_status"]
        # A connection that died before producing anything is worth another
        # attempt; a stalled or truncated stream is not retried, because the
        # server was demonstrably working and a retry would hide that.
        silent_failure = (
            status is None and not measured["response"] and measured["stream_chunks"] == 0
        )
        retryable = status in engine_api.RETRYABLE_STATUS or silent_failure
        if measured["ok"] or not retryable or attempt >= max_retries:
            break
        retry_statuses.append(status)
        time.sleep(retry_backoff_s * (2**attempt))
    return build_trial(
        measured,
        engine=engine,
        stream=stream,
        dropped_params=dropped_params,
        ignore_eos=ignore_eos,
        request_tag=request_tag,
        attempts=attempts,
        retry_statuses=retry_statuses,
    )


def failed_trial(error: str, *, engine: str, stream: bool, request_tag: str | None) -> dict:
    """Build the trial for a request that never produced a measurable response."""
    return build_trial(
        engine_api.empty_stream_measurement(error=error),
        engine=engine,
        stream=stream,
        dropped_params=[],
        ignore_eos=False,
        request_tag=request_tag,
        attempts=0,
        retry_statuses=[],
    )


def run_concurrent_group(
    request,
    *,
    concurrency: int,
    engine: str = "llama.cpp",
    stream: bool = True,
    barrier_timeout_s: float = 120.0,
) -> tuple[list[dict], dict]:
    """Release ``concurrency`` copies of one request at the same instant.

    ``request(index)`` returns a single trial.  Every worker waits on a barrier
    so no request can start before all of them are ready; without that,
    concurrent requests drift apart and the aggregate throughput describes a
    sequential run.  Trials come back in request-index order, together with the
    window during which all of them were in flight.
    """
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")

    def invoke(index: int) -> dict:
        try:
            return request(index)
        except Exception as exc:  # One lost request must not lose the group.
            return failed_trial(
                f"request worker failed: {type(exc).__name__}: {exc}",
                engine=engine,
                stream=stream,
                request_tag=None,
            )

    if concurrency == 1:
        started = time.perf_counter()
        trial = invoke(1)
        return [trial], {
            "wall_s": time.perf_counter() - started,
            "barrier_broken": False,
        }

    barrier = threading.Barrier(concurrency, timeout=barrier_timeout_s)
    finished: dict[int, dict] = {}
    windows: dict[int, tuple[float, float]] = {}
    state = {"barrier_broken": False}
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            # A worker never arrived.  Its timings are still real, but the run
            # can no longer claim the requests were simultaneous.
            with lock:
                state["barrier_broken"] = True
        began = time.perf_counter()
        trial = invoke(index)
        with lock:
            finished[index] = trial
            windows[index] = (began, time.perf_counter())

    threads = [
        threading.Thread(target=worker, args=(index,), name=f"bench-{index:02d}")
        for index in range(1, concurrency + 1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    trials = [
        finished.get(index)
        or failed_trial(
            f"request {index} never reported a result",
            engine=engine,
            stream=stream,
            request_tag=None,
        )
        for index in range(1, concurrency + 1)
    ]
    # Stamp when each request actually left the harness, relative to the first
    # one.  A stream's own timestamps are relative to its own start, so without
    # this offset the group cannot tell which stream began generating first.
    group_start = min(start for start, _end in windows.values()) if windows else None
    if group_start is not None:
        for index, (start, _end) in windows.items():
            finished[index]["group_start_offset_s"] = round(start - group_start, 6)
    # Wall time of the group: the first request leaving the harness until the
    # last response is complete.  A straggler thread can only lengthen it, which
    # lowers the reported totals instead of inflating them.
    wall_s = (
        max(end for _start, end in windows.values())
        - min(start for start, _end in windows.values())
        if len(windows) == concurrency
        else None
    )
    return trials, {"wall_s": wall_s, "barrier_broken": state["barrier_broken"]}


# Gates that judge the generated text rather than the measurement.  A group
# that fails only these still says something true about engine throughput, but
# it is not a clean measurement, so it stays invalid and gets flagged instead.
CONTENT_GATES = frozenset(
    {
        "too_few_text_tokens",
        "low_unique_token_ratio",
        "dominant_repeated_token",
        "highly_compressible_repetition",
    }
)


def summarize_load_group(
    trials: list[dict],
    *,
    group_id: str,
    concurrency: int,
    wall_s: float | None,
    barrier_broken: bool = False,
) -> dict:
    """Total up the requests one load group had in flight together.

    ``*_total`` numbers describe the engine under load: every token the group
    moved, divided by the group's own wall time.  The per-request medians
    describe one user.  Both are reported because they answer different
    questions - a engine can hold per-stream speed steady while total output
    climbs, or do neither.
    """
    def values(key: str) -> list:
        return [trial[key] for trial in trials if numeric(trial.get(key))]

    def total(key: str):
        # A partial token sum would overstate throughput for the whole window.
        counted = values(key)
        return sum(counted) if len(counted) == len(trials) else None

    reasons: dict[str, int] = {}
    for trial in trials:
        for reason in trial.get("invalid_reasons") or []:
            reasons[reason] = reasons.get(reason, 0) + 1
    if barrier_broken:
        reasons["concurrency_barrier_broken"] = reasons.get("concurrency_barrier_broken", 0) + 1
    measured_wall = numeric(wall_s) and wall_s > 0
    if not measured_wall:
        reasons["group_wall_unmeasured"] = reasons.get("group_wall_unmeasured", 0) + 1
    # At temperature 1.0 a single stream can collapse into repetition, and at
    # concurrency 8 that is a routine event rather than an anomaly.  Say so, so
    # a rejected group is not mistaken for an engine throughput result.
    content_only_failure = bool(reasons) and set(reasons) <= CONTENT_GATES

    def per_wall(tokens):
        return round(tokens / wall_s, 6) if measured_wall and tokens is not None else None

    output_tokens = total("completion_tokens")
    prompt_tokens = total("prompt_tokens")
    ttft = values("ttft_s")
    # The group's prefill phase ends when the slowest stream produces its first
    # token, because that is when the last prompt finished being digested.
    prefill_phase_s = max(ttft) if ttft and len(ttft) == len(trials) else None
    prefill_tps_total = (
        round(prompt_tokens / prefill_phase_s, 6)
        if prompt_tokens is not None and numeric(prefill_phase_s) and prefill_phase_s > 0
        else None
    )
    # Combined token generation, in llama.cpp's TG terms: every generated token
    # except each stream's first, over the window in which the engine was
    # generating for this group.  This is the sum of what the streams produced
    # together, not a per-stream rate multiplied by the request count - streams
    # share the engine, so the honest total is the only useful one.
    generating = [
        trial
        for trial in trials
        if numeric(trial.get("ttft_s"))
        and numeric(trial.get("decode_window_s"))
        and numeric(trial.get("completion_tokens"))
    ]
    generating_window_s = None
    token_generation_tps_total = None
    generating_tokens = None
    if len(generating) == len(trials):
        arrivals = [
            trial.get("group_start_offset_s", 0.0) + trial["ttft_s"]
            for trial in generating
        ]
        departures = [
            trial.get("group_start_offset_s", 0.0) + trial["ttft_s"] + trial["decode_window_s"]
            for trial in generating
        ]
        generating_window_s = max(departures) - min(arrivals)
        generating_tokens = sum(trial["completion_tokens"] - 1 for trial in generating)
        if generating_window_s > 0 and generating_tokens > 0:
            token_generation_tps_total = round(generating_tokens / generating_window_s, 6)
    completed = sum(1 for trial in trials if trial.get("ok"))
    valid = [trial for trial in trials if trial.get("performance_valid")]
    return {
        "group_id": group_id,
        "concurrency": concurrency,
        "requests": concurrency,
        "recorded_requests": len(trials),
        "missing_requests": max(concurrency - len(trials), 0),
        "completed_requests": completed,
        "valid_requests": len(valid),
        "group_valid": (
            len(valid) == concurrency
            and len(trials) == concurrency
            and measured_wall
            and not barrier_broken
        ),
        "content_only_failure": content_only_failure,
        "wall_s": round(wall_s, 6) if measured_wall else None,
        "prompt_tokens_total": prompt_tokens,
        "output_tokens_total": output_tokens,
        "tokens_total": (
            output_tokens + prompt_tokens
            if output_tokens is not None and prompt_tokens is not None
            else None
        ),
        "prefill_phase_s": round(prefill_phase_s, 6) if numeric(prefill_phase_s) else None,
        "prefill_tps_total": prefill_tps_total,
        "generating_window_s": (
            round(generating_window_s, 6) if numeric(generating_window_s) else None
        ),
        "generating_tokens_total": generating_tokens,
        "token_generation_tps_total": token_generation_tps_total,
        "decode_tps_total": per_wall(output_tokens),
        "tokens_tps_total": per_wall(
            output_tokens + prompt_tokens
            if output_tokens is not None and prompt_tokens is not None
            else None
        ),
        "requests_per_minute": (
            round(completed * 60.0 / wall_s, 6) if measured_wall else None
        ),
        "ttft_median_s": _median(ttft),
        "ttft_max_s": round(max(ttft), 6) if ttft else None,
        "prefill_tps_median": _median(values("prefill_tps")),
        "generation_tps_median": _median(values("generation_tps")),
        "invalid_reason_totals": dict(sorted(reasons.items())),
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


def evaluate_performance_trial(
    trial: dict,
    *,
    required_output_tokens: int,
    nominal_input_tokens: int,
    tolerance_percent: float,
) -> None:
    """Apply the engine-agnostic validity gates and record every failure reason.

    Nothing here depends on an engine's private counters.  A trial is valid
    when the request succeeded, the output is exactly the locked length, the
    prompt tokenized inside the requested tier, the generated text is not
    degenerate, and the engine did not report prompt-cache reuse.  Harness
    rates are reported whenever the transport allows them to be calculated.
    """
    prompt_tokens = trial.get("prompt_tokens")
    nominal = float(nominal_input_tokens)
    trial["fixed_length_valid"] = (
        bool(trial.get("ok")) and trial.get("produced_tokens") == required_output_tokens
    )
    trial["input_size_percent_from_nominal"] = (
        round((prompt_tokens - nominal) / nominal * 100.0, 4)
        if numeric(prompt_tokens)
        else None
    )
    trial["input_size_valid"] = bool(
        numeric(prompt_tokens)
        and abs(trial["input_size_percent_from_nominal"]) <= tolerance_percent
    )
    generated_text = "\n".join(
        part
        for part in (trial.get("reasoning_response", ""), trial.get("response", ""))
        if part
    )
    trial["output_content"] = analyze_output_content(generated_text)
    cache = trial.get("prompt_cache") or {}
    cached_tokens = cache.get("cached_tokens")
    trial["prompt_cache_hit"] = bool(numeric(cached_tokens) and cached_tokens > 0)
    trial["engine_timings_reported"] = bool(trial.get("timings"))
    reasons: list[str] = []
    if not trial.get("ok"):
        reasons.append("request_failed")
    if not trial["fixed_length_valid"]:
        reasons.append("output_length_mismatch")
    if not numeric(prompt_tokens):
        # Without a reported prompt length, no tier claim can be verified.
        reasons.append("prompt_tokens_not_reported")
    elif not trial["input_size_valid"]:
        reasons.append("input_size_outside_tolerance")
    if not trial["output_content"]["valid"]:
        reasons.extend(trial["output_content"]["reasons"])
    if trial["prompt_cache_hit"]:
        reasons.append("prompt_cache_hit")
    if trial.get("stream") and trial.get("stream_delivery") != "incremental":
        # A buffered stream hides the decode window, which would turn the whole
        # request duration into reported "prefill" time.  That is not a
        # measurement, so the trial is rejected instead of mislabelled.
        reasons.append("stream_not_incremental")
    trial["invalid_reasons"] = reasons
    notes: list[str] = []
    if not cache.get("reported"):
        notes.append("prompt_cache_not_reported_by_engine")
    if not trial["engine_timings_reported"]:
        notes.append("engine_reports_no_timings")
    notes.extend(trial.get("timing_reasons") or [])
    trial["notes"] = notes
    trial["performance_valid"] = not reasons


def _median(values: list) -> float | None:
    return round(statistics.median(values), 6) if values else None


def _metric_medians(trials: list[dict], keys: list[str]) -> dict:
    return {
        f"median_{key}": _median(
            [trial[key] for trial in trials if numeric(trial.get(key))]
        )
        for key in keys
    }


# Throughput and latency numbers the harness calculates for itself.
TIMING_METRICS = [
    "ttft_s",
    "decode_window_s",
    "prefill_tps",
    "generation_tps",
    "mean_inter_token_latency_ms",
    "output_tps_whole_request",
    "request_tps_total",
]


def summarize_trials(trials: list[dict]) -> dict:
    successful = [trial for trial in trials if trial["ok"]]
    summary = {
        "repetitions": len(trials),
        "successful_requests": len(successful),
        "passed_trials": sum(bool(trial.get("passed")) for trial in trials),
        "pass_rate": (
            sum(bool(trial.get("passed")) for trial in trials) / len(trials)
            if trials
            else None
        ),
        "median_elapsed_s": _median([trial["elapsed_s"] for trial in successful]),
        "median_prompt_tokens": _median(
            [
                trial["prompt_tokens"]
                for trial in trials
                if numeric(trial.get("prompt_tokens"))
            ]
        ),
        "median_prompt_tps_engine": _median(
            [
                trial["prompt_tps_engine"]
                for trial in trials
                if numeric(trial.get("prompt_tps_engine"))
            ]
        ),
        "median_generation_tps_engine": _median(
            [
                trial["generation_tps_engine"]
                for trial in trials
                if numeric(trial.get("generation_tps_engine"))
            ]
        ),
    }
    summary.update(_metric_medians(successful, TIMING_METRICS))
    return summary


def _agreement_median(trials: list[dict], branch: str) -> float | None:
    values = []
    for trial in trials:
        agreement = (trial.get("engine_cross_check") or {}).get(branch)
        if isinstance(agreement, dict) and numeric(agreement.get("relative_error")):
            values.append(agreement["relative_error"])
    return _median(values)


def summarize_performance_trials(
    trials: list[dict],
    required_output_tokens: int,
    nominal_input_tokens: int,
    tolerance_percent: float = INPUT_SIZE_TOLERANCE_PERCENT,
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
    summary["input_size_tolerance_percent"] = tolerance_percent
    summary["all_trials_input_size_valid"] = bool(trials) and all(
        trial.get("input_size_valid", False) for trial in trials
    )
    summary["median_input_size_percent_from_nominal"] = _median(
        [
            trial["input_size_percent_from_nominal"]
            for trial in valid_trials
            if numeric(trial.get("input_size_percent_from_nominal"))
        ]
    )
    summary["all_trials_output_content_valid"] = bool(trials) and all(
        trial.get("output_content", {}).get("valid", False) for trial in trials
    )
    summary["stream_delivery"] = sorted(
        {
            trial["stream_delivery"]
            for trial in trials
            if trial.get("stream_delivery")
        }
    ) or None
    summary["all_trials_stream_incremental"] = bool(trials) and all(
        trial.get("stream_delivery") == "incremental" for trial in trials
    )
    summary["all_trials_no_prompt_cache"] = bool(trials) and not any(
        trial.get("prompt_cache_hit", False) for trial in trials
    )
    summary["prompt_cache_reported_trials"] = sum(
        1 for trial in trials if (trial.get("prompt_cache") or {}).get("reported")
    )
    summary["engine_timings_reported_trials"] = sum(
        1 for trial in trials if trial.get("engine_timings_reported")
    )
    summary["engine_token_counts_consistent_trials"] = sum(
        1
        for trial in trials
        if (trial.get("engine_cross_check") or {}).get("engine_token_counts_consistent")
    )
    summary["median_engine_vs_harness_prefill_relative_error"] = _agreement_median(
        valid_trials, "prefill_agreement"
    )
    summary["median_engine_vs_harness_generation_relative_error"] = _agreement_median(
        valid_trials, "generation_agreement"
    )
    summary["valid_performance_trials"] = len(valid_trials)
    summary["performance_valid"] = bool(trials) and len(valid_trials) == len(trials)
    reason_totals: dict[str, int] = {}
    for trial in trials:
        for reason in trial.get("invalid_reasons") or []:
            reason_totals[reason] = reason_totals.get(reason, 0) + 1
    summary["invalid_reason_totals"] = reason_totals
    acceptance_rates = [
        trial["draft_acceptance_rate"]
        for trial in valid_trials
        if numeric(trial.get("draft_acceptance_rate"))
    ]
    summary["median_draft_acceptance_rate"] = _median(acceptance_rates)
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


def _group_median(groups: list[dict], key: str) -> float | None:
    """Median of one load-group total across a case's counted groups."""
    return _median(
        [group[key] for group in groups if numeric(group.get(key))]
    )


def build_performance_table(suites: dict) -> list[dict]:
    """Flatten every performance case into one row per input tier.

    Throughput columns are calculated by the harness from token counts and its
    own clock, so they are comparable across inference engines.  Prefill uses
    time-to-first-token; generation excludes the first generated token.
    """
    rows = []
    for suite_name, suite in suites.items():
        if not isinstance(suite, dict):
            continue
        for case in suite.get("cases") or []:
            performance = case.get("performance")
            if not performance:
                continue
            summary = performance["summary"]
            groups = performance.get("groups") or []
            # Same rule as the medians: only groups where every request passed
            # every gate contribute, so a row never advertises a bad aggregate.
            counted_groups = [group for group in groups if group.get("group_valid")]
            rows.append(
                {
                    "suite": suite_name,
                    "case": case["id"],
                    "nominal_input_tokens": case.get("nominal_input_tokens"),
                    "concurrency": performance.get("concurrency") or 1,
                    "median_prompt_tokens": summary["median_prompt_tokens"],
                    "prefill_tps": summary["median_prefill_tps"],
                    "token_generation_tps": summary["median_generation_tps"],
                    "ttft_s": summary["median_ttft_s"],
                    "mean_inter_token_latency_ms": summary[
                        "median_mean_inter_token_latency_ms"
                    ],
                    "whole_request_output_tps": summary["median_output_tps_whole_request"],
                    "whole_request_s": summary["median_elapsed_s"],
                    # Totals for the requests a load group held in flight at
                    # once, taken over its valid groups.  At concurrency 1 a
                    # group is one request, so these equal its own totals.
                    "requests": (performance.get("concurrency") or 1),
                    "wall_s": _group_median(counted_groups, "wall_s"),
                    "prompt_tokens_total": _group_median(
                        counted_groups, "prompt_tokens_total"
                    ),
                    "output_tokens_total": _group_median(
                        counted_groups, "output_tokens_total"
                    ),
                    "tokens_total": _group_median(counted_groups, "tokens_total"),
                    "prefill_tps_total": _group_median(counted_groups, "prefill_tps_total"),
                    "token_generation_tps_total": _group_median(
                        counted_groups, "token_generation_tps_total"
                    ),
                    "generating_window_s": _group_median(
                        counted_groups, "generating_window_s"
                    ),
                    "decode_tps_total": _group_median(counted_groups, "decode_tps_total"),
                    "tokens_tps_total": _group_median(counted_groups, "tokens_tps_total"),
                    "requests_per_minute": _group_median(
                        counted_groups, "requests_per_minute"
                    ),
                    "ttft_max_s": _group_median(counted_groups, "ttft_max_s"),
                    "groups": len(groups),
                    "valid_groups": sum(1 for group in groups if group.get("group_valid")),
                    # Groups rejected only because a sampled stream went
                    # degenerate, so a rejected run is not read as a slow engine.
                    "content_only_failure_groups": sum(
                        1 for group in groups if group.get("content_only_failure")
                    ),
                    "engine_prefill_tps": summary["median_prompt_tps_engine"],
                    "engine_token_generation_tps": summary["median_generation_tps_engine"],
                    "draft_acceptance_rate": summary["median_draft_acceptance_rate"],
                    "trials": summary["repetitions"],
                    "valid_trials": summary["valid_performance_trials"],
                    "performance_valid": summary["performance_valid"],
                    "invalid_reason_totals": summary["invalid_reason_totals"],
                }
            )
    rows.sort(
        key=lambda row: (
            row["nominal_input_tokens"] or 0,
            row["concurrency"] or 1,
            row["suite"],
            row["case"],
        )
    )
    return rows


def format_performance_table(rows: list[dict]) -> str:
    """Render the performance results as two fixed-width tables plus a legend.

    The first table is the whole load: PP and TG summed over every request the
    group held in flight, so a concurrent run is read without multiplying or
    averaging anything.  The second keeps the per-request rates, which describe
    what a single user of that engine experiences.
    """
    totals = (
        ("suite", "suite", None),
        ("tier", "tier", None),
        ("requests", "req", None),
        ("wall_s", "wall s", ",.1f"),
        ("prompt_tokens_total", "prompt tok", ",.0f"),
        ("output_tokens_total", "out tok", ",.0f"),
        ("prefill_tps_total", "TOTAL PP t/s", ",.1f"),
        ("token_generation_tps_total", "TOTAL TG t/s", ",.1f"),
        ("tokens_tps_total", "all tok t/s", ",.1f"),
        ("requests_per_minute", "req/min", ",.2f"),
        ("performance_valid", "valid", None),
    )
    per_stream = (
        ("suite", "suite", None),
        ("tier", "tier", None),
        ("requests", "req", None),
        ("median_prompt_tokens", "prompt tok/req", ",.0f"),
        ("prefill_tps", "PP t/s", ",.1f"),
        ("token_generation_tps", "TG t/s", ",.1f"),
        ("ttft_s", "TTFT med s", ",.2f"),
        ("ttft_max_s", "TTFT max s", ",.2f"),
        ("mean_inter_token_latency_ms", "ITL ms", ",.2f"),
        ("whole_request_output_tps", "out t/s/req", ",.1f"),
        ("performance_valid", "valid", None),
    )
    return "\n\n".join(
        (
            _render_console_table(
                rows, totals, "WHOLE LOAD - every request in the group added together"
            ),
            _render_console_table(
                rows,
                per_stream,
                "ONE STREAM - what a single request experiences (median of the group)",
            ),
            "\n".join(METRIC_LEGEND),
        )
    )


# Spelled out under every run, because a rate is meaningless without the
# denominator it was divided by.
METRIC_LEGEND = (
    "What the columns mean",
    "  wall s        first request sent until the last response finished",
    "  prompt tok    prompt tokens of all requests added together (req x prompt tokens each)",
    "  out tok       generated tokens of all requests added together",
    "  TOTAL PP t/s  all prompt tokens / time until the SLOWEST stream got its first token",
    "                = how fast the engine digested the whole batch of prompts",
    "  TOTAL TG t/s  all generated tokens (minus each stream's first) / time the group spent generating",
    "                = what the engine generated in total, NOT one stream's rate times req",
    "  all tok t/s   prompt + generated tokens / wall s = overall token throughput of the window",
    "  PP, TG        same definitions as llama.cpp prompt_per_second and predicted_per_second",
    "  req/min       completed requests per minute at this concurrency",
)


def _render_console_table(rows: list[dict], columns: tuple, title: str) -> str:
    """Render rows as a titled, fixed-width table; missing values print as n/a."""
    grid = []
    for row in rows:
        tier = (
            f"{row['nominal_input_tokens'] // 1024}k"
            if numeric(row.get("nominal_input_tokens"))
            else "n/a"
        )
        cells = []
        for key, _header, spec in columns:
            value = tier if key == "tier" else row.get(key)
            if key == "performance_valid":
                cells.append("yes" if value else "no")
            elif value is None:
                cells.append("n/a")
            elif spec and isinstance(value, float):
                cells.append(format(value, spec))
            elif spec and isinstance(value, int):
                cells.append(format(float(value), spec))
            else:
                cells.append(str(value))
        grid.append(cells)
    headers = [header for _key, header, _spec in columns]
    widths = [
        max(len(header), *(len(cells[index]) for cells in grid)) if grid else len(header)
        for index, header in enumerate(headers)
    ]
    # The suite name reads better left-aligned; everything else is numeric.
    aligns = ("<", *(">" * (len(headers) - 1)))
    lines = [
        title,
        "  ".join(
            format(header, f"{align}{width}")
            for header, width, align in zip(headers, widths, aligns)
        ),
    ]
    lines.append("-" * len(lines[1]))
    for cells in grid:
        lines.append(
            "  ".join(
                format(value, f"{align}{width}")
                for value, width, align in zip(cells, widths, aligns)
            )
        )
    return "\n".join(lines)


def run_suite(
    suite_name: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    repetitions_override: int | None,
    lane: str,
    sizes: list[str],
    chat_template_kwargs: dict | None = None,
    checkpoint_callback=None,
    engine: str = "llama.cpp",
    stream: bool = True,
    input_size_tolerance_percent: float = INPUT_SIZE_TOLERANCE_PERCENT,
    max_retries: int = 0,
    concurrency: int = 1,
    request_tag_scope: str = "",
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
        "engine": engine,
        "measurement": {
            "performance_stream": stream,
            "input_size_tolerance_percent": input_size_tolerance_percent,
            "concurrency": concurrency,
            "metric_sources": engine_api.metric_sources(streamed=stream, engine=engine),
        },
        "status": "running",
        "cases": cases,
        "summary": summarize_suite_cases(cases),
    }

    def checkpoint(event: str) -> None:
        suite_result["last_event"] = event
        suite_result["summary"] = summarize_suite_cases(cases)
        if checkpoint_callback is not None:
            checkpoint_callback(suite_result, event)

    def request_tag(*parts: str) -> str:
        """Tag a request so no other request shares its prompt prefix.

        The scope changes once per run, which matters more than it looks: an
        engine with automatic prefix caching keys its KV blocks on the leading
        tokens, so a tag that repeats between runs lets a re-run start from a
        warm prefill and report several times the real prefill speed.
        """
        return "-".join([part for part in (request_tag_scope, *parts) if part])

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
                    request_tag=request_tag(case["id"], "quality", f"{index:02d}"),
                    chat_template_kwargs=chat_template_kwargs,
                    engine=engine,
                    max_retries=max_retries,
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
                "stream": stream,
                "concurrency": concurrency,
                "cold_cache_guards": ["unique-request-tag"]
                + (["run-scoped-request-tag"] if request_tag_scope else [])
                + (["cache_prompt=false"] if engine == "llama.cpp" else []),
                "input_chars": len(performance_prompt),
                "input_sha256": sha256_bytes(performance_prompt.encode("utf-8")),
                "trials": [],
                "groups": [],
                "summary": summarize_performance_trials(
                    [],
                    required_tokens,
                    int(case["nominal_input_tokens"]),
                    input_size_tolerance_percent,
                ),
            }
            performance_trials = case_result["performance"]["trials"]
            performance_groups = case_result["performance"]["groups"]
            for index in range(1, repetitions + 1):
                group_id = f"{case['id']}-performance-{index:02d}"

                def request(request_index: int, repetition: int = index) -> dict:
                    # Each copy carries its own request tag, so N simultaneous
                    # identical prompts cannot merge into one shared prefix cache.
                    tag = request_tag(
                        case["id"],
                        "performance",
                        f"{repetition:02d}"
                        if concurrency == 1
                        else f"{repetition:02d}-{request_index:02d}",
                    )
                    trial = chat_once(
                        base_url,
                        api_key,
                        model,
                        suite["performance_system_prompt"],
                        performance_prompt,
                        required_tokens,
                        suite["performance_sampling"],
                        timeout,
                        request_tag=tag,
                        chat_template_kwargs=chat_template_kwargs,
                        engine=engine,
                        stream=stream,
                        max_retries=max_retries,
                    )
                    evaluate_performance_trial(
                        trial,
                        required_output_tokens=required_tokens,
                        nominal_input_tokens=int(case["nominal_input_tokens"]),
                        tolerance_percent=input_size_tolerance_percent,
                    )
                    return trial

                group_trials, group_span = run_concurrent_group(
                    request,
                    concurrency=concurrency,
                    engine=engine,
                    stream=stream,
                )
                for request_index, trial in enumerate(group_trials, start=1):
                    trial["request_index"] = request_index
                    trial["group_id"] = group_id
                performance_groups.append(
                    summarize_load_group(
                        group_trials,
                        group_id=group_id,
                        concurrency=concurrency,
                        wall_s=group_span["wall_s"],
                        barrier_broken=group_span["barrier_broken"],
                    )
                )
                performance_trials.extend(group_trials)
                case_result["performance"]["summary"] = summarize_performance_trials(
                    performance_trials,
                    required_tokens,
                    int(case["nominal_input_tokens"]),
                    input_size_tolerance_percent,
                )
                checkpoint(f"{case['id']}:performance:group-{index:02d}")
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
        choices=("8k", "16k", "32k", "64k", "128k", "all"),
        default=["all"],
        help="Run a list of long-input sizes, or 'all'",
    )
    parser.add_argument("--api-key", default=os.environ.get("LLM_BENCH_API_KEY", ""))
    parser.add_argument(
        "--chat-template-kwargs",
        default="",
        help="JSON object passed to OpenAI-compatible chat endpoints, e.g. '{\"enable_thinking\": false}'",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--engine",
        choices=engine_api.ENGINE_CHOICES,
        default=engine_api.AUTO_ENGINE,
        help="Server family. 'auto' probes engine-specific endpoints and otherwise "
        "falls back to the strict OpenAI-compatible payload",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=range(1, MAX_CONCURRENCY + 1),
        default=1,
        metavar=f"1-{MAX_CONCURRENCY}",
        help=f"Hold this many identical performance requests in flight at once. "
        f"Reports per-request rates plus aggregate throughput over the window they "
        f"shared; the quality lane always stays sequential and a concurrent run is "
        f"marked non-canonical",
    )
    parser.add_argument(
        "--request-tag-scope",
        default="",
        help="Prefix added to every request tag so re-runs cannot be served from a "
        "warm prefix cache. Defaults to the run start time and is recorded in the "
        "result; set it to a fixed value to repeat a previous run's exact prompts",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable SSE in the performance lane. Without streaming, prefill and "
        "decode cannot be separated and only the whole-request rate is reported",
    )
    parser.add_argument(
        "--input-size-tolerance-percent",
        type=float,
        default=INPUT_SIZE_TOLERANCE_PERCENT,
        help="Allowed deviation between server-reported prompt_tokens and the "
        "requested tier. Widen it for engines whose tokenizer or chat template "
        "differs from the pinned calibration tokenizer",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Extra attempts for a request that failed before producing any output "
        "(rate limits, connection resets)",
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Full server launch command including environment variables, e.g. "
        "'llama-server ...' or 'vllm serve ...'; saved unchanged",
    )
    parser.add_argument(
        "--system",
        required=True,
        help="Free-form system description: cards, OS, ROCm/CUDA, engine and version",
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
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.input_size_tolerance_percent < 0:
        parser.error("--input-size-tolerance-percent cannot be negative")
    if "all" in args.sizes and len(args.sizes) > 1:
        parser.error("--sizes all cannot be combined with individual sizes")
    if args.chat_template_kwargs:
        try:
            args.chat_template_kwargs = json.loads(args.chat_template_kwargs)
        except json.JSONDecodeError as exc:
            parser.error(f"--chat-template-kwargs must be valid JSON: {exc.msg}")
        if not isinstance(args.chat_template_kwargs, dict):
            parser.error("--chat-template-kwargs must be a JSON object")
    else:
        args.chat_template_kwargs = None
    args.sizes = list(dict.fromkeys(args.sizes))
    return args


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    suite_names = list(SUITES) if args.suite == "all" else [args.suite]
    streamed = not args.no_stream
    started_at = utc_now()
    # A per-run tag scope matters more than it looks: an engine with automatic
    # prefix caching keys its KV blocks on the leading prompt tokens, so a tag
    # that repeats between runs lets a re-run start from a warm prefill and
    # report several times the real prefill speed.  The value is recorded, so
    # the exact prompts of a past run stay reconstructible.
    digits_only = re.compile(r"\D")
    tag_scope = args.request_tag_scope or (
        f"{digits_only.sub('', started_at)}-{int(time.time() * 1000) % 1000:03d}"
    )
    result = {
        "benchmark_schema_version": 8,
        "tool": {"name": "llm-context-bench", "version": __version__},
        "status": "running",
        "profile": args.profile,
        "model": args.model,
        "base_url": base_url,
        "started_at": started_at,
        "lane": args.lane,
        "sizes": args.sizes,
        "canonical": is_canonical(args),
        "harness_sha256": harness_fingerprint(),
        "engine": {"requested": args.engine, "resolved": None},
        "measurement": {
            "performance_stream": streamed,
            "input_size_tolerance_percent": args.input_size_tolerance_percent,
            "max_retries": args.max_retries,
            "concurrency": args.concurrency,
            "request_tag_scope": tag_scope,
            "concurrency_definition": "identical performance requests released together; "
            "*_total rates divide the group's token counts by the group's wall time",
            "definitions": {
                "prefill_tps": "prompt_tokens / time from request send to the first "
                "token-bearing stream chunk",
                "token_generation_tps": "(completion_tokens - 1) / (last token arrival "
                "- first token arrival); the first token belongs to the prefill",
                "ttft_s": "request send to first token-bearing stream chunk",
                "whole_request_output_tps": "completion_tokens / whole request wall-clock",
                "mean_inter_token_latency_ms": "decode window / (completion_tokens - 1)",
            },
        },
        "run_parameters": build_run_parameters(args, sys.argv),
        "environment": {},
        "suites": {},
        "performance_table": [],
        "checkpoint_count": 0,
    }

    def checkpoint(event: str) -> None:
        result["checkpoint_count"] += 1
        result["last_checkpoint"] = {"event": event, "at": utc_now()}
        result["performance_table"] = build_performance_table(result["suites"])
        write_json_atomic(args.output, result)

    checkpoint("benchmark:initialized")
    try:
        health_elapsed, health = request_json(base_url, "/health", None, args.api_key, 20)
        result["environment"]["health_elapsed_s"] = round(health_elapsed, 6)
        result["environment"]["health"] = health
        checkpoint("environment:health")

        detection = engine_api.detect_engine(
            lambda path: request_json(base_url, path, None, args.api_key, 10),
            args.engine,
        )
        resolved_engine = detection["resolved"]
        result["environment"]["engine_detection"] = detection
        result["engine"] = {
            "requested": detection["requested"],
            "resolved": resolved_engine,
            "detection_method": detection["method"],
            "engine_only_params": detection["engine_only_params"],
            "refined_from_response": None,
        }
        result["measurement"]["metric_sources"] = engine_api.metric_sources(
            streamed=streamed, engine=resolved_engine
        )
        checkpoint("environment:engine")

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
                chat_template_kwargs=args.chat_template_kwargs,
                engine=resolved_engine,
                max_retries=args.max_retries,
            )
            warmup["nominal_input_tokens"] = MODEL_LOAD_WARMUP_NOMINAL_INPUT_TOKENS
            warmup["input_chars"] = len(warmup_prompt)
            warmup["input_sha256"] = sha256_bytes(warmup_prompt.encode("utf-8"))
            warmup["required_output_tokens"] = MODEL_LOAD_WARMUP_OUTPUT_TOKENS
            # Without ignore_eos a server is free to stop early, so the fixed
            # length is a hard requirement only when the flag was accepted.
            warmup["fixed_length_check_applies"] = warmup["ignore_eos_applied"]
            warmup["fixed_length_valid"] = (
                warmup["ok"]
                and warmup["produced_tokens"] == MODEL_LOAD_WARMUP_OUTPUT_TOKENS
            )
            result["environment"]["warmup"] = warmup
            # The warm-up itself is sent with the pre-refinement profile; it is
            # record-only, and the measured lanes then use the full llama.cpp
            # payload.
            refined = engine_api.refine_engine(resolved_engine, {"timings": warmup["timings"]})
            if refined:
                resolved_engine = refined
                result["engine"].update(
                    {
                        "resolved": refined,
                        "detection_method": "response-shape",
                        "refined_from_response": "per-response timings object",
                        "engine_only_params": list(
                            engine_api.profile(refined)["engine_only_params"]
                        ),
                    }
                )
                result["measurement"]["metric_sources"] = engine_api.metric_sources(
                    streamed=streamed, engine=resolved_engine
                )
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
                chat_template_kwargs=args.chat_template_kwargs,
                checkpoint_callback=suite_checkpoint,
                engine=resolved_engine,
                stream=streamed,
                input_size_tolerance_percent=args.input_size_tolerance_percent,
                max_retries=args.max_retries,
                concurrency=args.concurrency,
                request_tag_scope=tag_scope,
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
    table = build_performance_table(result["suites"])
    print(
        json.dumps(
            {
                "profile": args.profile,
                "canonical": result["canonical"],
                "engine": result["engine"]["resolved"],
                "lane": args.lane,
                "suites": compact,
                "performance": table,
            },
            indent=2,
        )
    )
    if table:
        print()
        print(format_performance_table(table))


if __name__ == "__main__":
    main()
