"""Engine-facing primitives: profiles, payloads, SSE capture, and timing math.

Every performance metric this benchmark reports is calculated by the harness
from its own wall-clock timestamps and token counts, so a llama.cpp server, a
vLLM/SGLang server, and a hosted OpenAI-compatible endpoint are measured the
same way.  Engine-reported counters (the llama.cpp ``timings`` object) are
parsed only as an optional cross-check of the harness numbers; they are never
required and never the source of a headline rate.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request

STATUS_MARKER = "__LLM_CONTEXT_BENCH_STATUS__:"
AUTO_ENGINE = "auto"

# Sampling/control keys that exist outside the OpenAI chat-completions
# contract.  Each profile lists the subset its server accepts; everything else
# is dropped and recorded, so one locked workload can be sent anywhere without
# an HTTP 400 from strict servers.
ENGINE_ONLY_PARAMS = ("cache_prompt", "ignore_eos", "top_k", "min_p")

# Probed with GET; a JSON body carrying any marker key identifies the engine.
ENGINE_PROFILES: dict[str, dict] = {
    "llama.cpp": {
        "engine_only_params": ("cache_prompt", "ignore_eos", "top_k", "min_p"),
        "signatures": (("/props", ("total_slots", "build_info", "model_path", "use_mlock", "webui")),),
    },
    "vllm": {
        "engine_only_params": ("ignore_eos", "top_k", "min_p"),
        "signatures": (("/server_info", ("vllm_config", "model_config", "cache_config", "parallel_config", "num_gpu_blocks")),),
    },
    "sglang": {
        "engine_only_params": ("top_k", "min_p"),
        "signatures": (("/get_model_info", ("model_path", "is_generation", "tp_size", "dp_size")),),
    },
    # Strict OpenAI-compatible surface only: no engine extras are sent.
    "openai": {"engine_only_params": (), "signatures": ()},
    # Unknown server: safest payload that still works on local engines.
    "generic": {"engine_only_params": (), "signatures": ()},
}
ENGINE_CHOICES = (AUTO_ENGINE, *ENGINE_PROFILES)

# Secondary identification: OpenAI-compatible servers usually label the model
# list with their own name.  Only used when no engine-specific endpoint answered.
OWNERSHIP_MARKERS = (
    ("vllm", "vllm"),
    ("sglang", "sglang"),
    ("llama.cpp", "llama.cpp"),
    ("llama.cpp.cpp", "llama.cpp"),
    ("llama-cpp", "llama.cpp"),
)

# Locations engines use to report prompt-cache reuse, in probe order.
CACHE_TOKEN_FIELDS = (
    ("timings", "cache_n"),
    ("usage_prompt_tokens_details", "cached_tokens"),
    ("usage", "cached_tokens"),
    ("usage", "cache_read_input_tokens"),
    ("usage", "prompt_cache_hit_tokens"),
)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def profile(engine: str) -> dict:
    try:
        return ENGINE_PROFILES[engine]
    except KeyError as exc:
        raise ValueError(f"unknown engine profile: {engine}") from exc


def detect_engine_from_ownership(body) -> str | None:
    """Read an engine name out of a /v1/models response, when it carries one."""
    if not isinstance(body, dict) or body.get("http_status") != 200:
        return None
    for entry in body.get("data") or []:
        if not isinstance(entry, dict):
            continue
        owned_by = str(entry.get("owned_by") or "").lower()
        for marker, name in OWNERSHIP_MARKERS:
            if marker and marker in owned_by:
                return name
    return None


def detect_engine(probe, requested: str = AUTO_ENGINE) -> dict:
    """Identify the server from engine-specific endpoints.

    ``probe(path)`` returns ``(elapsed_s, body)`` like ``request_json``.  A
    request failure is simply a non-match, so an unrecognised server resolves
    to the strictest useful profile (``generic``) instead of failing the run.
    """
    if requested != AUTO_ENGINE:
        return {
            "requested": requested,
            "resolved": requested,
            "method": "cli",
            "probes": [],
            "engine_only_params": list(profile(requested)["engine_only_params"]),
        }
    probes: list[dict] = []
    resolved = "generic"
    method = "endpoint-signature"
    for name, candidate in ENGINE_PROFILES.items():
        for path, markers in candidate["signatures"]:
            try:
                elapsed, body = probe(path)
            except Exception as exc:  # A missing endpoint is a non-match.
                probes.append({"path": path, "matched": False, "error": f"{type(exc).__name__}: {exc}"})
                continue
            keys = sorted(body) if isinstance(body, dict) else []
            status = body.get("http_status") if isinstance(body, dict) else None
            hits = [marker for marker in markers if isinstance(body, dict) and marker in body]
            matched = bool(hits) and status == 200
            probes.append(
                {
                    "path": path,
                    "engine": name,
                    "http_status": status,
                    "matched": matched,
                    "matched_markers": hits,
                    "elapsed_s": round(elapsed, 6),
                    "response_keys": keys[:12],
                }
            )
            if matched:
                resolved = name
                break
        if resolved != "generic":
            break
    if resolved == "generic":
        # No dedicated endpoint answered (proxied, restricted, or an older
        # build).  The model list often names the engine anyway.
        try:
            elapsed, body = probe("/v1/models")
        except Exception as exc:
            probes.append({"path": "/v1/models", "matched": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            owned = detect_engine_from_ownership(body)
            probes.append(
                {
                    "path": "/v1/models",
                    "engine": owned,
                    "http_status": body.get("http_status") if isinstance(body, dict) else None,
                    "matched": bool(owned),
                    "matched_markers": ["owned_by"] if owned else [],
                    "elapsed_s": round(elapsed, 6),
                    "response_keys": sorted(body)[:12] if isinstance(body, dict) else [],
                }
            )
            if owned:
                resolved = owned
                method = "models-owned-by"
    return {
        "requested": AUTO_ENGINE,
        "resolved": resolved,
        "method": method,
        "probes": probes,
        "engine_only_params": list(ENGINE_PROFILES[resolved]["engine_only_params"]),
    }


def refine_engine(resolved: str, body: dict) -> str | None:
    """Promote an unidentified server once a response proves what it is.

    Only ``generic`` is promoted, and only to ``llama.cpp``, which is the one
    engine that returns a per-response ``timings`` object.  This recovers full
    payload fidelity when a signature endpoint is unavailable (proxied,
    read-only, or older builds).
    """
    if resolved != "generic" or not isinstance(body, dict):
        return None
    timings = body.get("timings")
    if isinstance(timings, dict) and "predicted_n" in timings and "prompt_n" in timings:
        return "llama.cpp"
    return None


def build_chat_payload(
    *,
    engine: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    sampling: dict,
    stream: bool,
    ignore_eos: bool = False,
    chat_template_kwargs: dict | None = None,
) -> tuple[dict, list[str]]:
    """Build one chat request and report the engine extras that were dropped."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "seed": sampling["seed"],
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": stream,
    }
    if stream:
        # Ask the engine to append its authoritative usage block to the stream.
        payload["stream_options"] = {"include_usage": True}
    extras: dict = {key: sampling[key] for key in ENGINE_ONLY_PARAMS if key in sampling}
    if ignore_eos:
        extras["ignore_eos"] = True
    if engine == "llama.cpp":
        # llama.cpp-specific cache guard.  On every other engine the unique
        # per-request tag prefix is what keeps the prefill cold.
        extras["cache_prompt"] = False
    allowed = set(profile(engine)["engine_only_params"])
    dropped = []
    # Iterate the canonical order so an engine's payload key order is stable.
    for key in ENGINE_ONLY_PARAMS:
        if key not in extras:
            continue
        if key in allowed:
            payload[key] = extras[key]
        else:
            dropped.append(key)
    # A caller-supplied chat-template override is an explicit instruction and
    # is never silently removed, even for a strict profile.
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    return payload, sorted(dropped)


def empty_stream_measurement(started: float | None = None, error: str | None = None, http_status=None, clock=time.perf_counter) -> dict:
    elapsed = clock() - started if started is not None else 0.0
    return {
        "ok": False,
        "error": error,
        "http_status": http_status,
        "elapsed_s": elapsed,
        "response": "",
        "reasoning_response": "",
        "usage": {},
        "timings": {},
        "finish_reason": None,
        "first_token_at": None,
        "last_token_at": None,
        "stream_chunks": 0,
        "content_chars": 0,
        "done_at": None,
        "parse_errors": 0,
        "stream_truncated": False,
    }


def consume_sse(lines, started: float, clock=time.perf_counter) -> dict:
    """Read an SSE chat stream, stamping the arrival of every token-bearing chunk.

    Timestamps are relative to ``started`` and measured with ``clock`` (a
    ``time.perf_counter``-style monotonic clock).  Only chunks that actually
    carry text start or extend the decode window, so the empty role-only delta
    that servers send before the first token cannot shorten TTFT.
    """
    measured = empty_stream_measurement(started=started, clock=clock)
    measured["ok"] = True
    measured["error"] = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    note = None
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line:
            continue
        if line.startswith(STATUS_MARKER):
            status = line[len(STATUS_MARKER):].strip()
            try:
                measured["http_status"] = int(status)
            except ValueError:
                pass
            continue
        if not line.startswith("data:"):
            # SSE comments (": ping"), and event/id/retry fields are normal
            # traffic.  Only an unexpected body line is worth keeping as a
            # diagnostic for a stream that produced no tokens.
            if note is None and not line.startswith((":", "event:", "id:", "retry:")):
                note = line[:400]
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            measured["done_at"] = clock() - started
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            measured["parse_errors"] += 1
            continue
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("usage"), dict) and event["usage"]:
            measured["usage"] = event["usage"]
        if isinstance(event.get("timings"), dict) and event["timings"]:
            measured["timings"] = event["timings"]
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                measured["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {"content": choice.get("text") or ""}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            # A malformed field must not abort a benchmark run mid-stream.
            content = content if isinstance(content, str) else ""
            reasoning = reasoning if isinstance(reasoning, str) else ""
            if not content and not reasoning:
                continue
            arrived = clock() - started
            if measured["first_token_at"] is None:
                measured["first_token_at"] = arrived
            measured["last_token_at"] = arrived
            measured["stream_chunks"] += 1
            measured["content_chars"] += len(content)
            if content:
                text_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
    measured["elapsed_s"] = clock() - started
    if measured["done_at"] is None:
        # No [DONE] sentinel: either the transport ended or an error body was
        # returned in place of a stream.  Keep the first unparsed line so a
        # failure is diagnosable from the result JSON alone.
        measured["stream_truncated"] = True
    if measured["stream_chunks"] == 0 and not measured["error"]:
        measured["error"] = note or "stream contained no completion chunks"
    measured["ok"] = measured["error"] is None and measured["stream_chunks"] > 0
    measured["response"] = "".join(text_parts)
    measured["reasoning_response"] = "".join(reasoning_parts)
    return measured


def request_sse(
    base_url: str,
    path: str,
    payload: dict,
    api_key: str,
    timeout: int,
    client: str = "urllib",
) -> dict:
    """POST a streaming chat request and return the measured stream."""
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    started = time.perf_counter()
    if client == "curl":
        command = [
            "curl",
            "-N",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout),
            "--connect-timeout",
            str(min(timeout, 15)),
            "-H",
            "Content-Type: application/json",
            "-H",
            "Accept: text/event-stream",
            "-w",
            f"\n{STATUS_MARKER}%{{http_code}}",
        ]
        if api_key:
            command.extend(["-H", f"Authorization: Bearer {api_key}"])
        command.extend(["--data-binary", "@-", url])
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            return empty_stream_measurement(started, error=f"curl launch failed: {exc}")
        measured = None
        failure = None
        try:
            process.stdin.write(data)
            process.stdin.close()
            measured = consume_sse(process.stdout, started)
            if measured["http_status"] is None:
                # curl writes the -w status marker after the body, while
                # consume_sse stops reading at [DONE].  Pick the marker up so a
                # successful stream still records its status code.
                for tail in process.stdout:
                    text = tail.decode("utf-8", "replace").strip()
                    if text.startswith(STATUS_MARKER):
                        try:
                            measured["http_status"] = int(text[len(STATUS_MARKER):])
                        except ValueError:
                            pass
                        break
            process.wait(timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            failure = f"curl stream failed: {exc}"
        finally:
            stderr_text = b""
            if process.stderr:
                try:
                    stderr_text = process.stderr.read()
                except (OSError, ValueError):
                    stderr_text = b""
                process.stderr.close()
            if process.stdout:
                try:
                    process.stdout.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.kill()
        if measured is None:
            return empty_stream_measurement(started, error=failure)
        if failure:
            measured["error"] = failure
            measured["ok"] = False
        elif process.returncode and measured["stream_chunks"] == 0:
            measured["error"] = (
                measured["error"]
                or stderr_text.decode("utf-8", "replace").strip()
                or f"curl exited {process.returncode}"
            )
            measured["ok"] = False
        return measured

    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            # Bench requests are independent; do not wait on a keep-alive socket.
            "Connection": "close",
        },
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            measured = consume_sse(response, started)
            measured["http_status"] = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
            message = json.dumps(body)[:400]
        except (json.JSONDecodeError, TypeError):
            message = raw[:400]
        measured = empty_stream_measurement(started, error=message or str(exc), http_status=exc.code)
    except Exception as exc:  # Network errors are data, not harness crashes.
        measured = empty_stream_measurement(started, error=f"{type(exc).__name__}: {exc}")
    return measured


def token_counts(measured: dict) -> dict:
    """Resolve prompt/completion token counts from whatever the engine reported."""
    usage = measured.get("usage") or {}
    timings = measured.get("timings") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    source = "usage" if numeric(completion_tokens) else None
    if not numeric(completion_tokens) and numeric(timings.get("predicted_n")):
        completion_tokens = timings["predicted_n"]
        source = "engine-timings"
    if not numeric(prompt_tokens):
        prompt_tokens = timings.get("prompt_n")
        if numeric(prompt_tokens):
            source = source or "engine-timings"
    if not numeric(completion_tokens):
        # Last resort: one SSE event per token is the common case, but it is an
        # estimate, so it is labelled as one instead of passed off as a count.
        completion_tokens = measured.get("stream_chunks")
        source = "stream-chunk-estimate"
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "token_count_source": source,
    }


def detect_cached_tokens(measured: dict) -> dict:
    """Find prompt-cache reuse through whichever field this engine exposes."""
    usage = measured.get("usage") or {}
    timings = measured.get("timings") or {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    lookups = {
        "timings": timings,
        "usage": usage,
        "usage_prompt_tokens_details": details,
    }
    for container, field in CACHE_TOKEN_FIELDS:
        value = lookups[container].get(field)
        if numeric(value):
            return {
                "cached_tokens": value,
                "reported_by": f"{container}.{field}",
                "reported": True,
                "no_prompt_cache": value == 0,
            }
    return {
        "cached_tokens": None,
        "reported_by": None,
        "reported": False,
        # Unreported is not the same as unused.  The per-request tag prefix
        # keeps the prefill cold; the result says the engine cannot confirm it.
        "no_prompt_cache": None,
    }


def compute_timing_metrics(
    *,
    elapsed_s: float,
    prompt_tokens,
    completion_tokens,
    first_token_at,
    last_token_at,
    stream_chunks: int,
    streamed: bool,
) -> dict:
    """Derive prefill and decode throughput from harness-observed numbers.

    Definitions, held identical across engines:
      * ``ttft_s``          — request sent to first token-bearing stream chunk.
      * ``prefill_tps``     — ``prompt_tokens / ttft_s``.  Covers queueing,
                              network, prefill, and the first sampled token.
      * ``decode_window_s`` — first token arrival to last token arrival.
      * ``generation_tps``  — ``(completion_tokens - 1) / decode_window_s``.
                              The first token is excluded because it comes from
                              the prefill's final logits, matching the way
                              llama.cpp computes ``predicted_per_second``.
    """

    def rounded(value):
        return round(value, 6) if isinstance(value, float) else value

    metrics = {
        "streamed": streamed,
        "ttft_s": None,
        "decode_window_s": None,
        "prefill_tps": None,
        "generation_tps": None,
        "mean_inter_token_latency_ms": None,
        "output_tps_whole_request": None,
        "request_tps_total": None,
        "tokens_per_stream_chunk": None,
        "stream_delivery": "not-streamed" if not streamed else "unobserved",
        "timing_reasons": [] if streamed else ["streaming_disabled"],
    }
    if elapsed_s and numeric(completion_tokens):
        metrics["output_tps_whole_request"] = completion_tokens / elapsed_s
    if elapsed_s and numeric(completion_tokens) and numeric(prompt_tokens):
        metrics["request_tps_total"] = (prompt_tokens + completion_tokens) / elapsed_s
    if not streamed:
        return {key: rounded(value) for key, value in metrics.items()}
    if not numeric(first_token_at) or not numeric(last_token_at):
        metrics["timing_reasons"].append("no_stream_token_timestamps")
        return {key: rounded(value) for key, value in metrics.items()}
    metrics["ttft_s"] = first_token_at
    metrics["decode_window_s"] = max(last_token_at - first_token_at, 0.0)
    if numeric(prompt_tokens) and first_token_at > 0:
        metrics["prefill_tps"] = prompt_tokens / first_token_at
    elif numeric(prompt_tokens):
        metrics["timing_reasons"].append("non_positive_ttft")
    generated_after_first = completion_tokens - 1 if numeric(completion_tokens) else None
    if numeric(stream_chunks) and numeric(completion_tokens) and stream_chunks > 0:
        metrics["tokens_per_stream_chunk"] = completion_tokens / stream_chunks
    if stream_chunks <= 0:
        metrics["stream_delivery"] = "unobserved"
        metrics["timing_reasons"].append("no_stream_chunks")
    elif stream_chunks == 1:
        # A single flush means a proxy or engine buffered the whole response;
        # the decode window would be meaningless.
        metrics["stream_delivery"] = "buffered"
        metrics["timing_reasons"].append("stream_delivered_in_one_chunk")
    else:
        metrics["stream_delivery"] = "incremental"
    if generated_after_first and generated_after_first > 0 and metrics["decode_window_s"] > 0:
        metrics["generation_tps"] = generated_after_first / metrics["decode_window_s"]
        metrics["mean_inter_token_latency_ms"] = (
            metrics["decode_window_s"] * 1000.0 / generated_after_first
        )
    elif generated_after_first is not None:
        metrics["timing_reasons"].append("too_few_generated_tokens")
    return {key: rounded(value) for key, value in metrics.items()}


def cross_check_engine_timings(trial: dict) -> dict:
    """Compare harness rates with engine-reported rates when the engine has any.

    Advisory only: a missing ``timings`` object (every engine except
    llama.cpp) is reported as ``reported: false`` and never invalidates a
    trial.  When counters are present, this catches a broken proxy clock, a
    mis-instrumented server build, or silently cached prompts.
    """
    timings = trial.get("timings") or {}
    prompt_n = timings.get("prompt_n")
    prompt_ms = timings.get("prompt_ms")
    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    reported = {
        "prompt_tps_engine": timings.get("prompt_per_second"),
        "generation_tps_engine": timings.get("predicted_per_second"),
        "prompt_ms_engine": prompt_ms,
        "predicted_ms_engine": predicted_ms,
        "prompt_n_engine": prompt_n,
        "predicted_n_engine": predicted_n,
    }
    result = {
        "reported": bool(timings),
        "engine_prompt_tokens": prompt_n,
        "engine_generated_tokens": predicted_n,
        **reported,
        "engine_prompt_tps_recalculated": None,
        "engine_generation_tps_recalculated": None,
        # Does the engine's own advertised rate match its own counters?
        "engine_reported_prompt_rate_error": None,
        "engine_reported_generation_rate_error": None,
        "prefill_agreement": None,
        "generation_agreement": None,
        "engine_token_counts_consistent": None,
        "notes": [] if timings else ["engine_reports_no_timings"],
    }

    def relative_error(actual, expected):
        if not numeric(actual) or not numeric(expected) or expected == 0:
            return None
        return round(abs(actual - expected) / abs(expected), 6)

    if numeric(prompt_n) and numeric(prompt_ms) and prompt_ms > 0:
        result["engine_prompt_tps_recalculated"] = prompt_n * 1000.0 / prompt_ms
        result["engine_reported_prompt_rate_error"] = relative_error(
            reported["prompt_tps_engine"], result["engine_prompt_tps_recalculated"]
        )
    if numeric(predicted_n) and numeric(predicted_ms) and predicted_ms > 0:
        result["engine_generation_tps_recalculated"] = max(predicted_n - 1, 0) * 1000.0 / predicted_ms
        result["engine_reported_generation_rate_error"] = relative_error(
            reported["generation_tps_engine"], result["engine_generation_tps_recalculated"]
        )

    def agreement(harness_value, engine_value):
        if not numeric(harness_value) or not numeric(engine_value) or engine_value == 0:
            return None
        return {
            "harness": harness_value,
            "engine": engine_value,
            "relative_error": round(abs(harness_value - engine_value) / abs(engine_value), 6),
        }

    result["prefill_agreement"] = agreement(trial.get("prefill_tps"), result["engine_prompt_tps_recalculated"])
    result["generation_agreement"] = agreement(trial.get("generation_tps"), result["engine_generation_tps_recalculated"])
    prompt_usage = trial.get("prompt_tokens")
    completion_usage = trial.get("completion_tokens")
    if numeric(prompt_usage) and numeric(completion_usage) and numeric(prompt_n) and numeric(predicted_n):
        cached = (trial.get("prompt_cache") or {}).get("cached_tokens")
        cached = cached if numeric(cached) else 0
        result["engine_token_counts_consistent"] = (
            prompt_usage == prompt_n + cached and completion_usage == predicted_n
        )
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in result.items()
    }


def metric_sources(*, streamed: bool, engine: str) -> dict:
    """Declare, per metric, whether the harness measured it or nobody did."""
    return {
        "prefill_tps": "harness-wall-clock(prompt_tokens/ttft)" if streamed else "unavailable(streaming-required)",
        "generation_tps": "harness-wall-clock((completion_tokens-1)/decode_window)" if streamed else "unavailable(streaming-required)",
        "ttft_s": "harness-wall-clock" if streamed else "unavailable(streaming-required)",
        "output_tps_whole_request": "harness-wall-clock",
        "token_counts": "engine-usage-when-reported",
        "engine_reported_timings": "cross-check-only" if engine == "llama.cpp" else "not-reported-by-engine",
        "speculative_decoding_stats": "engine-timings" if engine == "llama.cpp" else "unavailable(not-exposed-over-http)",
    }
