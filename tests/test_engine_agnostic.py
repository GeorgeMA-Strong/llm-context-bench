"""Tests for engine-agnostic transport, self-calculated timings, and the gates.

These cover the parts that must behave identically for llama.cpp, vLLM,
SGLang, and hosted OpenAI-compatible endpoints: which request parameters are
sent, how a stream is timestamped, how throughput is calculated from token
counts, and which trials are allowed into a reported median.
"""

import contextlib
import io
import json
from pathlib import Path
import shutil
import socket
import threading
import subprocess
import sys
import tempfile
import time
import unittest

from llm_context_bench import runner as benchmark


ROOT = Path(__file__).resolve().parents[1]
ENGINE = benchmark.engine_api


def sse_lines(events, done=True):
    lines = [b"data: " + json.dumps(event).encode() + b"\n\n" for event in events]
    if done:
        lines.append(b"data: [DONE]\n\n")
    return lines


class Clock:
    """Deterministic monotonic clock that advances by ``step`` on every read."""

    def __init__(self, step=1.0, start=0.0):
        self.now = start
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def varied_text(count=300):
    return " ".join(
        f"section{index} explains component{index} with consequence{index} and example{index}."
        for index in range(count)
    )


class StreamMetricsTests(unittest.TestCase):
    def test_prefill_and_generation_rates_come_from_observed_counts_and_clock(self):
        metrics = ENGINE.compute_timing_metrics(
            elapsed_s=42.0,
            prompt_tokens=16384,
            completion_tokens=1024,
            first_token_at=30.0,
            last_token_at=39.0,
            stream_chunks=1023,
            streamed=True,
        )
        self.assertEqual(metrics["ttft_s"], 30.0)
        self.assertAlmostEqual(metrics["prefill_tps"], 16384 / 30.0, places=5)
        self.assertEqual(metrics["decode_window_s"], 9.0)
        self.assertAlmostEqual(metrics["generation_tps"], 1023 / 9.0, places=5)
        self.assertAlmostEqual(metrics["mean_inter_token_latency_ms"], 9000.0 / 1023, places=4)
        self.assertAlmostEqual(metrics["output_tps_whole_request"], 1024 / 42.0, places=5)
        self.assertAlmostEqual(metrics["request_tps_total"], (16384 + 1024) / 42.0, places=5)
        self.assertEqual(metrics["stream_delivery"], "incremental")
        self.assertEqual(metrics["timing_reasons"], [])

    def test_buffered_stream_reports_no_decode_rate_instead_of_a_false_one(self):
        buffered = ENGINE.compute_timing_metrics(
            elapsed_s=42.0,
            prompt_tokens=16384,
            completion_tokens=1024,
            first_token_at=41.0,
            last_token_at=41.0,
            stream_chunks=1,
            streamed=True,
        )
        self.assertEqual(buffered["stream_delivery"], "buffered")
        self.assertIsNone(buffered["generation_tps"])
        self.assertIn("stream_delivered_in_one_chunk", buffered["timing_reasons"])
        self.assertAlmostEqual(buffered["output_tps_whole_request"], 1024 / 42.0, places=5)

    def test_non_streaming_lane_still_reports_the_whole_request_rate(self):
        whole = ENGINE.compute_timing_metrics(
            elapsed_s=20.0,
            prompt_tokens=16384,
            completion_tokens=1024,
            first_token_at=None,
            last_token_at=None,
            stream_chunks=0,
            streamed=False,
        )
        self.assertIsNone(whole["prefill_tps"])
        self.assertIsNone(whole["ttft_s"])
        self.assertEqual(whole["output_tps_whole_request"], 1024 / 20.0)
        self.assertEqual(whole["timing_reasons"], ["streaming_disabled"])

    def test_stream_capture_skips_empty_deltas_and_reads_usage(self):
        events = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "Alpha "}}]},
            {"choices": [{"index": 0, "delta": {"content": "beta"}, "finish_reason": "length"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 16384,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        ]
        measured = ENGINE.consume_sse(sse_lines(events), started=0.0, clock=Clock())
        self.assertTrue(measured["ok"])
        self.assertEqual(measured["response"], "Alpha beta")
        self.assertEqual(measured["stream_chunks"], 2)
        # The role-only delta must not be counted as the first token.
        self.assertEqual(measured["first_token_at"], 2.0)
        self.assertEqual(measured["last_token_at"], 3.0)
        self.assertEqual(measured["finish_reason"], "length")
        self.assertEqual(measured["usage"]["prompt_tokens"], 16384)

    def test_stream_capture_keeps_engine_timings_and_reasoning(self):
        events = [
            {
                "choices": [{"index": 0, "delta": {"reasoning_content": "thinking "}}],
                "timings": {"prompt_n": 10, "predicted_n": 1, "cache_n": 0},
            },
            {
                "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "length"}],
                "timings": {"prompt_n": 10, "predicted_n": 2, "cache_n": 0},
            },
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "timings": {
                    "prompt_n": 10,
                    "predicted_n": 2,
                    "cache_n": 0,
                    "prompt_ms": 20.0,
                    "predicted_ms": 40.0,
                },
            },
        ]
        measured = ENGINE.consume_sse(sse_lines(events), started=0.0, clock=Clock())
        self.assertEqual(measured["reasoning_response"], "thinking ")
        self.assertEqual(measured["response"], "answer")
        self.assertEqual(measured["timings"]["predicted_n"], 2)
        cache = ENGINE.detect_cached_tokens(measured)
        self.assertTrue(cache["reported"])
        self.assertEqual(cache["reported_by"], "timings.cache_n")
        self.assertTrue(cache["no_prompt_cache"])

    def test_stream_failures_become_data(self):
        error_body = ENGINE.consume_sse(
            [b'{"error":{"message":"context length exceeded"}}\n\n'],
            started=0.0,
            clock=Clock(),
        )
        self.assertFalse(error_body["ok"])
        self.assertIn("context length exceeded", error_body["error"])
        empty = ENGINE.consume_sse([b"data: {}\n\n", b"data: [DONE]\n\n"], started=0.0, clock=Clock())
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["error"], "stream contained no completion chunks")
        truncated = ENGINE.consume_sse(
            [b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'],
            started=0.0,
            clock=Clock(),
        )
        self.assertTrue(truncated["stream_truncated"])

    def test_sse_control_lines_are_not_mistaken_for_errors(self):
        measured = ENGINE.consume_sse(
            [b": ping\n\n", b"event: message\n\ndata: {}\n\n", b"data: [DONE]\n\n"],
            started=0.0,
            clock=Clock(),
        )
        self.assertEqual(measured["error"], "stream contained no completion chunks")
        self.assertIsNone(measured["first_token_at"])

    def test_token_counts_name_their_source(self):
        self.assertEqual(
            ENGINE.token_counts({"usage": {"prompt_tokens": 5, "completion_tokens": 7}}),
            {"prompt_tokens": 5, "completion_tokens": 7, "token_count_source": "usage"},
        )
        from_timings = ENGINE.token_counts({"usage": {}, "timings": {"prompt_n": 9, "predicted_n": 3}})
        self.assertEqual(from_timings["completion_tokens"], 3)
        self.assertEqual(from_timings["token_count_source"], "engine-timings")
        estimated = ENGINE.token_counts({"usage": {}, "timings": {}, "stream_chunks": 4})
        self.assertEqual(estimated["completion_tokens"], 4)
        self.assertEqual(estimated["token_count_source"], "stream-chunk-estimate")

    def test_cache_detection_covers_each_engine_field(self):
        cases = {
            "vllm": ({"usage": {"prompt_tokens_details": {"cached_tokens": 32}}}, "usage_prompt_tokens_details.cached_tokens"),
            "sglang": ({"usage": {"cached_tokens": 16}}, "usage.cached_tokens"),
            "openai": ({"usage": {"prompt_tokens_details": {"cached_tokens": 8}}}, "usage_prompt_tokens_details.cached_tokens"),
        }
        for name, (measured, field) in cases.items():
            with self.subTest(engine=name):
                cache = ENGINE.detect_cached_tokens(measured)
                self.assertTrue(cache["reported"])
                self.assertEqual(cache["reported_by"], field)
                self.assertFalse(cache["no_prompt_cache"])
        silent = ENGINE.detect_cached_tokens({"usage": {}, "timings": {}})
        self.assertFalse(silent["reported"])
        self.assertIsNone(silent["cached_tokens"])


class PayloadAndDetectionTests(unittest.TestCase):
    SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 3407}

    def build(self, engine, **kwargs):
        return ENGINE.build_chat_payload(
            engine=engine,
            model="model",
            messages=[{"role": "user", "content": "prompt"}],
            max_tokens=1024,
            sampling=self.SAMPLING,
            stream=True,
            **kwargs,
        )

    def test_payload_only_carries_parameters_the_selected_engine_accepts(self):
        llama, dropped = self.build("llama.cpp", ignore_eos=True)
        self.assertEqual(dropped, [])
        self.assertIs(llama["cache_prompt"], False)
        self.assertEqual(llama["top_k"], 20)
        self.assertIs(llama["ignore_eos"], True)
        self.assertEqual(llama["stream_options"], {"include_usage": True})

        vllm, dropped = self.build("vllm", ignore_eos=True)
        # cache_prompt is never offered to a non-llama.cpp engine at all, so
        # there is nothing to drop for it; only sampling extras can be dropped.
        self.assertEqual(dropped, [])
        self.assertNotIn("cache_prompt", vllm)
        self.assertEqual(vllm["min_p"], 0.0)

        hosted, dropped = self.build("openai", ignore_eos=True)
        self.assertEqual(dropped, ["ignore_eos", "min_p", "top_k"])
        for key in ("top_k", "min_p", "cache_prompt", "ignore_eos"):
            self.assertNotIn(key, hosted)
        self.assertEqual(hosted["temperature"], 1.0)
        self.assertEqual(hosted["seed"], 3407)
        self.assertIs(hosted["stream"], True)

    def test_explicit_chat_template_override_is_never_dropped(self):
        payload, _ = self.build("openai", chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_engine_detection_uses_endpoint_signatures(self):
        def llama_probe(path):
            if path == "/props":
                return 0.01, {"total_slots": 4, "build_info": "mock", "http_status": 200}
            return 0.01, {"error": "not found", "http_status": 404}

        self.assertEqual(ENGINE.detect_engine(llama_probe)["resolved"], "llama.cpp")

        def vllm_probe(path):
            if path == "/server_info":
                return 0.01, {"vllm_config": {}, "http_status": 200}
            return 0.01, {"error": "not found", "http_status": 404}

        self.assertEqual(ENGINE.detect_engine(vllm_probe)["resolved"], "vllm")

        def nothing(path):
            return 0.01, {"error": "not found", "http_status": 404}

        self.assertEqual(ENGINE.detect_engine(nothing)["resolved"], "generic")
        self.assertEqual(ENGINE.detect_engine(nothing)["method"], "endpoint-signature")

        def models_only(path):
            if path == "/v1/models":
                return 0.01, {
                    "object": "list",
                    "data": [{"id": "active", "owned_by": "vllm"}],
                    "http_status": 200,
                }
            return 0.01, {"error": "not found", "http_status": 404}

        by_ownership = ENGINE.detect_engine(models_only)
        self.assertEqual(by_ownership["resolved"], "vllm")
        self.assertEqual(by_ownership["method"], "models-owned-by")
        # A 200 response that is not a model list must not guess an engine.
        self.assertIsNone(ENGINE.detect_engine_from_ownership({"data": None, "http_status": 200}))
        self.assertIsNone(ENGINE.detect_engine_from_ownership({"data": [{"id": "m"}], "http_status": 200}))

        def broken(path):
            raise OSError("connection refused")

        self.assertEqual(ENGINE.detect_engine(broken)["resolved"], "generic")
        explicit = ENGINE.detect_engine(nothing, "sglang")
        self.assertEqual((explicit["resolved"], explicit["method"]), ("sglang", "cli"))

    def test_generic_engine_is_promoted_by_its_timings_object(self):
        promoted = ENGINE.refine_engine("generic", {"timings": {"prompt_n": 8, "predicted_n": 2}})
        self.assertEqual(promoted, "llama.cpp")
        self.assertIsNone(ENGINE.refine_engine("generic", {"timings": {}}))
        self.assertIsNone(ENGINE.refine_engine("vllm", {"timings": {"prompt_n": 8, "predicted_n": 2}}))

    def test_metric_sources_declare_what_nobody_measures(self):
        hosted = ENGINE.metric_sources(streamed=False, engine="openai")
        self.assertIn("unavailable", hosted["prefill_tps"])
        self.assertIn("not-exposed-over-http", hosted["speculative_decoding_stats"])
        streamed = ENGINE.metric_sources(streamed=True, engine="llama.cpp")
        self.assertIn("harness-wall-clock", streamed["prefill_tps"])
        self.assertEqual(streamed["engine_reported_timings"], "cross-check-only")


class ChatMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.original_sse = benchmark.request_sse

    def tearDown(self):
        benchmark.request_sse = self.original_sse

    def fake_stream(self, token_count=4, prompt_tokens=16384, timings=None):
        def transport(_base_url, _path, payload, _api_key, _timeout, client="urllib"):
            events = [
                {"choices": [{"index": 0, "delta": {"content": f"term{index} "}}]}
                for index in range(token_count)
            ]
            events[-1]["choices"][0]["finish_reason"] = "length"
            final = {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": token_count}}
            if timings:
                final["timings"] = timings
            events.append(final)
            measured = ENGINE.consume_sse(
                sse_lines(events), started=0.0, clock=Clock(step=1.0)
            )
            measured["request_payload"] = payload
            return measured

        return transport

    def test_chat_once_calculates_prefill_and_decode_rates_from_the_stream(self):
        benchmark.request_sse = self.fake_stream()
        trial = benchmark.chat_once(
            "http://example.invalid",
            "",
            "model",
            "system",
            "prompt",
            1024,
            {"temperature": 1.0, "top_p": 0.95, "seed": 3407},
            10,
            stream=True,
            engine="vllm",
        )
        self.assertTrue(trial["ok"], trial["error"])
        self.assertEqual(trial["prompt_tokens"], 16384)
        self.assertEqual(trial["completion_tokens"], 4)
        self.assertEqual(trial["stream_chunks"], 4)
        self.assertEqual(trial["ttft_s"], 2.0)
        self.assertEqual(trial["decode_window_s"], 3.0)
        self.assertAlmostEqual(trial["prefill_tps"], 16384 / 2.0, places=5)
        self.assertAlmostEqual(trial["generation_tps"], 1.0, places=5)
        self.assertAlmostEqual(trial["mean_inter_token_latency_ms"], 1000.0, places=4)
        self.assertEqual(trial["stream_delivery"], "incremental")
        self.assertEqual(trial["token_count_source"], "usage")
        # vLLM has no in-band counters, so the cross-check stays silent.
        self.assertFalse(trial["engine_cross_check"]["reported"])
        self.assertIsNone(trial["engine_cross_check"]["generation_agreement"])

    def test_engine_reported_rates_are_compared_with_the_measured_ones(self):
        benchmark.request_sse = self.fake_stream(
            timings={"prompt_n": 16384, "predicted_n": 4, "cache_n": 0, "prompt_ms": 2000.0, "predicted_ms": 3000.0}
        )
        trial = benchmark.chat_once(
            "http://example.invalid",
            "",
            "model",
            "system",
            "prompt",
            1024,
            {"temperature": 1.0, "top_p": 0.95, "seed": 3407},
            10,
            stream=True,
            engine="llama.cpp",
        )
        check = trial["engine_cross_check"]
        self.assertTrue(check["reported"])
        self.assertEqual(check["engine_prompt_tps_recalculated"], round(16384 * 1000.0 / 2000.0, 6))
        self.assertEqual(check["engine_generation_tps_recalculated"], round(3 * 1000.0 / 3000.0, 6))
        self.assertEqual(check["prefill_agreement"]["relative_error"], 0.0)
        self.assertTrue(check["engine_token_counts_consistent"])
        self.assertTrue(trial["prompt_cache"]["no_prompt_cache"])

    def test_retryable_failures_are_retried_then_recorded(self):
        attempts = []

        def transport(*_args, **_kwargs):
            attempts.append(1)
            if len(attempts) < 2:
                return ENGINE.empty_stream_measurement(
                    started=0.0, error="rate limited", http_status=429
                )
            return ENGINE.consume_sse(
                sse_lines(
                    [
                        {
                            "choices": [{"delta": {"content": "ok"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        }
                    ]
                ),
                started=0.0,
                clock=Clock(),
            )

        benchmark.request_sse = transport
        trial = benchmark.chat_once(
            "http://example.invalid",
            "",
            "model",
            "system",
            "prompt",
            8,
            {"temperature": 0.0, "top_p": 1.0, "seed": 3407},
            10,
            stream=True,
            max_retries=1,
            retry_backoff_s=0.0,
        )
        self.assertTrue(trial["ok"], trial["error"])
        self.assertEqual(trial["attempts"], 2)
        self.assertEqual(trial["retry_statuses"], [429])

    def test_stalled_stream_is_not_retried(self):
        calls = []

        def transport(*_args, **_kwargs):
            calls.append(1)
            return ENGINE.empty_stream_measurement(started=0.0, error="timeout", http_status=None)

        benchmark.request_sse = transport
        trial = benchmark.chat_once(
            "http://example.invalid",
            "",
            "model",
            "system",
            "prompt",
            8,
            {"temperature": 0.0, "top_p": 1.0, "seed": 3407},
            10,
            stream=True,
            max_retries=2,
            retry_backoff_s=0.0,
        )
        self.assertFalse(trial["ok"])
        self.assertEqual(len(calls), 3)


class ValidityGateTests(unittest.TestCase):
    def base_trial(self):
        return {
            "ok": True,
            "error": None,
            "response": varied_text(700),
            "reasoning_response": "",
            "prompt_tokens": 16384,
            "completion_tokens": 1024,
            "produced_tokens": 1024,
            "stream": True,
            "stream_delivery": "incremental",
            "ttft_s": 20.0,
            "prefill_tps": 819.2,
            "generation_tps": 51.2,
            "timing_reasons": [],
            "prompt_cache": {"cached_tokens": 0, "reported": True, "no_prompt_cache": True},
            "timings": {},
        }

    def evaluate(self, trial, tolerance=2.0):
        benchmark.evaluate_performance_trial(
            trial,
            required_output_tokens=1024,
            nominal_input_tokens=16384,
            tolerance_percent=tolerance,
        )
        return trial

    def test_clean_engine_agnostic_trial_is_valid(self):
        trial = self.evaluate(self.base_trial())
        self.assertTrue(trial["performance_valid"], trial["invalid_reasons"])
        self.assertEqual(trial["invalid_reasons"], [])
        self.assertEqual(trial["input_size_percent_from_nominal"], 0.0)
        self.assertIn("engine_reports_no_timings", trial["notes"])

    def test_reported_prompt_cache_hit_invalidates_the_trial(self):
        trial = self.base_trial()
        trial["prompt_cache"] = {"cached_tokens": 4096, "reported": True, "no_prompt_cache": False}
        self.assertIn("prompt_cache_hit", self.evaluate(trial)["invalid_reasons"])

    def test_unreported_prompt_cache_is_flagged_not_assumed(self):
        trial = self.base_trial()
        trial["prompt_cache"] = {"cached_tokens": None, "reported": False, "no_prompt_cache": None}
        trial = self.evaluate(trial)
        self.assertTrue(trial["performance_valid"])
        self.assertIn("prompt_cache_not_reported_by_engine", trial["notes"])

    def test_buffered_stream_is_rejected(self):
        trial = self.base_trial()
        trial["stream_delivery"] = "buffered"
        self.assertIn("stream_not_incremental", self.evaluate(trial)["invalid_reasons"])

    def test_tokenizer_drift_uses_the_configured_tolerance(self):
        trial = self.base_trial()
        trial["prompt_tokens"] = 17000
        self.assertIn("input_size_outside_tolerance", self.evaluate(trial)["invalid_reasons"])
        trial = self.base_trial()
        trial["prompt_tokens"] = 17000
        trial = self.evaluate(trial, tolerance=5.0)
        self.assertTrue(trial["performance_valid"])
        self.assertAlmostEqual(trial["input_size_percent_from_nominal"], 3.7598, places=4)

    def test_short_output_is_rejected(self):
        trial = self.base_trial()
        trial["produced_tokens"] = 512
        self.assertIn("output_length_mismatch", self.evaluate(trial)["invalid_reasons"])

    def test_unreported_prompt_length_is_named_not_guessed(self):
        trial = self.base_trial()
        trial["prompt_tokens"] = None
        reasons = self.evaluate(trial)["invalid_reasons"]
        self.assertIn("prompt_tokens_not_reported", reasons)
        self.assertNotIn("input_size_outside_tolerance", reasons)


class SummaryTests(unittest.TestCase):
    def test_medians_and_table_describe_speed_per_input_tier(self):
        trial = {
            "ok": True,
            "elapsed_s": 40.0,
            "prompt_tokens": 16384,
            "ttft_s": 30.0,
            "decode_window_s": 9.0,
            "prefill_tps": 16384 / 30.0,
            "generation_tps": 1023 / 9.0,
            "mean_inter_token_latency_ms": 9000.0 / 1023,
            "output_tps_whole_request": 1024 / 40.0,
            "request_tps_total": (16384 + 1024) / 40.0,
            "prompt_tps_engine": 550.0,
            "generation_tps_engine": 114.0,
            "input_size_percent_from_nominal": 0.0,
            "stream_delivery": "incremental",
            "prompt_cache": {"reported": True},
            "engine_cross_check": {
                "prefill_agreement": {"relative_error": 0.02},
                "generation_agreement": {"relative_error": 0.5},
            },
            "draft_acceptance_rate": 0.6,
            "performance_valid": True,
            "invalid_reasons": [],
        }
        summary = benchmark.summarize_performance_trials([trial], 1024, 16384)
        self.assertEqual(summary["median_prefill_tps"], round(16384 / 30.0, 6))
        self.assertEqual(summary["median_generation_tps"], round(1023 / 9.0, 6))
        self.assertEqual(summary["median_ttft_s"], 30.0)
        self.assertEqual(summary["median_prompt_tps_engine"], 550.0)
        self.assertTrue(summary["performance_valid"])
        self.assertEqual(summary["median_engine_vs_harness_generation_relative_error"], 0.5)
        self.assertEqual(summary["median_draft_acceptance_rate"], 0.6)

        table = benchmark.build_performance_table(
            {
                "regular": {
                    "cases": [
                        {
                            "id": "regular-05-context-16k",
                            "nominal_input_tokens": 16384,
                            "performance": {"summary": summary},
                        }
                    ]
                }
            }
        )
        row = table[0]
        self.assertEqual(row["case"], "regular-05-context-16k")
        self.assertEqual(row["prefill_tps"], round(16384 / 30.0, 6))
        self.assertEqual(row["token_generation_tps"], round(1023 / 9.0, 6))
        text = benchmark.format_performance_table(table)
        self.assertIn("TOTAL PP t/s", text)
        self.assertIn("regular", text)
        self.assertIn("16k", text)


class LoadGroupTests(unittest.TestCase):
    """Simultaneous-request maths; no server involved."""

    def trial(self, index, *, ok=True, valid=True, ttft=2.0, generation=40.0, offset=0.0):
        return {
            "ok": ok,
            "performance_valid": valid,
            "request_index": index,
            "prompt_tokens": 8192,
            "completion_tokens": 1024,
            "ttft_s": ttft,
            "decode_window_s": 20.0,
            "group_start_offset_s": offset,
            "prefill_tps": 8192 / ttft if ttft else None,
            "generation_tps": generation,
            "elapsed_s": 25.0,
            "invalid_reasons": [] if valid else ["output_length_mismatch"],
        }

    def group(self, trials, **kwargs):
        arguments = {"wall_s": 10.0}
        arguments.update(kwargs)
        return benchmark.summarize_load_group(
            trials,
            group_id="regular-09-context-8k#01",
            concurrency=kwargs.get("concurrency", len(trials)),
            wall_s=arguments["wall_s"],
            barrier_broken=arguments.get("barrier_broken", False),
        )

    def test_group_totals_add_up_every_token_the_group_processed(self):
        group = self.group([self.trial(index) for index in range(1, 5)])
        self.assertEqual(group["requests"], 4)
        self.assertEqual(group["valid_requests"], 4)
        self.assertTrue(group["group_valid"])
        self.assertEqual(group["wall_s"], 10.0)
        # 4 requests x 8,192 prompt tokens and 4 x 1,024 generated tokens.
        self.assertEqual(group["prompt_tokens_total"], 32768)
        self.assertEqual(group["output_tokens_total"], 4096)
        self.assertEqual(group["tokens_total"], 36864)

    def test_totals_report_the_speed_of_the_whole_load_not_of_one_stream(self):
        group = self.group([self.trial(index) for index in range(1, 5)])
        # All four streams together produced 4,096 tokens in the 10 s wall time.
        self.assertEqual(group["decode_tps_total"], 409.6)
        # Prompt and generated tokens together are the engine's total throughput.
        self.assertEqual(group["tokens_tps_total"], 3686.4)
        # All 32,768 prompt tokens were digested by the time the slowest stream
        # produced its first token, 2 s after the release.
        self.assertEqual(group["prefill_tps_total"], 16384.0)
        self.assertEqual(group["requests_per_minute"], 24.0)
        self.assertEqual(group["ttft_max_s"], 2.0)
        self.assertEqual(group["invalid_reason_totals"], {})

    def test_prefill_total_waits_for_the_slowest_stream_to_start_generating(self):
        # One stream queued for 6 s while the others started in 2 s, so the
        # group's prefill phase ended at 6 s, not at the fastest request's 2 s.
        group = self.group([self.trial(1), self.trial(2, ttft=6.0)])
        self.assertEqual(group["prefill_tps_total"], round(2 * 8192 / 6.0, 6))
        self.assertEqual(group["ttft_max_s"], 6.0)

    def test_total_tg_sums_every_stream_rather_than_scaling_one_of_them(self):
        # Four streams generating together over the same 20 s window: the total
        # is all their tokens over that window, which happens to be 4 x one
        # stream here because the synthetic streams overlap perfectly.
        group = self.group([self.trial(index) for index in range(1, 5)])
        self.assertEqual(group["generating_window_s"], 20.0)
        self.assertEqual(group["generating_tokens_total"], 4 * 1023)
        self.assertEqual(group["token_generation_tps_total"], round(4 * 1023 / 20.0, 6))
        self.assertEqual(group["prefill_tps_total"], round(4 * 8192 / 2.0, 6))

        # Stagger the streams and the shared window lengthens, so the total
        # drops even though each stream is unchanged - which is exactly what a
        # per-stream rate times the request count would get wrong.
        staggered = self.group([self.trial(1), self.trial(2, offset=5.0)])
        self.assertEqual(staggered["generating_window_s"], 25.0)
        self.assertEqual(
            staggered["token_generation_tps_total"], round(2 * 1023 / 25.0, 6)
        )
        self.assertLess(
            staggered["token_generation_tps_total"], group["token_generation_tps_total"]
        )

    def test_one_broken_request_makes_the_whole_load_group_unusable(self):
        group = self.group([self.trial(1), self.trial(2, valid=False)])
        self.assertFalse(group["group_valid"])
        self.assertEqual(group["valid_requests"], 1)
        self.assertEqual(group["invalid_reason_totals"], {"output_length_mismatch": 1})
        # Totals stay recorded; they are simply not counted as valid.
        self.assertGreater(group["decode_tps_total"], 0)

    def test_a_group_without_a_measured_wall_time_reports_no_totals(self):
        group = self.group([self.trial(1), self.trial(2)], wall_s=None)
        self.assertIsNone(group["decode_tps_total"])
        self.assertIsNone(group["tokens_tps_total"])
        self.assertFalse(group["group_valid"])
        self.assertIn("group_wall_unmeasured", group["invalid_reason_totals"])

    def test_a_broken_start_barrier_invalidates_simultaneity_not_single_requests(self):
        group = self.group([self.trial(1), self.trial(2)], barrier_broken=True)
        self.assertFalse(group["group_valid"])
        self.assertIn("concurrency_barrier_broken", group["invalid_reason_totals"])

    def test_partial_group_reports_what_never_arrived(self):
        group = self.group([self.trial(1)], concurrency=4)
        self.assertEqual(group["recorded_requests"], 1)
        self.assertEqual(group["missing_requests"], 3)
        self.assertFalse(group["group_valid"])

    def test_totals_need_token_counts_from_every_request(self):
        trials = [self.trial(1), self.trial(2)]
        trials[1]["prompt_tokens"] = None
        group = self.group(trials)
        self.assertIsNone(group["tokens_tps_total"])
        self.assertIsNone(group["prefill_tps_total"])
        self.assertEqual(group["output_tokens_total"], 2 * 1024)
        self.assertEqual(group["decode_tps_total"], 2 * 1024 / 10.0)

    def test_a_degenerate_stream_is_flagged_apart_from_a_measurement_failure(self):
        degenerate = [self.trial(1), self.trial(2, valid=True)]
        degenerate[1]["performance_valid"] = False
        degenerate[1]["invalid_reasons"] = ["dominant_repeated_token"]
        group = self.group(degenerate)
        self.assertFalse(group["group_valid"])
        self.assertTrue(group["content_only_failure"])

        measured = self.group([self.trial(1), self.trial(2, valid=False)])
        self.assertFalse(measured["content_only_failure"])

    def test_invalid_load_groups_are_excluded_from_the_reported_totals(self):
        summary = benchmark.summarize_performance_trials([], 1024, 8192)
        group = {
            "group_valid": False,
            "decode_tps_total": 900.0,
            "tokens_tps_total": 9000.0,
            "token_generation_tps_total": 850.0,
            "prefill_tps_total": 90000.0,
            "requests_per_minute": 60.0,
            "ttft_max_s": 1.0,
            "wall_s": 10.0,
            "prompt_tokens_total": 32768,
            "output_tokens_total": 4096,
            "tokens_total": 36864,
        }
        row = benchmark.build_performance_table(
            {
                "regular": {
                    "cases": [
                        {
                            "id": "regular-09-context-8k",
                            "nominal_input_tokens": 8192,
                            "performance": {
                                "summary": summary,
                                "concurrency": 4,
                                "groups": [group],
                            },
                        }
                    ]
                }
            }
        )[0]
        self.assertIsNone(row["decode_tps_total"])
        self.assertEqual(row["groups"], 1)
        self.assertEqual(row["valid_groups"], 0)

    def test_single_flight_group_needs_no_threads(self):
        seen = []

        def request(index):
            seen.append(index)
            return self.trial(index)

        trials, span = benchmark.run_concurrent_group(request, concurrency=1)
        self.assertEqual(len(trials), 1)
        self.assertEqual(seen, [1])
        self.assertFalse(span["barrier_broken"])
        self.assertGreater(span["wall_s"], 0)

    def test_concurrent_requests_all_start_before_any_of_them_finish(self):
        delay = 0.05

        def request(index):
            time.sleep(delay)
            return self.trial(index)

        trials, span = benchmark.run_concurrent_group(request, concurrency=4)
        self.assertEqual([trial["request_index"] for trial in trials], [1, 2, 3, 4])
        # Serialised requests would need 4 x delay; parallel ones share the window.
        self.assertLess(span["wall_s"], delay * 3)

    def test_start_barrier_holds_requests_until_every_worker_is_ready(self):
        """No request may begin until all workers have reached the barrier."""
        rendezvous = threading.Barrier(3, timeout=5)
        order = []
        lock = threading.Lock()

        def request(index):
            with lock:
                order.append(index)
            try:
                # If the runner serialised requests, the first one would wait
                # here alone and the barrier would break.
                rendezvous.wait()
            except threading.BrokenBarrierError:
                self.fail(f"request {index} did not run alongside the others")
            return self.trial(index)

        trials, span = benchmark.run_concurrent_group(request, concurrency=3)
        self.assertEqual(len(trials), 3)
        self.assertFalse(span["barrier_broken"])
        self.assertEqual(sorted(order), [1, 2, 3])

    def test_a_worker_that_raises_costs_only_its_own_slot(self):
        def request(index):
            if index == 2:
                raise RuntimeError("worker exploded")
            return self.trial(index)

        trials, _span = benchmark.run_concurrent_group(request, concurrency=3)
        self.assertEqual(len(trials), 3)
        self.assertTrue(trials[0]["ok"])
        self.assertFalse(trials[1]["ok"])
        self.assertIn("worker exploded", trials[1]["error"])


class MockEngineIntegrationTests(unittest.TestCase):
    """Drive the real CLI against a streaming OpenAI-compatible server.

    The mock runs as a subprocess so this process stays single-threaded: the
    quality lane executes candidate code in a forked child, which is not safe
    in a multi-threaded process.
    """

    MOCK = ROOT / "tools" / "mock_openai_engine.py"

    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        cls.process = subprocess.Popen(
            [
                sys.executable,
                str(cls.MOCK),
                "--port",
                str(port),
                "--prefill-delay",
                "0.02",
                "--decode-delay",
                "0",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.wait_for_ready()

    @classmethod
    def wait_for_ready(cls) -> None:
        for _ in range(100):
            _, body = benchmark.request_json(cls.base_url, "/health", None, "", 2)
            if isinstance(body, dict) and body.get("status"):
                return
            time.sleep(0.05)
        raise RuntimeError("mock engine did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.process.kill()

    def run_bench(self, output: Path, *extra) -> dict:
        """Run the real entry point with argv substituted and output captured."""
        argv = [
            "--base-url", self.base_url,
            "--model", "mock-model",
            "--profile", "mock-profile",
            "--output", str(output),
            "--suite", "regular",
            "--lane", "performance",
            "--sizes", "16k",
            "--command", "python tools/mock_openai_engine.py",
            "--system", "mock engine; CI container",
            *extra,
        ]
        original_parse_args = benchmark.parse_args
        benchmark.parse_args = lambda: original_parse_args(argv)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as console:
                benchmark.main()
        finally:
            benchmark.parse_args = original_parse_args
        self.assertIn("TOTAL PP t/s", console.getvalue())
        return json.loads(Path(output).read_text())

    def test_streaming_server_yields_measured_prefill_and_decode_speeds(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_bench(Path(directory) / "result.json")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["benchmark_schema_version"], 8)
            self.assertEqual(result["engine"]["resolved"], "llama.cpp")
            self.assertEqual(result["engine"]["detection_method"], "endpoint-signature")
            row = result["performance_table"][0]
            self.assertTrue(row["performance_valid"], row["invalid_reason_totals"])
            self.assertGreater(row["prefill_tps"], 0)
            self.assertGreater(row["token_generation_tps"], 0)
            self.assertGreater(row["ttft_s"], 0)
            self.assertGreater(row["whole_request_output_tps"], 0)
            case = result["suites"]["regular"]["cases"][0]["performance"]
            trial = case["trials"][0]
            self.assertEqual(trial["stream_delivery"], "incremental")
            self.assertEqual(trial["token_count_source"], "usage")
            self.assertTrue(trial["prompt_cache"]["no_prompt_cache"])
            # The mock reports a llama.cpp-style timings object, so the
            # advisory cross-check has something to compare the measured
            # rates against even though nothing here gates on it.
            self.assertTrue(trial["engine_cross_check"]["reported"])
            self.assertIsNotNone(trial["engine_cross_check"]["prefill_agreement"])
            self.assertEqual(
                result["measurement"]["metric_sources"]["generation_tps"],
                "harness-wall-clock((completion_tokens-1)/decode_window)",
            )

    def test_hosted_profile_sends_the_strict_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_bench(Path(directory) / "result.json", "--engine", "openai")
            self.assertEqual(result["engine"]["resolved"], "openai")
            trial = result["suites"]["regular"]["cases"][0]["performance"]["trials"][0]
            self.assertEqual(trial["engine_only_params_dropped"], ["min_p", "top_k"])
            self.assertEqual(result["engine"]["engine_only_params"], [])
            self.assertTrue(trial["performance_valid"], trial["invalid_reasons"])

    def test_curl_transport_streams_the_same_measurement(self):
        """The opt-in curl client must measure identically to urllib."""
        if not shutil.which("curl"):
            self.skipTest("curl is not installed")
        measured = benchmark.request_sse(
            self.base_url,
            "/v1/chat/completions",
            {
                "model": "mock-model",
                "messages": [{"role": "user", "content": "count to eight"}],
                "max_tokens": 8,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            "",
            30,
            client="curl",
        )
        self.assertTrue(measured["ok"], measured["error"])
        self.assertEqual(measured["http_status"], 200)
        self.assertEqual(measured["stream_chunks"], 8)
        self.assertEqual(measured["usage"]["completion_tokens"], 8)
        self.assertIsNotNone(measured["first_token_at"])
        self.assertIsNotNone(measured["done_at"])
        self.assertFalse(measured["stream_truncated"])

    def test_concurrent_identical_requests_are_aggregated_per_load_group(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_bench(
                Path(directory) / "concurrency.json",
                "--sizes", "8k",
                "--concurrency", "4",
                # The mock's token estimate differs from the pinned tokenizer.
                "--input-size-tolerance-percent", "6",
            )
            self.assertEqual(result["run_parameters"]["concurrency"], 4)
            self.assertFalse(result["canonical"], "a concurrent run is not the canonical suite")
            case = result["suites"]["regular"]["cases"][0]
            self.assertEqual(case["id"], "regular-09-context-8k")
            performance = case["performance"]
            self.assertEqual(performance["concurrency"], 4)
            self.assertEqual(len(performance["groups"]), 1)

            group = performance["groups"][0]
            self.assertEqual(group["requests"], 4)
            self.assertEqual(group["recorded_requests"], 4)
            self.assertTrue(group["group_valid"], group["invalid_reason_totals"])
            self.assertGreater(group["wall_s"], 0)
            self.assertGreater(group["decode_tps_total"], 0)
            self.assertGreater(group["token_generation_tps_total"], 0)
            self.assertGreater(group["prefill_tps_total"], 0)
            self.assertGreater(group["tokens_tps_total"], group["decode_tps_total"])

            # The totals are the plain sum of what every request in the group
            # moved, divided by the group's own wall time.
            trials = performance["trials"]
            output = sum(trial["completion_tokens"] for trial in trials)
            prompt = sum(trial["prompt_tokens"] for trial in trials)
            self.assertEqual(group["output_tokens_total"], output)
            self.assertEqual(group["prompt_tokens_total"], prompt)
            self.assertEqual(group["tokens_total"], output + prompt)
            expected = output / group["wall_s"]
            self.assertAlmostEqual(
                expected, group["decode_tps_total"], delta=expected * 1e-5
            )
            # Four streams together beat one stream, which is the point of the
            # column; if they never add up, the aggregate is a sum of nothing.
            self.assertGreater(
                group["decode_tps_total"], group["generation_tps_median"]
            )

            self.assertEqual([trial["request_index"] for trial in trials], [1, 2, 3, 4])
            self.assertEqual(len({trial["request_tag"] for trial in trials}), 4)
            self.assertTrue(all(trial["performance_valid"] for trial in trials))

            row = next(
                entry for entry in result["performance_table"] if entry["case"] == case["id"]
            )
            self.assertEqual(row["concurrency"], 4)
            self.assertEqual(row["valid_groups"], 1)
            self.assertGreater(row["decode_tps_total"], 0)
            console = benchmark.format_performance_table([row])
            self.assertIn("WHOLE LOAD", console)
            self.assertIn("ONE STREAM", console)
            self.assertIn("TOTAL TG t/s", console)
            self.assertIn("TOTAL PP t/s", console)
            self.assertIn("not one stream's rate times req", console.lower())

    def test_single_flight_runs_still_record_a_load_group(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_bench(Path(directory) / "single.json")
            self.assertEqual(result["run_parameters"]["concurrency"], 1)
            self.assertTrue(result["canonical"])
            performance = result["suites"]["regular"]["cases"][0]["performance"]
            self.assertEqual(len(performance["groups"]), 1)
            self.assertEqual(performance["groups"][0]["requests"], 1)

    def test_only_a_single_flight_full_suite_run_is_canonical(self):
        base = [
            "--base-url", "http://127.0.0.1:1", "--model", "m", "--profile", "p",
            "--output", "/tmp/x.json", "--command", "c", "--system", "s",
        ]
        self.assertTrue(benchmark.is_canonical(benchmark.parse_args(base)))
        concurrent = benchmark.parse_args([*base, "--concurrency", "4"])
        self.assertFalse(benchmark.is_canonical(concurrent))
        self.assertFalse(
            benchmark.is_canonical(benchmark.parse_args([*base, "--repetitions", "3"]))
        )

    def test_each_run_scopes_its_request_tags_so_a_rerun_cannot_warm_the_prefill(self):
        """Engines with automatic prefix caching key KV blocks on leading tokens.

        A tag that repeats between runs lets the second run start from a warm
        prefill, which on the real vLLM host measured 4,451 t/s against 1,310 t/s
        for the same prompt - and that endpoint reports no cached-token count.
        """
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_bench(Path(directory) / "first.json")
            second = self.run_bench(Path(directory) / "second.json")

        def tag_of(result):
            return result["suites"]["regular"]["cases"][0]["performance"]["trials"][0][
                "request_tag"
            ]

        self.assertNotEqual(tag_of(first), tag_of(second))
        for result in (first, second):
            scope = result["measurement"]["request_tag_scope"]
            self.assertTrue(tag_of(result).startswith(f"{scope}-"))
            self.assertIn(
                "run-scoped-request-tag",
                result["suites"]["regular"]["cases"][0]["performance"]["cold_cache_guards"],
            )

    def test_a_fixed_request_tag_scope_reproduces_an_earlier_runs_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_bench(
                Path(directory) / "a.json", "--request-tag-scope", "fixed-scope"
            )
            second = self.run_bench(
                Path(directory) / "b.json", "--request-tag-scope", "fixed-scope"
            )
        first_tags = [
            trial["request_tag"]
            for case in first["suites"]["regular"]["cases"]
            for trial in case["performance"]["trials"]
        ]
        second_tags = [
            trial["request_tag"]
            for case in second["suites"]["regular"]["cases"]
            for trial in case["performance"]["trials"]
        ]
        self.assertEqual(first_tags, second_tags)
        self.assertTrue(all(tag.startswith("fixed-scope-") for tag in first_tags))

    def test_concurrency_above_the_supported_maximum_is_refused(self):
        argv = [
            "--base-url", "http://127.0.0.1:1", "--model", "m", "--profile", "p",
            "--output", "/tmp/x.json", "--command", "c", "--system", "s",
            "--concurrency", "9",
        ]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(argv)

    def test_eight_k_tier_is_available_for_both_lanes(self):
        self.assertEqual(benchmark.SIZE_TOKENS["8k"], 8192)
        for name in ("regular", "coding"):
            suite, _digest = benchmark.load_suite(name)
            case = next(
                entry
                for entry in suite["cases"]
                if entry.get("nominal_input_tokens") == 8192
            )
            self.assertEqual(case["id"], f"{name}-09-context-8k")
            self.assertEqual(case["performance_max_tokens"], 1024)
            self.assertTrue(case["prompt_sha256"], "fixture hash must be locked in the suite")
            rendered = benchmark.render_prompt(case)
            self.assertIn("AUTHORITATIVE SYNTHETIC", rendered)

    def test_non_streaming_lane_keeps_the_whole_request_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_bench(Path(directory) / "result.json", "--no-stream")
            row = result["performance_table"][0]
            self.assertIsNone(row["prefill_tps"])
            self.assertGreater(row["whole_request_output_tps"], 0)
            trial = result["suites"]["regular"]["cases"][0]["performance"]["trials"][0]
            self.assertIn("streaming_disabled", trial["notes"])
            self.assertTrue(trial["performance_valid"], trial["invalid_reasons"])


if __name__ == "__main__":
    unittest.main()
