import gzip
import json
from pathlib import Path
import re
import tempfile
import unittest

from llm_context_bench import runner as benchmark


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(benchmark.__file__).resolve().parent


class SuiteTests(unittest.TestCase):
    def test_repetitive_forced_output_is_rejected(self):
        pathological = '{"answer": 42}\n' + ".\n" * 1000
        analysis = benchmark.analyze_output_content(pathological)
        self.assertFalse(analysis["valid"])
        self.assertIn("dominant_repeated_token", analysis["reasons"])

    def test_varied_output_is_accepted(self):
        varied = " ".join(
            f"section{i} explains component{i} with consequence{i} and example{i}."
            for i in range(300)
        )
        self.assertTrue(benchmark.analyze_output_content(varied)["valid"])

    def test_engine_timings_are_an_advisory_cross_check_not_a_gate(self):
        trial = {
            "prompt_tokens": 1024,
            "completion_tokens": 512,
            "prefill_tps": 500.0,
            "generation_tps": 40.0,
            "prompt_cache": {"cached_tokens": 0, "reported": True},
            "timings": {
                "cache_n": 0,
                "prompt_n": 1024,
                "prompt_ms": 2048.0,
                "prompt_per_second": 500.0,
                "predicted_n": 512,
                "predicted_ms": 12775.0,
                "predicted_per_second": 40.0,
            },
        }
        check = benchmark.engine_api.cross_check_engine_timings(trial)
        self.assertTrue(check["reported"])
        self.assertAlmostEqual(check["engine_prompt_tps_recalculated"], 500.0, places=3)
        self.assertAlmostEqual(check["prefill_agreement"]["relative_error"], 0.0, places=6)
        self.assertEqual(check["engine_reported_prompt_rate_error"], 0.0)
        self.assertEqual(check["engine_reported_generation_rate_error"], 0.0)
        self.assertTrue(check["engine_token_counts_consistent"])
        # Cached prompts are still visible through the cross-check.
        trial["timings"]["cache_n"] = 100
        trial["timings"]["prompt_n"] = 924
        trial["timings"]["prompt_ms"] = 1848.0
        trial["prompt_cache"] = benchmark.engine_api.detect_cached_tokens(
            {"timings": trial["timings"], "usage": {}}
        )
        self.assertFalse(trial["prompt_cache"]["no_prompt_cache"])
        # A rate the engine advertises but its own counters contradict is
        # surfaced, never silently trusted.
        trial["timings"]["prompt_per_second"] = 900.0
        skewed = benchmark.engine_api.cross_check_engine_timings(trial)
        self.assertGreater(skewed["engine_reported_prompt_rate_error"], 0.5)
        self.assertEqual(skewed["engine_reported_generation_rate_error"], 0.0)

    def test_engine_without_timings_still_reports_measured_rates(self):
        trial = {
            "ok": True,
            "prompt_tokens": 33455,
            "completion_tokens": 1024,
            "produced_tokens": 1024,
            "response": " ".join(f"item{i} clause{i}" for i in range(600)),
            "stream": True,
            "stream_delivery": "incremental",
            "ttft_s": 30.0,
            "prefill_tps": 1115.17,
            "generation_tps": 54.0,
            "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
        }
        trial["prompt_cache"] = benchmark.engine_api.detect_cached_tokens(trial)
        trial["timings"] = {}
        benchmark.evaluate_performance_trial(
            trial,
            required_output_tokens=1024,
            nominal_input_tokens=32768,
            tolerance_percent=2.5,
        )
        self.assertTrue(trial["performance_valid"], trial["invalid_reasons"])
        self.assertFalse(trial["engine_timings_reported"])
        self.assertIn("engine_reports_no_timings", trial["notes"])
        self.assertTrue(trial["prompt_cache"]["no_prompt_cache"])
        self.assertEqual(trial["prompt_cache"]["reported_by"], "usage_prompt_tokens_details.cached_tokens")

    def test_chat_payload_does_not_disable_thinking_or_force_eos(self):
        original = benchmark.request_json
        captured = {}

        def fake_request(_base_url, _path, payload, _api_key, _timeout):
            captured.update(payload)
            return 1.0, {
                "choices": [{"message": {"content": "natural response"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "timings": {},
                "http_status": 200,
            }

        benchmark.request_json = fake_request
        try:
            benchmark.chat_once(
                "http://example.invalid",
                "",
                "model",
                "system",
                "prompt",
                2,
                {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 3407},
                10,
            )
        finally:
            benchmark.request_json = original
        self.assertNotIn("chat_template_kwargs", captured)
        self.assertNotIn("ignore_eos", captured)
        self.assertEqual(captured["temperature"], 1.0)
        self.assertEqual(captured["top_k"], 20)

    def test_repeatability_metadata_accepts_plain_strings(self):
        command = (
            "CUDA_VISIBLE_DEVICES=0 FLASH_ATTN=true "
            "llama-server -m model.gguf -c 131072"
        )
        system = "2x W7900; Ubuntu 24.04; ROCm 6.4.1; llama.cpp b6322"
        args = benchmark.parse_args(
            [
                "--base-url", "http://127.0.0.1:8080",
                "--model", "model-id",
                "--profile", "profile-id",
                "--output", "result.json",
                "--command", command,
                "--system", system,
            ]
        )
        self.assertEqual(args.command, command)
        self.assertEqual(args.system, system)

    def test_recorded_benchmark_command_redacts_api_key(self):
        argv = benchmark.redacted_benchmark_argv(
            ["benchmark.py", "--api-key", "secret-one", "--api-key=secret-two"]
        )
        self.assertEqual(
            argv,
            ["benchmark.py", "--api-key", "<redacted>", "--api-key=<redacted>"],
        )

    def test_result_run_parameters_include_supplied_metadata(self):
        command = "CUDA_VISIBLE_DEVICES=0 llama-server -m model.gguf"
        system = "2x W7900; Ubuntu 24.04; ROCm 6.4.1; llama.cpp b6322"
        args = benchmark.parse_args(
            [
                "--base-url", "http://127.0.0.1:8080",
                "--model", "model-id",
                "--profile", "profile-id",
                "--output", "result.json",
                "--api-key", "do-not-save-this",
                "--command", command,
                "--system", system,
            ]
        )
        metadata = benchmark.build_run_parameters(
            args,
            ["benchmark.py", "--api-key", "do-not-save-this"],
        )
        self.assertEqual(metadata["command"], command)
        self.assertEqual(metadata["system"], system)
        self.assertTrue(metadata["model_load_warmup"])
        self.assertNotIn("do-not-save-this", metadata["benchmark_command"])
        self.assertNotIn("do-not-save-this", json.dumps(metadata["benchmark_argv"]))
        self.assertTrue(metadata["runtime"]["python_version"])

    def test_model_load_warmup_is_nominal_1k_in_and_512_out(self):
        prompt = benchmark.load_model_warmup_prompt()
        self.assertEqual(
            len(prompt),
            benchmark.MODEL_LOAD_WARMUP_INPUT_CHARS,
        )
        self.assertEqual(benchmark.MODEL_LOAD_WARMUP_NOMINAL_INPUT_TOKENS, 1024)
        self.assertEqual(benchmark.MODEL_LOAD_WARMUP_OUTPUT_TOKENS, 512)
        self.assertGreater(len(gzip.compress(prompt.encode())) / len(prompt.encode()), 0.20)

    def test_suite_manifests_are_valid_and_ids_are_unique(self):
        all_ids = []
        for name, path in benchmark.SUITES.items():
            suite = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(suite["schema_version"], 2)
            self.assertEqual(suite["repetitions"], 1)
            self.assertTrue(suite["suite_id"].endswith(f"{name}-v2"))
            ids = [case["id"] for case in suite["cases"]]
            self.assertEqual(len(ids), len(set(ids)))
            all_ids.extend(ids)
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_both_suites_have_locked_long_input_ladder(self):
        for name in ("regular", "coding"):
            suite, _ = benchmark.load_suite(name)
            sizes = [
                case["nominal_input_tokens"]
                for case in suite["cases"]
                if "nominal_input_tokens" in case
            ]
            self.assertEqual(sizes, [8192, 16384, 32768, 65536, 131072])

    def test_performance_outputs_are_fixed_by_profile(self):
        for name, expected in (("regular", 1024), ("coding", 1024)):
            suite, _ = benchmark.load_suite(name)
            values = [
                case["performance_max_tokens"]
                for case in suite["cases"]
                if "nominal_input_tokens" in case
            ]
            self.assertEqual(values, [expected] * 5)

    def test_visible_long_prompt_files_are_locked_and_grow(self):
        for name in ("regular", "coding"):
            suite, _ = benchmark.load_suite(name)
            cases = [case for case in suite["cases"] if "nominal_input_tokens" in case]
            prompts = [benchmark.render_prompt(case) for case in cases]
            self.assertEqual(
                [case["prompt_file"] for case in cases],
                [f"prompts/{name}-{size}.txt" for size in ("8k", "16k", "32k", "64k", "128k")],
            )
            self.assertEqual([len(prompt) for prompt in prompts], sorted(len(prompt) for prompt in prompts))
            for case, prompt in zip(cases, prompts):
                self.assertEqual(benchmark.sha256_bytes(prompt.encode()), case["prompt_sha256"])

    def test_long_prompts_are_not_repetitive_filler(self):
        for path in (PACKAGE_ROOT / "prompts").glob("*.txt"):
            raw = path.read_bytes()
            self.assertGreater(len(gzip.compress(raw, compresslevel=9)) / len(raw), 0.20)
            self.assertLess(path.read_text().count("The quick brown fox jumps over the lazy dog"), 10)

    def test_long_prompts_do_not_contain_obvious_credentials(self):
        pattern = re.compile(
            r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY|password\s*[:=]\s*`[^`]+`|sk-[A-Za-z0-9]{20,}",
            re.IGNORECASE,
        )
        for path in (PACKAGE_ROOT / "prompts").glob("*.txt"):
            self.assertIsNone(pattern.search(path.read_text()), path.name)

    def test_synthetic_fixture_manifest_is_present_and_matches_files(self):
        manifest = json.loads((ROOT / "FIXTURES.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["license"], "MIT")
        self.assertEqual(manifest["seed"], 3407)
        self.assertEqual(len(manifest["prompts"]), 10)
        for record in manifest["prompts"].values():
            path = ROOT / record["path"]
            raw = path.read_bytes()
            self.assertEqual(len(raw), record["bytes"])
            self.assertEqual(benchmark.sha256_bytes(raw), record["sha256"])

    def test_public_fixtures_have_no_local_or_private_identifiers(self):
        forbidden = re.compile(
            r"/(?:Users|home)/|192\.168\.|BEGIN [A-Z ]+PRIVATE KEY|"
            r"(?:api[_-]?key|password|secret)\s*[:=]\s*[^\s]+",
            re.IGNORECASE,
        )
        for path in (PACKAGE_ROOT / "prompts").glob("*.txt"):
            self.assertIsNone(forbidden.search(path.read_text()), path.name)

    def test_performance_lane_requires_fixed_output_and_reports_pp_tg(self):
        original = benchmark.chat_once
        suite, _ = benchmark.load_suite("regular")
        sizes_by_prompt_length = {
            len(benchmark.render_performance_prompt(
                benchmark.render_prompt(case), suite["performance_instruction"]
            )): case["nominal_input_tokens"]
            for case in suite["cases"]
            if "nominal_input_tokens" in case
        }

        def fake_chat(*args, **kwargs):
            max_tokens = args[5]
            prompt_tokens = sizes_by_prompt_length[len(args[4])]
            return {
                "ok": True,
                "elapsed_s": 2.0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens,
                "output_tps_whole_request": max_tokens / 2,
                "prompt_tps_engine": 500.0,
                "generation_tps_engine": 40.0,
                "draft_tokens": None,
                "accepted_draft_tokens": None,
                "produced_tokens": max_tokens,
                "ignore_eos": kwargs.get("ignore_eos", False),
                "request_tag": kwargs.get("request_tag"),
                "timings": {
                    "cache_n": 0,
                    "prompt_n": prompt_tokens,
                    "prompt_ms": prompt_tokens * 2.0,
                    "prompt_per_second": 500.0,
                    "predicted_n": max_tokens,
                    "predicted_ms": (max_tokens - 1) * 25.0,
                    "predicted_per_second": 40.0,
                },
                "finish_reason": "length",
                "response": " ".join(f"token{i}" for i in range(max_tokens)),
                "reasoning_response": "",
                "error": None,
                "http_status": 200,
            }

        benchmark.chat_once = fake_chat
        try:
            result = benchmark.run_suite(
                "regular", "http://example.invalid", "", "model", 10, 1, "performance", ["all"]
            )
        finally:
            benchmark.chat_once = original
        self.assertEqual(result["summary"]["performance_cases"], 5)
        self.assertEqual(result["summary"]["valid_performance_cases"], 5)
        first = result["cases"][0]["performance"]["summary"]
        self.assertTrue(first["all_trials_fixed_length"])
        self.assertTrue(first["all_trials_input_size_valid"])
        self.assertTrue(first["performance_valid"])
        self.assertEqual(first["median_prompt_tps_engine"], 500.0)
        self.assertEqual(first["median_generation_tps_engine"], 40.0)

    def test_size_selector_runs_only_requested_tier(self):
        original = benchmark.chat_once

        def fake_chat(*args, **kwargs):
            max_tokens = args[5]
            return {
                "ok": True,
                "elapsed_s": 1.0,
                "prompt_tokens": 32768,
                "completion_tokens": max_tokens,
                "output_tps_whole_request": float(max_tokens),
                "prompt_tps_engine": 500.0,
                "generation_tps_engine": 40.0,
                "draft_tokens": None,
                "accepted_draft_tokens": None,
                "produced_tokens": max_tokens,
                "ignore_eos": kwargs.get("ignore_eos", False),
                "request_tag": kwargs.get("request_tag"),
                "timings": {
                    "cache_n": 0,
                    "prompt_n": 32768,
                    "prompt_ms": 65536.0,
                    "prompt_per_second": 500.0,
                    "predicted_n": max_tokens,
                    "predicted_ms": (max_tokens - 1) * 25.0,
                    "predicted_per_second": 40.0,
                },
                "finish_reason": "length",
                "response": " ".join(f"token{i}" for i in range(max_tokens)),
                "reasoning_response": "",
                "error": None,
                "http_status": 200,
            }

        benchmark.chat_once = fake_chat
        try:
            result = benchmark.run_suite(
                "coding", "http://example.invalid", "", "model", 10, 1, "performance", ["32k"]
            )
        finally:
            benchmark.chat_once = original
        self.assertEqual(result["sizes"], ["32k"])
        self.assertEqual(len(result["cases"]), 1)
        self.assertEqual(result["cases"][0]["nominal_input_tokens"], 32768)
        self.assertEqual(result["summary"]["valid_performance_cases"], 1)

    def test_size_selector_accepts_multiple_tiers(self):
        original = benchmark.chat_once
        suite, _ = benchmark.load_suite("regular")
        sizes_by_prompt_length = {
            len(benchmark.render_performance_prompt(
                benchmark.render_prompt(case), suite["performance_instruction"]
            )): case["nominal_input_tokens"]
            for case in suite["cases"]
            if "nominal_input_tokens" in case
        }

        def fake_chat(*args, **kwargs):
            max_tokens = args[5]
            prompt_tokens = sizes_by_prompt_length[len(args[4])]
            return {
                "ok": True,
                "elapsed_s": 1.0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens,
                "output_tps_whole_request": float(max_tokens),
                "prompt_tps_engine": 500.0,
                "generation_tps_engine": 40.0,
                "draft_tokens": None,
                "accepted_draft_tokens": None,
                "produced_tokens": max_tokens,
                "ignore_eos": kwargs.get("ignore_eos", False),
                "request_tag": kwargs.get("request_tag"),
                "timings": {
                    "cache_n": 0,
                    "prompt_n": prompt_tokens,
                    "prompt_ms": prompt_tokens * 2.0,
                    "prompt_per_second": 500.0,
                    "predicted_n": max_tokens,
                    "predicted_ms": (max_tokens - 1) * 25.0,
                    "predicted_per_second": 40.0,
                },
                "finish_reason": "length",
                "response": " ".join(f"token{i}" for i in range(max_tokens)),
                "reasoning_response": "",
                "error": None,
                "http_status": 200,
            }

        benchmark.chat_once = fake_chat
        try:
            result = benchmark.run_suite(
                "regular", "http://example.invalid", "", "model", 10, 1,
                "performance", ["16k", "64k", "128k"]
            )
        finally:
            benchmark.chat_once = original
        self.assertEqual(result["sizes"], ["16k", "64k", "128k"])
        self.assertEqual(
            [case["nominal_input_tokens"] for case in result["cases"]],
            [16384, 65536, 131072],
        )

    def test_suite_checkpoints_after_every_performance_part(self):
        original = benchmark.chat_once
        events = []
        requests = []

        def fake_chat(*args, **kwargs):
            requests.append((args, kwargs))
            max_tokens = args[5]
            return {
                "ok": True,
                "elapsed_s": 1.0,
                "prompt_tokens": 16384,
                "completion_tokens": max_tokens,
                "output_tps_whole_request": float(max_tokens),
                "prompt_tps_engine": 500.0,
                "generation_tps_engine": 40.0,
                "draft_tokens": None,
                "accepted_draft_tokens": None,
                "produced_tokens": max_tokens,
                "ignore_eos": kwargs.get("ignore_eos", False),
                "request_tag": kwargs.get("request_tag"),
                "timings": {
                    "cache_n": 0,
                    "prompt_n": 16384,
                    "prompt_ms": 32768.0,
                    "prompt_per_second": 500.0,
                    "predicted_n": max_tokens,
                    "predicted_ms": (max_tokens - 1) * 25.0,
                    "predicted_per_second": 40.0,
                },
                "finish_reason": "length",
                "response": " ".join(f"token{i}" for i in range(max_tokens)),
                "reasoning_response": "",
                "error": None,
                "http_status": 200,
            }

        benchmark.chat_once = fake_chat
        try:
            benchmark.run_suite(
                "regular",
                "http://example.invalid",
                "",
                "model",
                10,
                2,
                "performance",
                ["16k"],
                checkpoint_callback=lambda _result, event: events.append(event),
            )
        finally:
            benchmark.chat_once = original
        self.assertEqual(
            events,
            [
                "regular-05-context-16k:started",
                "regular-05-context-16k:performance:group-01",
                "regular-05-context-16k:performance:group-02",
                "regular-05-context-16k:complete",
                "suite:complete",
            ],
        )
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            [request[1]["request_tag"] for request in requests],
            [
                "regular-05-context-16k-performance-01",
                "regular-05-context-16k-performance-02",
            ],
        )

    def test_atomic_checkpoint_is_always_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            benchmark.write_json_atomic(path, {"status": "running", "trials": [1]})
            self.assertEqual(json.loads(path.read_text()), {"status": "running", "trials": [1]})
            benchmark.write_json_atomic(path, {"status": "complete", "trials": [1, 2]})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete", "trials": [1, 2]})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_exact_json_checker(self):
        passed, _ = benchmark.score_response('{"answer": 42}', {"type": "exact_json", "expected": {"answer": 42}})
        self.assertTrue(passed)
        failed, _ = benchmark.score_response('{"answer": 41}', {"type": "exact_json", "expected": {"answer": 42}})
        self.assertFalse(failed)

    def test_json_rules_checker(self):
        checker = {
            "type": "json_rules",
            "exact_keys": ["owner", "notes"],
            "scalar_exact": {"owner": "Priya"},
            "array_item_terms": {"notes": [["blue", "rollback"]]},
        }
        passed, detail = benchmark.score_response(
            '{"owner":"Priya","notes":["Blue remains the rollback"]}', checker
        )
        self.assertTrue(passed, detail)

    def test_python_checker_accepts_correct_function(self):
        checker = {
            "type": "python_tests",
            "function": "double",
            "tests": [{"args": [4], "expected": 8}, {"args": [-2], "expected": -4}],
        }
        passed, detail = benchmark.score_response("def double(value):\n    return value * 2", checker)
        self.assertTrue(passed, detail)

    def test_python_checker_blocks_imports(self):
        checker = {
            "type": "python_tests",
            "function": "double",
            "tests": [{"args": [4], "expected": 8}],
        }
        passed, detail = benchmark.score_response("import os\ndef double(value):\n    return value * 2", checker)
        self.assertFalse(passed)
        self.assertIn("forbidden", detail)

    def test_sql_checker(self):
        checker = {
            "type": "sqlite_query",
            "setup": ["CREATE TABLE values_table(value INTEGER)", "INSERT INTO values_table VALUES (2), (1)"],
            "expected_columns": ["value"],
            "expected_rows": [[1], [2]],
        }
        passed, detail = benchmark.score_response("SELECT value FROM values_table ORDER BY value", checker)
        self.assertTrue(passed, detail)


if __name__ == "__main__":
    unittest.main()
