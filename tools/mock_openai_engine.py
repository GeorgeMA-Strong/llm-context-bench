#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible mock engine for exercising the benchmark.

The runner only speaks the OpenAI chat-completions HTTP API, so this mock is
enough to test the harness end to end without a GPU: it streams server-sent
events, reports usage, optionally advertises an engine-specific signature
endpoint, and optionally returns a llama.cpp-style ``timings`` object.

Run it and point the benchmark at it:

    python3 tools/mock_openai_engine.py --port 8099 --signature llama.cpp
    python3 -m llm_context_bench.runner \
        --base-url http://127.0.0.1:8099 --model mock-model --profile mock \
        --suite regular --lane performance --sizes 16k \
        --output results/mock.json --command 'mock' --system 'mock'

Token counts are synthesised from prompt length (there is no tokenizer here),
so the mock's throughput numbers are meaningless as engine measurements.  They
exist to prove that the harness measures, validates, and reports correctly.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_MODEL = "mock-model"
PROMETHEUS_TEXT = "# HELP mock_engine_requests_total Mock counter\n# TYPE mock_engine_requests_total counter\nmock_engine_requests_total 1\n"


def completion_text(index: int) -> str:
    """One distinct, low-repetition token chunk per generated token."""
    return f"term{index} "


class MockEngineHandler(BaseHTTPRequestHandler):
    """Serves the handful of endpoints the benchmark touches."""

    protocol_version = "HTTP/1.0"
    server_version = "mock-openai-engine"

    # Class attributes so a test can retune one server without a CLI.
    completion_tokens = 1024
    chars_per_token = 4.1
    prefill_delay_s = 0.05
    decode_delay_s = 0.001
    report_timings = True
    signature = "llama.cpp"

    def log_message(self, *_args) -> None:
        """Keep the console readable while the harness is talking to it."""

    def _respond(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._respond(json.dumps(payload).encode("utf-8"), "application/json", status)

    def _signature_enabled(self, name: str) -> bool:
        return self.signature == name

    def do_GET(self) -> None:  # http.server API
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/v1/models"):
            if path == "/health":
                self._json({"status": "no_slot"})
            else:
                self._json({"object": "list", "data": [{"id": MOCK_MODEL, "object": "model"}]})
        elif path == "/props" and self._signature_enabled("llama.cpp"):
            self._json(
                {
                    "build_info": "mock-b0000",
                    "total_slots": 1,
                    "model_path": "mock.gguf",
                    "use_mlock": False,
                    "webui": True,
                }
            )
        elif path == "/server_info" and self._signature_enabled("vllm"):
            self._json({"vllm_config": {"model_config": {}}, "cache_config": {}})
        elif path == "/get_model_info" and self._signature_enabled("sglang"):
            self._json({"model_path": "mock", "is_generation": True, "tp_size": 1})
        elif path == "/metrics":
            self._respond(PROMETHEUS_TEXT.encode("utf-8"), "text/plain; version=0.0.4")
        else:
            self._json({"error": {"message": f"no route for {path}"}}, 404)

    def _prompt_length(self, payload: dict) -> int:
        total = 0
        for message in payload.get("messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
        return total

    def _usage(self, prompt_tokens: int, completion_tokens: int) -> dict:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # Reported explicitly so the harness can prove it detects reuse.
            "prompt_tokens_details": {"cached_tokens": 0},
        }

    def _timings(self, prompt_tokens: int, completion_tokens: int, prefill_ms: float) -> dict:
        if not self.report_timings:
            return {}
        return {
            "prompt_n": prompt_tokens,
            "prompt_ms": prefill_ms,
            "prompt_per_second": prompt_tokens * 1000.0 / prefill_ms if prefill_ms else 0.0,
            "predicted_n": completion_tokens,
            "predicted_ms": completion_tokens * 20.0,
            "predicted_per_second": 50.0,
            "cache_n": 0,
        }

    def _stream_event(self, delta: dict, finish_reason, usage, timings) -> bytes:
        event = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": MOCK_MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage:
            event["usage"] = usage
        if timings:
            event["timings"] = timings
        return b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"

    def do_POST(self) -> None:  # http.server API
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._json({"error": {"message": f"no route for {path}"}}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": {"message": "invalid JSON body"}}, 400)
            return
        # Unknown engine-specific parameters are accepted on purpose: the
        # harness, not the server, decides which ones to send.
        prompt_tokens = max(1, round(self._prompt_length(payload) / self.chars_per_token))
        completion_tokens = max(1, int(payload.get("max_tokens") or self.completion_tokens))
        started = time.perf_counter()
        if payload.get("stream"):
            self._stream_chat(prompt_tokens, completion_tokens, started)
        else:
            self._json_chat(prompt_tokens, completion_tokens, started)

    def _stream_chat(self, prompt_tokens: int, completion_tokens: int, started: float) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # Servers commonly flush a role-only delta before any token exists.
        self.wfile.write(self._stream_event({"role": "assistant"}, None, None, None))
        self.wfile.flush()
        time.sleep(self.prefill_delay_s)
        prefill_done = time.perf_counter()
        for index in range(completion_tokens):
            if index:
                time.sleep(self.decode_delay_s)
            self.wfile.write(
                self._stream_event({"content": completion_text(index)}, None, None, None)
            )
            self.wfile.flush()
        prefill_ms = (prefill_done - started) * 1000.0
        final = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": MOCK_MODEL,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "length"}
            ],
            "usage": self._usage(prompt_tokens, completion_tokens),
        }
        timings = self._timings(prompt_tokens, completion_tokens, prefill_ms)
        if timings:
            final["timings"] = timings
        self.wfile.write(b"data: " + json.dumps(final).encode("utf-8") + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json_chat(self, prompt_tokens: int, completion_tokens: int, started: float) -> None:
        time.sleep(self.prefill_delay_s + self.decode_delay_s * completion_tokens)
        text = "".join(completion_text(index) for index in range(completion_tokens))
        prefill_ms = self.prefill_delay_s * 1000.0
        self._json(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": MOCK_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "length",
                    }
                ],
                "usage": self._usage(prompt_tokens, completion_tokens),
                "timings": self._timings(prompt_tokens, completion_tokens, prefill_ms),
            }
        )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--completion-tokens", type=int, default=1024, help="Output length used when a request omits max_tokens")
    parser.add_argument("--chars-per-token", type=float, default=4.1)
    parser.add_argument("--prefill-delay", type=float, default=0.05, help="Simulated prefill seconds before the first token")
    parser.add_argument("--decode-delay", type=float, default=0.001, help="Simulated seconds between generated tokens")
    parser.add_argument(
        "--signature",
        choices=("llama.cpp", "vllm", "sglang", "none"),
        default="llama.cpp",
        help="Which engine-specific endpoint the mock answers, for auto-detection tests",
    )
    parser.add_argument("--no-timings", action="store_true", help="Omit the llama.cpp-style timings object")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    MockEngineHandler.completion_tokens = args.completion_tokens
    MockEngineHandler.chars_per_token = args.chars_per_token
    MockEngineHandler.prefill_delay_s = args.prefill_delay
    MockEngineHandler.decode_delay_s = args.decode_delay
    MockEngineHandler.signature = args.signature
    MockEngineHandler.report_timings = not args.no_timings
    server = ThreadingHTTPServer((args.host, args.port), MockEngineHandler)
    print(
        f"mock OpenAI engine on http://{args.host}:{server.server_address[1]} "
        f"(signature={args.signature}, timings={MockEngineHandler.report_timings})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
