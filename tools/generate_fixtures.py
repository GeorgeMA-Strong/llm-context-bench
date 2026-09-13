#!/usr/bin/env python3
"""Deterministically build (or verify) the long-context fixtures.

The fixtures assemble real third-party content from pinned public sources:

- **regular** profile: public-domain documents from Project Gutenberg
  (Dickens, Melville, Darwin, and a 1900s legal manual).
- **coding** profile: source files from ``microsoft/vscode``
  (dual-licensed MIT/Apache-2.0), pinned at commit ``VSCODE_SHA``.

Each fixture also embeds three small benchmark reference blocks
("AUTHORITATIVE" needles) and a ``TASK:`` footer per profile. The sources
are named inside each fixture, and the per-file SHA-256 hashes are locked in
the suite manifests and [FIXTURES.json](../FIXTURES.json).

Usage::

    python3 -m pip install tokenizers
    python3 tools/generate_fixtures.py           # verify committed fixtures
    python3 tools/generate_fixtures.py --write   # rebuild + refresh hashes

Build-time dependency: ``tokenizers`` (not needed at runtime). Downloads are
cached under ``tools/.corpus_cache/`` (git-ignored). Output is byte-identical
across runs for the pinned sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "llm_context_bench"
PROMPTS = PACKAGE / "prompts"
SUITES = PACKAGE / "suites"
MANIFEST = ROOT / "FIXTURES.json"
CACHE = ROOT / "tools" / ".corpus_cache"

# Pinned calibration tokenizer (raw-fixture calibration only; the
# server-reported chat prompt count is authoritative at run time).
TOKENIZER_URL = (
    "https://huggingface.co/Qwen/Qwen3-8B/resolve/"
    "b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json"
)

# Pinned source revision for the coding corpus.
VSCODE_SHA = "08d4889f9ec4a1685d257b9b95de036c8e1ce1e5"
VSCODE_FILES = ROOT / "tools" / "corpus" / "vscode-files.json"

# (title, author, gutenberg id, pinned file URL) — order is significant.
BOOKS = [
    (
        "A Tale of Two Cities",
        "Charles Dickens",
        98,
        "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    ),
    (
        "Moby-Dick; or, The Whale",
        "Herman Melville",
        2701,
        "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    ),
    (
        "On the Origin of Species",
        "Charles Darwin",
        1228,
        "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
    ),
    (
        "Putnam's Handy Law Book for the Layman",
        "Albert Sidney Bolles",
        33088,
        "https://www.gutenberg.org/cache/epub/33088/pg33088.txt",
    ),
]

NOMINALS = [8_192, 16_384, 32_768, 65_536, 131_072]
SIZES = ["8k", "16k", "32k", "64k", "128k"]

REG_HEADER = (
    "BENCHMARK PROFILE: REGULAR DOCUMENT ANALYSIS\n"
    "This fixture bundles public-domain documents for long-context benchmarking.\n"
    "All text is original source material, except the three blocks marked\n"
    "AUTHORITATIVE, which are benchmark reference records.\n"
)
REG_RECORDS = [
    "\n===== AUTHORITATIVE SYNTHETIC RECORD A =====\naccount=AC-771; owner=Inez; tier=gold; renewal=2026-11-02; state=active.\n",
    "\n===== AUTHORITATIVE SYNTHETIC RECORD B =====\naccount=AC-204; owner=Omar; tier=silver; renewal=2027-01-19; state=paused.\n",
    "\n===== AUTHORITATIVE SYNTHETIC RECORD C =====\naccount=AC-990; owner=Keiko; tier=platinum; renewal=2026-09-30; state=active.\n",
]
REG_TASK = (
    "\nTASK: Return only JSON with keys AC-771, AC-204, AC-990 in that order. "
    "Each value must be an object with keys owner, tier, renewal, state copied "
    "exactly from the authoritative synthetic records.\n"
)

COD_HEADER = (
    "BENCHMARK PROFILE: CODE ANALYSIS\n"
    "This fixture bundles source files from the open-source project\n"
    "microsoft/vscode (dual-licensed MIT/Apache-2.0), pinned at commit\n"
    f"{VSCODE_SHA} (tag 1.135.0).\n"
    "All text is original project source, except the three blocks marked\n"
    "AUTHORITATIVE, which are benchmark reference functions.\n"
)
COD_RECORDS = [
    "\n# ===== AUTHORITATIVE SYNTHETIC FUNCTION A =====\ndef billing_units(items):\n    return sum(quantity * price for quantity, price in items) + 7\n",
    "\n# ===== AUTHORITATIVE SYNTHETIC FUNCTION B =====\ndef retry_window(base, ceiling):\n    return [min(ceiling, base * factor) for factor in (1, 2, 4, 8)]\n",
    "\n# ===== AUTHORITATIVE SYNTHETIC FUNCTION C =====\ndef shard_name(region, number):\n    return f\"{region.lower()}-{number:03d}\"\n",
]
COD_TASK = (
    "\nTASK: Return only JSON with keys billing_units, retry_window, shard_name in "
    "that order. Values are the results of billing_units([(2, 15), (3, 8)]), "
    "retry_window(3, 10), and shard_name('EU', 7).\n"
)

# Same gates the unit tests enforce on the committed fixtures.
CRED_RE = re.compile(
    r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY|password\s*[:=]\s*`[^`]+`|sk-[A-Za-z0-9]{20,}",
    re.IGNORECASE,
)
PRIVATE_RE = re.compile(
    r"/(?:Users|home)/|192\.168\.|(?:api[_-]?key|password|secret)\s*[:=]\s*[^\s]+",
    re.IGNORECASE,
)
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog"


def fetch(url: str, name: str) -> Path:
    """Download ``url`` into the git-ignored cache, reusing an existing copy."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if not dest.exists() or dest.stat().st_size == 0:
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
            tmp.write_bytes(resp.read())
        tmp.rename(dest)
    return dest


def clean_book(raw: str) -> str:
    start = re.search(r"\*{3} START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n", raw)
    end = re.search(r"\*{3} END OF THE PROJECT GUTENBERG EBOOK[^\n]*", raw)
    if not start or not end:
        raise ValueError("Gutenberg file is missing the standard START/END markers")
    return raw[start.end() : end.start()].strip() + "\n"


def regular_corpus() -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    for title, author, gid, url in BOOKS:
        raw = fetch(url, f"gutenberg-{gid}.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        sep = (
            f"\n\n===== SOURCE DOCUMENT: {title} — {author} "
            f"(public domain; Project Gutenberg #{gid}) =====\n\n"
        )
        abs_start = sum(len(p) for p in parts)
        parts.append(sep + clean_book(raw))
        spans.append((abs_start, abs_start + len(sep)))
    return "".join(parts), spans


def coding_corpus() -> tuple[str, list[tuple[int, int]]]:
    paths = json.loads(VSCODE_FILES.read_text(encoding="utf-8"))
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, path in enumerate(paths):
        url = f"https://raw.githubusercontent.com/microsoft/vscode/{VSCODE_SHA}/{path}"
        raw = fetch(url, f"vscode-{index:03d}-{path.rsplit('/', 1)[-1]}").read_text(
            encoding="utf-8", errors="replace"
        )
        if CRED_RE.search(raw) or PRIVATE_RE.search(raw) or raw.count(FILLER_SENTENCE) >= 10:
            raise ValueError(f"pinned source file failed content gates: {path}")
        text = raw.rstrip("\n") + "\n"
        sep = f"\n# ===== SOURCE FILE: {path} (microsoft/vscode@{VSCODE_SHA[:9]}) =====\n\n"
        abs_start = sum(len(p) for p in parts)
        parts.append(sep + text)
        spans.append((abs_start, abs_start + len(sep)))
    return "".join(parts), spans


def assemble(
    tokenizer,
    header: str,
    corpus: str,
    sep_spans: list[tuple[int, int]],
    records: list[str],
    task: str,
    nominal: int,
) -> str:
    """Place the three needles at ~10/50/90% of the nominal token budget."""
    enc = tokenizer.encode(corpus, add_special_tokens=False)
    header_tokens = len(tokenizer.encode(header, add_special_tokens=False).ids)
    record_tokens = [len(tokenizer.encode(r, add_special_tokens=False).ids) for r in records]
    task_tokens = len(tokenizer.encode(task, add_special_tokens=False).ids)

    span_tokens = [
        int(0.10 * nominal) - header_tokens,
        int(0.40 * nominal) - record_tokens[0],
        int(0.40 * nominal) - record_tokens[1],
        int(0.10 * nominal) - record_tokens[2] - task_tokens,
    ]

    def char_at_token(k: int) -> int:
        if k < 0:
            return 0
        if k >= len(enc.offsets):
            return len(corpus)
        return enc.offsets[k][0]

    def snap_back(pos: int) -> int:
        for a, b in sep_spans:  # never cut inside a source separator
            if a < pos < b:
                pos = b
                break
        if pos <= 0:
            return 0
        space = corpus.rfind(" ", 0, pos)
        return space + 1 if space > 0 else pos

    cuts: list[int] = []
    absolute = 0
    for span in span_tokens:
        absolute += span
        cuts.append(snap_back(char_at_token(absolute)))
    c1, c2, c3, c4 = cuts
    return (
        header
        + corpus[:c1]
        + records[0]
        + corpus[c1:c2]
        + records[1]
        + corpus[c2:c3]
        + records[2]
        + corpus[c3:c4]
        + task
    )


def build_fixtures(tokenizer) -> dict[str, str]:
    reg_corpus, reg_spans = regular_corpus()
    cod_corpus, cod_spans = coding_corpus()
    out: dict[str, str] = {}
    for size, nominal in zip(SIZES, NOMINALS):
        out[f"regular-{size}"] = assemble(
            tokenizer, REG_HEADER, reg_corpus, reg_spans, REG_RECORDS, REG_TASK, nominal
        )
        out[f"coding-{size}"] = assemble(
            tokenizer, COD_HEADER, cod_corpus, cod_spans, COD_RECORDS, COD_TASK, nominal
        )
    for name, text in out.items():
        if CRED_RE.search(text) or PRIVATE_RE.search(text):
            raise ValueError(f"fixture {name} failed content gates")
        if text.count(FILLER_SENTENCE) >= 10:
            raise ValueError(f"fixture {name} is repetitive filler")
    return out


def refresh_hashes(names: list[str]) -> None:
    for name in ("regular", "coding"):
        path = SUITES / f"{name}-v2.json"
        suite = json.loads(path.read_text(encoding="utf-8"))
        for case in suite["cases"]:
            prompt_file = case.get("prompt_file")
            if prompt_file:
                # prompt_file is stored relative to the package root.
                case["prompt_sha256"] = hashlib.sha256(
                    (PACKAGE / prompt_file).read_bytes()
                ).hexdigest()
        path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key in names:
        record = manifest["prompts"][key]
        raw = (ROOT / record["path"]).read_bytes()
        record["chars"] = len(raw.decode("utf-8"))
        record["bytes"] = len(raw)
        record["sha256"] = hashlib.sha256(raw).hexdigest()
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write rebuilt fixtures and refresh suite/manifest hashes",
    )
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        str(fetch(TOKENIZER_URL, "qwen3-8b-tokenizer.json"))
    )
    built = build_fixtures(tokenizer)

    if args.write:
        for name, text in built.items():
            (PROMPTS / f"{name}.txt").write_text(text, encoding="utf-8")
        refresh_hashes(list(built))
        print(f"wrote {len(built)} fixtures and refreshed hashes")
        return 0

    mismatch = 0
    for name, text in built.items():
        path = PROMPTS / f"{name}.txt"
        if not path.exists():
            print(f"MISMATCH {name}: missing")
            mismatch += 1
            continue
        if path.read_bytes() != text.encode("utf-8"):
            print(f"MISMATCH {name}: committed file differs from deterministic build")
            mismatch += 1
    if mismatch:
        print(f"{mismatch} fixture(s) out of sync; rerun with --write to rebuild")
        return 1
    print(f"all {len(built)} fixtures match the deterministic build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
