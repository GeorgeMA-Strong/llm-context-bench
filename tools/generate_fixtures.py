#!/usr/bin/env python3
"""Generate the repository's public-safe, deterministic long-context fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "src" / "llm_context_bench" / "prompts"
SUITES = ROOT / "src" / "llm_context_bench" / "suites"
SEED = 3407
TARGET_CHARS = {
    "regular": {"16k": 33_100, "32k": 65_800, "64k": 131_500, "128k": 262_000},
    "coding": {"16k": 28_500, "32k": 56_700, "64k": 112_500, "128k": 223_500},
}

PEOPLE = [
    "Amina", "Bruno", "Chiyo", "Daria", "Elias", "Farah", "Gita", "Hugo",
    "Imani", "Jonas", "Kavya", "Luka", "Mei", "Nadia", "Oren", "Pavel",
    "Rina", "Sofia", "Tomas", "Uma", "Vera", "Wale", "Xenia", "Yara", "Zane",
]
SERVICES = [
    "atlas", "beacon", "cascade", "delta", "ember", "fjord", "grove", "harbor",
    "ion", "juniper", "keystone", "lattice", "meridian", "nova", "orbit", "prairie",
    "quartz", "relay", "summit", "tundra", "uplink", "vector", "willow", "xylem",
]
REGIONS = ["north", "south", "east", "west", "central", "coastal", "alpine", "river"]
VERBS = [
    "validated", "measured", "reconciled", "scheduled", "isolated", "reviewed",
    "deployed", "replayed", "compared", "documented", "restored", "inspected",
]
NOUNS = [
    "checkpoint", "manifest", "queue", "snapshot", "ledger", "gateway", "worker",
    "archive", "policy", "timeline", "dataset", "handoff", "pipeline", "replica",
]
STATES = [
    "queued", "running", "draining", "verified", "deferred", "replayed",
    "accepted", "rejected", "archived", "restored", "observed", "committed",
]


def digest(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def evidence_block(rng: random.Random, index: int, count: int = 18) -> str:
    """Return varied, realistic telemetry rather than compressible prose filler."""
    rows = ["Evidence samples (synthetic):"]
    for sample in range(count):
        stamp = (
            f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}T"
            f"{rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}:"
            f"{rng.randrange(0, 60):02d}.{rng.randrange(0, 1000):03d}Z"
        )
        rows.append(
            f"- {stamp} actor={rng.choice(PEOPLE).lower()} service={rng.choice(SERVICES)} "
            f"region={rng.choice(REGIONS)} state={rng.choice(STATES)} "
            f"count={rng.randrange(1, 9_000_000)} latency_us={rng.randrange(40, 900_000)} "
            f"sample={digest(f'{index}:{sample}:{rng.random()}', 28)}"
        )
    return "\n".join(rows) + "\n"


def test_vector_block(rng: random.Random, index: int, count: int = 18) -> str:
    """Return diverse synthetic test vectors for the generated code bundle."""
    rows = ["# Synthetic boundary and regression vectors:"]
    for sample in range(count):
        key = f"{rng.choice(SERVICES)}-{rng.choice(REGIONS)}-{rng.randrange(10_000, 99_999)}"
        numbers = ", ".join(str(rng.randrange(-50_000, 500_000)) for _ in range(5))
        rows.append(
            f"# vector={index:05d}-{sample:02d} key={key} values=[{numbers}] "
            f"expected={rng.randrange(-900_000, 9_000_000)} checksum={digest(f'{index}:{sample}:{rng.random()}', 28)}"
        )
    return "\n".join(rows) + "\n"


def regular_section(rng: random.Random, index: int) -> str:
    owner = rng.choice(PEOPLE)
    service = rng.choice(SERVICES)
    region = rng.choice(REGIONS)
    ticket = f"SYN-{rng.randrange(10000, 99999)}"
    latency = rng.randrange(18, 980)
    volume = rng.randrange(1_000, 900_000)
    threshold = rng.randrange(70, 99)
    trace = digest(f"regular:{index}:{rng.random()}", 24)
    family = index % 6
    if family == 0:
        body = f"""## Incident review {ticket}

The synthetic {service} service in the {region} region reported a {latency} ms p95 latency
during observation window {index:05d}. {owner} {rng.choice(VERBS)} the request timeline and
confirmed that {volume:,} events crossed the gateway. The first alarm came from the
{rng.choice(NOUNS)} monitor; trace `{trace}` connected it to a delayed replica rather than
packet loss. The response team preserved the original logs, reduced concurrency by
{rng.randrange(2, 17)} percent, and compared three recovery options before changing state.

Decision: keep the canary at {threshold}% confidence until two consecutive checks pass.
Rollback: restore snapshot `snap-{digest(trace, 10)}` if error growth exceeds
{rng.randrange(3, 12)} percent. The owner must record the exact start time, affected shard,
and verification query so the next operator can reproduce the conclusion.
"""
    elif family == 1:
        body = f"""## Operations procedure {ticket}

Purpose: rotate the {service} worker pool without losing queued work. Scope is limited to
the synthetic {region} environment. Before starting, {owner} checks the manifest checksum
`{trace}`, available capacity of {volume:,} units, and alert threshold {threshold}.

1. Freeze new assignments and capture queue depth plus oldest-item age.
2. Drain one worker group, compare counters, then inspect the {rng.choice(NOUNS)} record.
3. Resume at {rng.randrange(10, 45)} percent traffic and observe for {rng.randrange(5, 25)} minutes.
4. Abort if latency exceeds {latency} ms or reconciliation differs by more than
   {rng.randrange(1, 8)} records.

Evidence belongs in case {ticket}; screenshots alone are insufficient because values must
remain searchable and machine-readable. A second reviewer signs the final checkpoint.
"""
    elif family == 2:
        body = f"""## Meeting handoff {ticket}

Participants reviewed the {service} migration for the {region} region. {owner} owns the
next action. The group chose a staged transition because the archive contains {volume:,}
records and the current p95 is {latency} ms. Finance needs a reconciled ledger, Operations
needs a fresh snapshot, and Support needs a customer-safe status paragraph.

The decision is reversible until checkpoint `{trace}` is approved. Open questions include
whether the {rng.choice(NOUNS)} retains ordering during retries and whether the new worker
honors the {threshold}% saturation guard. The next review occurs on synthetic day
{rng.randrange(1, 29):02d} at {rng.randrange(0, 24):02d}:{rng.choice(("00", "15", "30", "45"))} UTC.
"""
    elif family == 3:
        body = f"""## API contract note {ticket}

Endpoint `/v2/{service}/{region}/items` accepts an idempotency key, a sequence number, and
a payload digest. Example digest: `{trace}`. A successful response returns status 202,
the normalized owner `{owner.lower()}`, and a checkpoint cursor. Clients retry 408, 429,
and 503 responses with bounded jitter; they do not retry validation failures.

The server rejects batches above {rng.randrange(100, 900)} items or payloads above
{rng.randrange(2, 24)} MiB. At {threshold}% capacity it advertises a retry window of
{rng.randrange(2, 20)} seconds. Consumers must compare the returned cursor with their local
{rng.choice(NOUNS)} before acknowledging delivery. This contract is synthetic and uses
reserved names only.
"""
    elif family == 4:
        values = ", ".join(str(rng.randrange(10, 9999)) for _ in range(16))
        body = f"""## Capacity experiment {ticket}

Experiment {trace} exercised {service} with {volume:,} synthetic messages in the {region}
region. Sample observations were [{values}]. Median latency was {latency} ms, the acceptance
gate was {threshold}%, and {owner} independently checked the arithmetic. The experiment
changed only batch size; cache policy, model revision, power mode, and background load
remained fixed.

Interpretation distinguishes throughput from complete-request latency. A faster processing
phase does not imply a faster user-visible response when queueing or generation dominates.
Raw counters remain authoritative; summaries must exclude failed, cached, or truncated runs.
"""
    else:
        body = f"""## Policy memorandum {ticket}

Records for {service} are retained for {rng.randrange(14, 180)} synthetic days. {owner} may
approve routine restoration in the {region} region, but deletion requires two reviewers and
a matching manifest. The controlling checksum is `{trace}` and the current inventory is
{volume:,} objects. Exceptions expire after {rng.randrange(1, 12)} days.

Operators must state the observed fact, the applicable rule, and the resulting action.
Inference is labeled explicitly. If the {rng.choice(NOUNS)} is incomplete or the confidence
score falls below {threshold}%, the operator stops and requests new evidence rather than
guessing. Every example here is fictional and contains no production identifiers.
"""
    return (
        f"\n===== SYNTHETIC DOCUMENT {index:05d} =====\n\n{body}\n"
        f"{evidence_block(rng, index)}"
    )


def coding_section(rng: random.Random, index: int) -> str:
    module = f"{rng.choice(SERVICES)}_{index:05d}"
    fn = f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}_{index}"
    limit = rng.randrange(8, 4096)
    salt = digest(f"coding:{index}:{rng.random()}", 12)
    family = index % 5
    if family == 0:
        code = f'''def {fn}(records, limit={limit}):
    """Normalize synthetic records without mutating the caller's list."""
    selected = []
    for position, record in enumerate(records):
        if position >= limit:
            break
        key = str(record.get("key", "")).strip().lower()
        if not key:
            continue
        selected.append({{"key": key, "value": int(record.get("value", 0)), "tag": "{salt}"}})
    return sorted(selected, key=lambda item: (item["key"], item["value"]))
'''
    elif family == 1:
        code = f'''class {module.title().replace("_", "")}Buffer:
    def __init__(self, capacity={limit}):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items = []

    def push(self, item):
        self.items.append(("{salt}", item))
        if len(self.items) > self.capacity:
            return self.items.pop(0)[1]
        return None

    def snapshot(self):
        return [item for _, item in self.items]
'''
    elif family == 2:
        code = f'''def {fn}(base, attempts, ceiling={limit}):
    """Return bounded synthetic retry delays."""
    if attempts <= 0:
        return []
    delays = []
    current = max(0, int(base))
    for attempt in range(attempts):
        jitter = (attempt * {rng.randrange(3, 97)} + {rng.randrange(1, 31)}) % {rng.randrange(7, 53)}
        delays.append(min(ceiling, current + jitter))
        current = min(ceiling, max(1, current * 2))
    return delays
'''
    elif family == 3:
        code = f'''CREATE TABLE {module}_events (
    event_id INTEGER PRIMARY KEY,
    stream_key TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    payload_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(stream_key, sequence_no)
);

-- Synthetic query marker {salt}
SELECT stream_key, COUNT(*) AS event_count, MAX(sequence_no) AS high_watermark
FROM {module}_events
WHERE sequence_no >= {rng.randrange(0, 500)}
GROUP BY stream_key
HAVING COUNT(*) >= {rng.randrange(2, 12)}
ORDER BY event_count DESC, stream_key ASC;
'''
    else:
        code = f'''#!/usr/bin/env sh
set -eu

service_name="{module}"
manifest_id="{salt}"
max_attempts={rng.randrange(2, 9)}

attempt=1
while [ "$attempt" -le "$max_attempts" ]; do
    printf '%s attempt=%s manifest=%s\\n' "$service_name" "$attempt" "$manifest_id"
    attempt=$((attempt + 1))
done
'''
    note = (
        f"\nThe `{module}` excerpt is synthetic. Reviewers should check boundary conditions, "
        f"stable ordering, explicit error behavior, and whether limit {limit} is applied before "
        f"or after normalization. Reference `{salt}` exists only to make this fixture varied.\n"
    )
    return (
        f"\n# ===== SYNTHETIC SOURCE: {module} =====\n\n{code}{note}\n"
        f"{test_vector_block(rng, index)}"
    )


REGULAR_MARKERS = [
    "\n===== AUTHORITATIVE SYNTHETIC RECORD A =====\naccount=AC-771; owner=Inez; tier=gold; renewal=2026-11-02; state=active.\n",
    "\n===== AUTHORITATIVE SYNTHETIC RECORD B =====\naccount=AC-204; owner=Omar; tier=silver; renewal=2027-01-19; state=paused.\n",
    "\n===== AUTHORITATIVE SYNTHETIC RECORD C =====\naccount=AC-990; owner=Keiko; tier=platinum; renewal=2026-09-30; state=active.\n",
]
REGULAR_TASK = (
    "\nTASK: Return only JSON with keys AC-771, AC-204, AC-990 in that order. "
    "Each value must be an object with keys owner, tier, renewal, state copied "
    "exactly from the authoritative synthetic records.\n"
)

CODING_MARKERS = [
    '''\n# ===== AUTHORITATIVE SYNTHETIC FUNCTION A =====
def billing_units(items):
    return sum(quantity * price for quantity, price in items) + 7
''',
    '''\n# ===== AUTHORITATIVE SYNTHETIC FUNCTION B =====
def retry_window(base, ceiling):
    return [min(ceiling, base * factor) for factor in (1, 2, 4, 8)]
''',
    '''\n# ===== AUTHORITATIVE SYNTHETIC FUNCTION C =====
def shard_name(region, number):
    return f"{region.lower()}-{number:03d}"
''',
]
CODING_TASK = (
    "\nTASK: Return only JSON with keys billing_units, retry_window, shard_name in "
    "that order. Values are the results of billing_units([(2, 15), (3, 8)]), "
    "retry_window(3, 10), and shard_name('EU', 7).\n"
)


def build_fixture(profile: str, target: int, seed_offset: int) -> str:
    rng = random.Random(SEED + seed_offset)
    if profile == "regular":
        header = (
            "BENCHMARK PROFILE: SYNTHETIC REGULAR DOCUMENT ANALYSIS\n"
            "All people, systems, identifiers, and events in this fixture are fictional.\n"
        )
        markers, footer, section_fn = REGULAR_MARKERS, REGULAR_TASK, regular_section
    else:
        header = (
            "BENCHMARK PROFILE: SYNTHETIC CODE ANALYSIS\n"
            "All modules, identifiers, and examples in this fixture are generated and fictional.\n"
        )
        markers, footer, section_fn = CODING_MARKERS, CODING_TASK, coding_section

    chunks = [header]
    section_index = 0

    def fill_to(length: int) -> None:
        nonlocal section_index
        while sum(map(len, chunks)) < length:
            section = section_fn(rng, section_index)
            section_index += 1
            remaining = length - sum(map(len, chunks))
            piece = section[:remaining]
            if piece.endswith((" ", "\t")):
                piece = piece[:-1] + "."
            chunks.append(piece)

    for fraction, marker in zip((0.10, 0.50, 0.90), markers):
        fill_to(int(target * fraction))
        chunks.append(marker)
    fill_to(target - len(footer))
    chunks.append(footer)
    result = "".join(chunks)
    if len(result) != target:
        raise AssertionError(f"{profile} fixture is {len(result)} chars, expected {target}")
    return result


def main() -> None:
    PROMPTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "license": "MIT",
        "generator": "tools/generate_fixtures.py",
        "seed": SEED,
        "description": "Deterministic synthetic fixtures; no private or third-party corpus data.",
        "reference_tokenizer": {
            "repository": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "file": "tokenizer.json",
            "scope": "Raw fixture calibration only; the server-reported chat prompt count is authoritative.",
        },
        "prompts": {},
    }
    hashes = {}
    for profile, sizes in TARGET_CHARS.items():
        for ordinal, (size, target) in enumerate(sizes.items()):
            text = build_fixture(profile, target, ordinal * 100 + (0 if profile == "regular" else 1000))
            path = PROMPTS / f"{profile}-{size}.txt"
            path.write_text(text, encoding="utf-8")
            raw = text.encode()
            prompt_hash = hashlib.sha256(raw).hexdigest()
            hashes[f"prompts/{profile}-{size}.txt"] = prompt_hash
            manifest["prompts"][f"{profile}-{size}"] = {
                "path": str(path.relative_to(ROOT)),
                "chars": len(text),
                "bytes": len(raw),
                "nominal_tokens": int(size.removesuffix("k")) * 1024,
                "sha256": prompt_hash,
            }

    for suite_path in SUITES.glob("*-v2.json"):
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        for case in suite["cases"]:
            prompt_file = case.get("prompt_file")
            if prompt_file in hashes:
                case["prompt_sha256"] = hashes[prompt_file]
        suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")

    (ROOT / "FIXTURES.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
