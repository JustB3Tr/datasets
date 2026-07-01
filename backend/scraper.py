import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Callable, Awaitable

import httpx

from .generator import RawItem

logger = logging.getLogger(__name__)

_thread_pool = ThreadPoolExecutor(max_workers=4)
_http_semaphore = asyncio.Semaphore(5)

# ── HuggingFace Dataset Sources ───────────────────────────────────────────────

HF_SOURCES: list[dict] = [
    # cybersecurity
    {"dataset_id": "CyberNative-AI/Cybersecurity-Data", "config": None, "split": "train", "domain": "cybersecurity",
     "q_field": "instruction", "a_field": "response", "filter_kw": []},
    # networking / IT via OpenHermes filtered
    {"dataset_id": "teknium/OpenHermes-2.5", "config": None, "split": "train", "domain": "networking",
     "q_field": "conversations", "a_field": None,
     "filter_kw": ["network", "protocol", "TCP", "UDP", "IP", "routing", "firewall", "DNS", "HTTP", "OSI", "subnet", "packet"]},
    # IT / computers
    {"dataset_id": "garage-bAInd/Open-Platypus", "config": None, "split": "train", "domain": "it",
     "q_field": "instruction", "a_field": "output", "filter_kw": []},
    # reasoning
    {"dataset_id": "openai/gsm8k", "config": "main", "split": "train", "domain": "reasoning",
     "q_field": "question", "a_field": "answer", "filter_kw": []},
    {"dataset_id": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "train", "domain": "reasoning",
     "q_field": "question", "a_field": "answerKey", "filter_kw": []},
    # three.js / animations filtered from code alpaca
    {"dataset_id": "sahil2801/CodeAlpaca-20k", "config": None, "split": "train", "domain": "threejs",
     "q_field": "instruction", "a_field": "output",
     "filter_kw": ["three.js", "threejs", "webgl", "three", "3d", "shader", "animation", "canvas", "particle", "geometry"]},
    # CSS / frontend
    {"dataset_id": "theblackcat102/evol-codealpaca-v1", "config": None, "split": "train", "domain": "css",
     "q_field": "instruction", "a_field": "output",
     "filter_kw": ["css", "html", "flexbox", "grid", "responsive", "animation", "stylesheet", "sass", "tailwind"]},
    # backend
    {"dataset_id": "iamtarun/python_code_instructions_18k_alpaca", "config": None, "split": "train", "domain": "backend",
     "q_field": "instruction", "a_field": "output", "filter_kw": []},
    # full-stack / frontend fallback
    {"dataset_id": "TokenBender/code_instructions_122k_alpaca_style", "config": None, "split": "train", "domain": "frontend",
     "q_field": "instruction", "a_field": "output",
     "filter_kw": ["react", "vue", "javascript", "frontend", "component", "api", "fetch", "async"]},
]

# Fallback synthetic items if scraping is insufficient
SYNTHETIC_FALLBACK: list[dict] = [
    {"domain": "cybersecurity", "question": "What is SQL injection?", "answer": "SQL injection is an attack where malicious SQL code is inserted into input fields to manipulate a database. Attackers use it to bypass authentication, extract data, or destroy records. Prevention involves parameterized queries and input validation."},
    {"domain": "networking", "question": "Explain the TCP three-way handshake.", "answer": "TCP establishes connections via SYN, SYN-ACK, ACK. The client sends SYN, server responds with SYN-ACK, client sends ACK. This ensures both parties are ready before data transfer begins."},
    {"domain": "it", "question": "How does Active Directory work?", "answer": "Active Directory is Microsoft's directory service. It stores information about network objects like users, computers, and groups. Domain controllers authenticate users and enforce group policies."},
    {"domain": "reasoning", "question": "Solve: If a train travels 60mph for 2 hours, how far does it go?", "answer": "Distance = speed × time = 60 × 2 = 120 miles. This is a direct application of the distance formula."},
    {"domain": "threejs", "question": "How do you create a rotating cube in Three.js?", "answer": "Create a BoxGeometry, apply a MeshBasicMaterial, combine into a Mesh, add to scene, and rotate in the animation loop using mesh.rotation.x += 0.01."},
    {"domain": "animations", "question": "What is CSS keyframe animation?", "answer": "CSS keyframes define animation states at percentages of duration. Use @keyframes to specify from/to states, then apply via animation property with duration, easing, and iteration count."},
    {"domain": "css", "question": "Explain CSS Flexbox.", "answer": "Flexbox is a 1D layout system. Set display:flex on a container. Children become flex items. Control with justify-content (main axis) and align-items (cross axis). Use flex-wrap for multi-line layouts."},
    {"domain": "frontend", "question": "What is the React useEffect hook?", "answer": "useEffect runs side effects after render. It accepts a callback and dependency array. Empty deps = run once on mount. With deps = run when they change. Return a cleanup function to prevent memory leaks."},
    {"domain": "backend", "question": "What is REST API design?", "answer": "REST APIs use HTTP verbs (GET, POST, PUT, DELETE) on resources identified by URLs. They are stateless, meaning each request contains all needed info. Use proper status codes and consistent JSON response structures."},
    {"domain": "cybersecurity", "question": "What is a buffer overflow attack?", "answer": "A buffer overflow occurs when more data is written to a buffer than it can hold, overwriting adjacent memory. Attackers can overwrite return addresses to redirect execution. Mitigations include stack canaries, ASLR, and DEP/NX."},
]


def _extract_fields(row: dict, source: dict) -> tuple[str, str] | None:
    """Extract question/answer from a dataset row based on field config."""
    q_field = source["q_field"]
    a_field = source["a_field"]

    if q_field == "conversations":
        # OpenHermes format: list of {from, value}
        convs = row.get("conversations", [])
        human = next((c["value"] for c in convs if c.get("from") == "human"), None)
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), None)
        if human and gpt:
            return human, gpt
        return None

    q = row.get(q_field, "")
    if a_field:
        a = row.get(a_field, "")
    else:
        # Try common fallbacks
        a = row.get("output", row.get("response", row.get("answer", "")))

    if isinstance(a, list):
        # ARC answer choices — just use the answerKey label
        a = str(a[0]) if a else ""

    if q and a and len(str(q)) > 10 and len(str(a)) > 10:
        return str(q), str(a)
    return None


def _keyword_match(row: dict, keywords: list[str]) -> bool:
    if not keywords:
        return True
    text = " ".join(str(v) for v in row.values()).lower()
    return any(kw.lower() in text for kw in keywords)


def _load_hf_stream(dataset_id: str, config: str | None, split: str, source: dict, max_items: int) -> list[RawItem]:
    """Synchronous HF streaming call — run in thread pool."""
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(dataset_id, config, split=split, streaming=True, trust_remote_code=False)
        items = []
        for row in ds:
            if len(items) >= max_items:
                break
            if not _keyword_match(row, source.get("filter_kw", [])):
                continue
            extracted = _extract_fields(row, source)
            if extracted:
                q, a = extracted
                items.append(RawItem(
                    question=q[:2000],
                    answer=a[:4000],
                    domain=source["domain"],
                    source=f"huggingface:{dataset_id}",
                ))
        return items
    except Exception as e:
        logger.warning("HF dataset %s failed: %s", dataset_id, e)
        return []


async def stream_hf_dataset(
    source: dict,
    max_items: int,
    progress_cb: Callable[[str], Awaitable[None]] | None = None,
) -> list[RawItem]:
    loop = asyncio.get_running_loop()
    if progress_cb:
        await progress_cb(f"Loading {source['dataset_id']} ({source['domain']})...")
    items = await loop.run_in_executor(
        _thread_pool,
        _load_hf_stream,
        source["dataset_id"],
        source.get("config"),
        source["split"],
        source,
        max_items,
    )
    if progress_cb:
        await progress_cb(f"Got {len(items)} items from {source['dataset_id']}")
    return items


async def fetch_github_dataset(
    url: str,
    domain: str,
    max_items: int,
) -> list[RawItem]:
    """Fetch raw JSONL/JSON from a GitHub URL with retry."""
    items = []
    delay = 2
    async with _http_semaphore:
        async with httpx.AsyncClient(timeout=30) as client:
            for attempt in range(3):
                try:
                    resp = await client.get(url, follow_redirects=True)
                    if resp.status_code in (429, 503):
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    resp.raise_for_status()
                    for line in resp.text.splitlines():
                        if len(items) >= max_items:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            import json
                            row = json.loads(line)
                            q = row.get("instruction", row.get("question", row.get("input", "")))
                            a = row.get("output", row.get("answer", row.get("response", "")))
                            if q and a and len(q) > 10 and len(a) > 10:
                                items.append(RawItem(
                                    question=str(q)[:2000],
                                    answer=str(a)[:4000],
                                    domain=domain,
                                    source=f"github:{url}",
                                ))
                        except Exception:
                            continue
                    break
                except Exception as e:
                    logger.warning("GitHub fetch %s attempt %d failed: %s", url, attempt, e)
                    await asyncio.sleep(delay)
                    delay *= 2
    return items


def get_synthetic_fallback(domain: str, count: int) -> list[RawItem]:
    """Return synthetic fallback items for a domain."""
    pool = [s for s in SYNTHETIC_FALLBACK if s["domain"] == domain]
    if not pool:
        pool = SYNTHETIC_FALLBACK
    import random
    items = []
    for i in range(count):
        s = pool[i % len(pool)]
        items.append(RawItem(
            question=s["question"],
            answer=s["answer"],
            domain=s["domain"],
            source="synthetic:fallback",
        ))
    return items


async def gather_raw_items(
    total: int,
    domains: list[str] | None,
    progress_cb: Callable[[str], Awaitable[None]] | None = None,
) -> list[RawItem]:
    """Gather raw items from all sources, balanced across domains."""
    active_sources = HF_SOURCES
    if domains:
        active_sources = [s for s in HF_SOURCES if s["domain"] in domains]
    if not active_sources:
        active_sources = HF_SOURCES

    per_source = max(10, (total // len(active_sources)) + 5)
    tasks = [stream_hf_dataset(src, per_source, progress_cb) for src in active_sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: list[RawItem] = []
    domain_counts: dict[str, int] = {}
    for result in results:
        if isinstance(result, list):
            all_items.extend(result)
            for item in result:
                domain_counts[item.domain] = domain_counts.get(item.domain, 0) + 1

    # Fill gaps with synthetic fallback
    target_domains = domains or list({s["domain"] for s in HF_SOURCES})
    for d in target_domains:
        if domain_counts.get(d, 0) < 3:
            fallback = get_synthetic_fallback(d, 5)
            all_items.extend(fallback)
            if progress_cb:
                await progress_cb(f"Using synthetic fallback for domain: {d}")

    import random
    random.shuffle(all_items)
    return all_items
