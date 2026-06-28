import asyncio
import json
import logging
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .eval import run_eval
from .generator import generate_example
from .scraper import gather_raw_items

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Qwen Dataset Generator")

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ── Models ────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    SCRAPING = "scraping"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    count: int = Field(default=500, ge=1, le=10000)
    domains: list[str] | None = None


class SSEEvent(BaseModel):
    type: str
    data: Any

    def to_sse(self) -> dict:
        return {"event": self.type, "data": json.dumps(self.data)}


class Job:
    def __init__(self, job_id: str, count: int, domains: list[str] | None):
        self.job_id = job_id
        self.count = count
        self.domains = domains
        self.status = JobStatus.PENDING
        self.queue: asyncio.Queue = asyncio.Queue()
        self.output_path: Path | None = None
        self.done = 0
        self.task: asyncio.Task | None = None

    async def emit(self, event_type: str, data: Any):
        await self.queue.put(SSEEvent(type=event_type, data=data))

    async def log(self, msg: str, level: str = "info"):
        await self.emit("log", {"msg": msg, "level": level})
        if level == "error":
            logger.error(msg)
        else:
            logger.info(msg)


jobs: dict[str, Job] = {}


# ── Generation Worker ─────────────────────────────────────────────────────────

BATCH_SIZE = 20


async def run_generation(job: Job):
    try:
        job.status = JobStatus.SCRAPING
        await job.log(f"Starting dataset generation: {job.count} examples")
        await job.emit("progress", {"done": 0, "total": job.count, "pct": 0.0, "status": "scraping"})

        async def progress_cb(msg: str):
            await job.log(msg)

        # Gather more raw items than needed to account for filtering losses
        fetch_count = max(job.count * 3, 100)
        raw_items = await gather_raw_items(fetch_count, job.domains, progress_cb)
        await job.log(f"Scraped {len(raw_items)} raw items from sources")

        if not raw_items:
            await job.emit("error", {"msg": "No raw items could be scraped from any source"})
            job.status = JobStatus.FAILED
            await job.queue.put(None)
            return

        job.status = JobStatus.GENERATING
        output_path = OUTPUT_DIR / f"{job.job_id}.jsonl"
        job.output_path = output_path

        generated = 0
        raw_idx = 0

        seen_questions: set[str] = set()

        async with aiofiles.open(output_path, "w") as f:
            while generated < job.count and raw_idx < len(raw_items):
                batch_raw = raw_items[raw_idx:raw_idx + BATCH_SIZE]
                raw_idx += BATCH_SIZE

                # Generate examples from batch
                batch_results = []
                for raw in batch_raw:
                    if generated + len(batch_results) >= job.count:
                        break
                    example = generate_example(raw)
                    if not example:
                        continue
                    # Dedup by normalized user question
                    user_q = next(
                        (m["content"].strip().lower() for m in example["messages"] if m["role"] == "user"),
                        "",
                    )
                    if user_q in seen_questions:
                        continue
                    seen_questions.add(user_q)
                    batch_results.append(example)

                # Write batch
                for example in batch_results:
                    await f.write(json.dumps(example) + "\n")
                    generated += 1

                pct = round(generated / job.count * 100, 1)
                domain = batch_raw[-1].domain if batch_raw else "unknown"
                await job.emit("progress", {
                    "done": generated,
                    "total": job.count,
                    "pct": pct,
                    "domain": domain,
                    "status": "generating",
                })
                await job.log(f"Generated {generated}/{job.count} examples ({pct}%)")

                # Yield control to event loop
                await asyncio.sleep(0)

        # If we ran out of raw items before reaching target, pad with synthetics
        if generated < job.count:
            await job.log(f"Only generated {generated}/{job.count} — raw source exhausted", "warn")

        await job.log(f"Generation complete: {generated} examples written")

        # ── Eval ──────────────────────────────────────────────────────────────
        job.status = JobStatus.EVALUATING
        await job.emit("progress", {"done": generated, "total": job.count, "pct": 100.0, "status": "evaluating"})
        await job.log("Running smoke test eval on first 100 examples...")

        report = run_eval(str(output_path), sample_size=100)
        report_dict = report.model_dump()

        if report.passed:
            await job.log(f"Eval PASSED — {report.total} examples checked, all {len(report.checks_run)} checks passed")
            await job.emit("complete", {
                "file": f"{job.job_id}.jsonl",
                "generated": generated,
                "eval": report_dict,
            })
        else:
            await job.log(f"Eval completed with issues: {report.failed_checks}", "warn")
            await job.emit("eval_failed", {
                "file": f"{job.job_id}.jsonl",
                "generated": generated,
                "eval": report_dict,
            })

        job.status = JobStatus.COMPLETE

    except Exception as e:
        logger.exception("Generation job %s failed", job.job_id)
        await job.emit("error", {"msg": str(e)})
        job.status = JobStatus.FAILED
    finally:
        await job.queue.put(None)  # sentinel


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/generate")
async def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job = Job(job_id, req.count, req.domains)
    jobs[job_id] = job
    job.task = asyncio.create_task(run_generation(job))
    return {"job_id": job_id}


@app.get("/progress/{job_id}")
async def progress(job_id: str, request: Request):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    async def event_generator():
        # Send current status immediately
        yield {
            "event": "status",
            "data": json.dumps({"status": job.status, "done": job.done, "total": job.count}),
        }

        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(job.queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}
                continue

            if event is None:
                break
            yield event.to_sse()

    return EventSourceResponse(event_generator())


@app.get("/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.status not in (JobStatus.COMPLETE, JobStatus.EVALUATING):
        raise HTTPException(status_code=400, detail=f"Job not complete (status: {job.status})")
    if not job.output_path or not job.output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(
        path=str(job.output_path),
        filename=f"qwen_dataset_{job_id[:8]}.jsonl",
        media_type="application/x-ndjson",
    )


@app.get("/jobs")
async def list_jobs():
    return [
        {"job_id": jid, "status": j.status, "count": j.count, "done": j.done}
        for jid, j in jobs.items()
    ]


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
