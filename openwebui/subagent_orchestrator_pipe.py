"""
title: Subagent Orchestrator
author: brady
version: 1.0.0
description: Main model plans and dispatches sequential subagents (research, backend,
             frontend, api, ...). Each subagent gets tools (run_python, run_shell,
             write_file, read_file, caption_image, share_subagent_context) and its
             thinking/tool trace streams live into the chat inside collapsible blocks.
             Images attached by the user are captioned once with moondream and the
             captions are shared with every agent. Everything runs against a local
             Ollama server.
requirements: requests
"""

import base64
import json
import os
import subprocess
import sys
import requests
from typing import Generator, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (Ollama tool-calling format)
# ─────────────────────────────────────────────────────────────────────────────

def _tool(name, desc, params):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            },
        },
    }


SUBAGENT_TOOLS = [
    _tool(
        "run_python",
        "Execute Python code and return stdout/stderr. Use for computation, file "
        "generation, data processing, or testing code you wrote.",
        {"code": {"type": "string", "description": "Python source code to execute"}},
    ),
    _tool(
        "run_shell",
        "Execute a shell command in the workspace directory and return its output.",
        {"command": {"type": "string", "description": "Shell command to run"}},
    ),
    _tool(
        "write_file",
        "Write content to a file inside the workspace directory (relative path).",
        {
            "path": {"type": "string", "description": "Relative file path, e.g. backend/app.py"},
            "content": {"type": "string", "description": "Full file content"},
        },
    ),
    _tool(
        "read_file",
        "Read a file from the workspace directory (relative path).",
        {"path": {"type": "string", "description": "Relative file path"}},
    ),
    _tool(
        "caption_image",
        "Get a detailed text description of an image file in the workspace "
        "(uses the local vision model).",
        {"path": {"type": "string", "description": "Relative path to an image file"}},
    ),
    _tool(
        "share_subagent_context",
        "Save important findings/results so LATER subagents and the main model can "
        "see them. Call this once, right before you finish, with everything the next "
        "agent needs (key facts, file paths you created, API shapes, decisions).",
        {"context": {"type": "string", "description": "Context to pass forward"}},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Pipe
# ─────────────────────────────────────────────────────────────────────────────

class Pipe:
    class Valves(BaseModel):
        OLLAMA_URL: str = Field(default="http://127.0.0.1:11434")
        MAIN_MODEL: str = Field(
            default="qwen2.5:3b-instruct",
            description="Model that plans, dispatches subagents, and writes the final answer",
        )
        SUBAGENT_MODEL: str = Field(
            default="qwen2.5:3b-instruct",
            description="Model used for each subagent (can differ from MAIN_MODEL)",
        )
        VISION_MODEL: str = Field(default="moondream")
        WORKSPACE_DIR: str = Field(
            default=os.path.expanduser("~/openwebui-workspace"),
            description="Directory where subagents read/write files and run code",
        )
        MAX_SUBAGENTS: int = Field(default=6, description="Hard cap incl. follow-up rounds")
        MAX_TOOL_ITERATIONS: int = Field(
            default=12, description="Max tool-call round-trips per subagent"
        )
        CODE_TIMEOUT: int = Field(default=120, description="Seconds per code/shell execution")
        REQUEST_TIMEOUT: int = Field(default=600)
        ENABLE_CODE_EXECUTION: bool = Field(
            default=True,
            description="Allow run_python / run_shell (executes on THIS machine)",
        )

    def __init__(self):
        self.name = "Subagent Orchestrator"
        self.valves = self.Valves()
        # Per-request state (reset in pipe())
        self.shared_context: List[dict] = []
        self.image_captions: List[str] = []

    # ── ollama plumbing ──────────────────────────────────────────────────────

    def _chat(self, model, messages, tools=None, fmt=None, stream=False):
        payload = {"model": model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        if fmt:
            payload["format"] = fmt
        resp = requests.post(
            f"{self.valves.OLLAMA_URL}/api/chat",
            json=payload,
            timeout=self.valves.REQUEST_TIMEOUT,
            stream=stream,
        )
        resp.raise_for_status()
        return resp

    def _stream_turn(self, model, messages, tools=None):
        """Stream one model turn. Yields ('text', chunk) pieces, then finally
        ('done', {'content': full_text, 'tool_calls': [...]})."""
        full, tool_calls = [], []
        with self._chat(model, messages, tools=tools, stream=True) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                piece = msg.get("content", "")
                if piece:
                    full.append(piece)
                    yield ("text", piece)
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
                if chunk.get("done"):
                    break
        yield ("done", {"content": "".join(full), "tool_calls": tool_calls})

    # ── image handling (same moondream pipeline as the vision router) ────────

    @staticmethod
    def _strip_data_url(url: str) -> Optional[str]:
        if not url:
            return None
        if url.startswith("data:"):
            _, _, b64 = url.partition(",")
            return b64 or None
        if url.startswith("http"):
            # Refuse remote URLs (SSRF risk); Open WebUI sends data: URLs.
            return None
        return url

    def _caption_b64(self, image_b64: str) -> str:
        resp = self._chat(
            self.valves.VISION_MODEL,
            [{
                "role": "user",
                "content": (
                    "Describe this image in thorough detail: objects, people, text, "
                    "layout, colors, and anything notable. Be factual and complete."
                ),
                "images": [image_b64],
            }],
        )
        return resp.json()["message"]["content"].strip()

    def _extract_user_request(self, body: dict):
        """Return (chat_history_as_text, latest_user_text, image_b64_list)."""
        messages = body.get("messages", [])
        history_lines, latest_text, images = [], "", []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            text_parts = []
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        b64 = self._strip_data_url(
                            part.get("image_url", {}).get("url", "")
                        )
                        if b64 and i == len(messages) - 1:
                            images.append(b64)
                text = "\n".join(text_parts)
            else:
                text = content
            if i == len(messages) - 1 and msg.get("role") == "user":
                latest_text = text
            else:
                history_lines.append(f"{msg.get('role', 'user')}: {text}")
        return "\n".join(history_lines[-20:]), latest_text, images

    # ── tool execution ───────────────────────────────────────────────────────

    def _exec_tool(self, name: str, args: dict, agent_label: str) -> str:
        ws = self.valves.WORKSPACE_DIR
        os.makedirs(ws, exist_ok=True)

        def safe_path(rel):
            root = os.path.realpath(ws)
            p = os.path.realpath(os.path.join(root, rel))
            if os.path.commonpath([root, p]) != root:
                raise ValueError("Path escapes workspace directory")
            return p

        try:
            if name == "run_python":
                if not self.valves.ENABLE_CODE_EXECUTION:
                    return "Code execution is disabled in valves."
                proc = subprocess.run(
                    [sys.executable, "-c", args["code"]],
                    capture_output=True, text=True, cwd=ws,
                    timeout=self.valves.CODE_TIMEOUT,
                )
                out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
                return out.strip() or "(no output, exit code %d)" % proc.returncode

            if name == "run_shell":
                if not self.valves.ENABLE_CODE_EXECUTION:
                    return "Code execution is disabled in valves."
                proc = subprocess.run(
                    args["command"], shell=True,
                    capture_output=True, text=True, cwd=ws,
                    timeout=self.valves.CODE_TIMEOUT,
                )
                out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
                return out.strip() or "(no output, exit code %d)" % proc.returncode

            if name == "write_file":
                p = safe_path(args["path"])
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    f.write(args["content"])
                return f"Wrote {len(args['content'])} chars to {args['path']}"

            if name == "read_file":
                with open(safe_path(args["path"])) as f:
                    return f.read()[:20000]

            if name == "caption_image":
                with open(safe_path(args["path"]), "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return self._caption_b64(b64)

            if name == "share_subagent_context":
                self.shared_context.append(
                    {"from": agent_label, "context": args["context"]}
                )
                return "Context saved and will be shown to later agents."

            return f"Unknown tool: {name}"
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {self.valves.CODE_TIMEOUT}s."
        except Exception as e:
            return f"Tool error: {e}"

    # ── subagent runner ──────────────────────────────────────────────────────

    def _shared_context_block(self) -> str:
        if not self.shared_context:
            return "None yet — you are the first agent."
        return "\n\n".join(
            f"--- From {c['from']} ---\n{c['context']}" for c in self.shared_context
        )

    def _image_block(self) -> str:
        if not self.image_captions:
            return ""
        caps = "\n".join(
            f"[Image {i} context: {c}]" for i, c in enumerate(self.image_captions, 1)
        )
        return (
            "\n\nThe user attached image(s). A vision model produced these "
            f"descriptions — treat them as ground truth about the images:\n{caps}"
        )

    def _run_subagent(self, idx, role, task, user_request) -> Generator[str, None, str]:
        """Run one subagent's tool loop, yielding display chunks.
        Returns the agent's final answer text via StopIteration.value."""
        label = f"subagent {idx} ({role})"
        system = (
            f"You are {label}, a focused specialist working as part of a team of "
            "sequential AI agents on the user's request. Do ONLY your assigned task; "
            "other agents handle the rest.\n\n"
            f"YOUR TASK: {task}\n\n"
            f"ORIGINAL USER REQUEST (for context): {user_request}\n\n"
            f"CONTEXT FROM PREVIOUS AGENTS:\n{self._shared_context_block()}"
            f"{self._image_block()}\n\n"
            "You have tools: run_python, run_shell, write_file, read_file, "
            "caption_image, share_subagent_context. Files you write go to a shared "
            "workspace that later agents can read.\n"
            "WORKFLOW: think step by step, use tools as needed, and when done "
            "(1) call share_subagent_context ONCE with the key findings/file paths/"
            "decisions the NEXT agent needs, then (2) write a concise final summary "
            "of what you did as a normal message with no tool calls."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Begin your task now: {task}"},
        ]

        final_text = ""
        for _ in range(self.valves.MAX_TOOL_ITERATIONS):
            result = None
            gen = self._stream_turn(self.valves.SUBAGENT_MODEL, messages, tools=SUBAGENT_TOOLS)
            for kind, payload in gen:
                if kind == "text":
                    yield payload
                else:
                    result = payload
            content, tool_calls = result["content"], result["tool_calls"]

            if not tool_calls:
                final_text = content
                break

            messages.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )
            for tc in tool_calls:
                fn = tc.get("function", {})
                tname = fn.get("name", "?")
                targs = fn.get("arguments", {})
                if isinstance(targs, str):
                    try:
                        targs = json.loads(targs)
                    except json.JSONDecodeError:
                        targs = {}
                arg_preview = json.dumps(targs)[:300]
                yield f"\n\n⚙️ **tool call:** `{tname}({arg_preview})`\n"
                output = self._exec_tool(tname, targs, label)
                shown = output if len(output) <= 1500 else output[:1500] + "\n... (truncated)"
                yield f"```\n{shown}\n```\n"
                # tool_name correlates the result to the call (Ollama's field;
                # ignored by servers that don't use it).
                messages.append(
                    {"role": "tool", "tool_name": tname, "content": output}
                )
        else:
            final_text = "(subagent hit the tool-iteration limit before finishing)"
            yield f"\n\n⚠️ {final_text}\n"

        return final_text

    # ── planning ─────────────────────────────────────────────────────────────

    PLAN_SCHEMA_HINT = (
        'Respond ONLY with JSON in this exact shape:\n'
        '{"needs_subagents": true/false, '
        '"subagents": [{"role": "short-name", "task": "detailed instructions"}]}\n'
        "Rules: use subagents only for multi-part tasks (research + build, multiple "
        "components, etc.). Order matters — earlier agents' findings are passed to "
        "later ones. Typical roles: research, backend, frontend, api, testing, docs. "
        "For simple questions set needs_subagents=false with an empty list."
    )

    def _plan(self, user_request, history) -> List[dict]:
        resp = self._chat(
            self.valves.MAIN_MODEL,
            [
                {"role": "system", "content": "You are an orchestrator that plans AI subagent pipelines.\n" + self.PLAN_SCHEMA_HINT},
                {"role": "user", "content": f"Conversation so far:\n{history or '(none)'}\n\nUser request: {user_request}{self._image_block()}\n\nProduce the plan JSON."},
            ],
            fmt="json",
        )
        try:
            plan = json.loads(resp.json()["message"]["content"])
            if not plan.get("needs_subagents"):
                return []
            subs = plan.get("subagents", [])
            return [
                {"role": s.get("role", f"agent{i+1}"), "task": s.get("task", "")}
                for i, s in enumerate(subs) if s.get("task")
            ][: self.valves.MAX_SUBAGENTS]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _plan_followup(self, user_request) -> List[dict]:
        """After all subagents ran, ask the main model if more are needed."""
        resp = self._chat(
            self.valves.MAIN_MODEL,
            [
                {"role": "system", "content": "You review a finished AI subagent pipeline and decide if MORE subagents are needed to fully satisfy the user's request.\n" + self.PLAN_SCHEMA_HINT},
                {"role": "user", "content": f"User request: {user_request}\n\nWork completed so far:\n{self._shared_context_block()}\n\nIf the request is fully covered, set needs_subagents=false. Otherwise list ONLY the missing subagents."},
            ],
            fmt="json",
        )
        try:
            plan = json.loads(resp.json()["message"]["content"])
            if not plan.get("needs_subagents"):
                return []
            return [
                {"role": s.get("role", "extra"), "task": s.get("task", "")}
                for s in plan.get("subagents", []) if s.get("task")
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    # ── main entry point ─────────────────────────────────────────────────────

    def pipe(self, body: dict, __event_emitter__=None) -> Generator[str, None, None]:
        self.shared_context = []
        self.image_captions = []

        history, user_request, images = self._extract_user_request(body)

        # Shared image pipeline: caption once, every agent sees the captions.
        if images:
            yield f"> 🖼️ Captioning {len(images)} attached image(s) with {self.valves.VISION_MODEL}...\n\n"
            for img in images:
                try:
                    self.image_captions.append(self._caption_b64(img))
                except Exception as e:
                    self.image_captions.append(f"(captioning failed: {e})")

        # 1. Plan
        yield "> 🧠 **Main model** — deciding whether subagents are needed...\n\n"
        queue = self._plan(user_request, history)

        if not queue:
            yield "> ✅ No subagents needed — answering directly.\n\n"
            messages = [
                {"role": "system", "content": "You are a helpful assistant." + self._image_block()},
                {"role": "user", "content": user_request},
            ]
            for kind, payload in self._stream_turn(self.valves.MAIN_MODEL, messages):
                if kind == "text":
                    yield payload
            return

        plan_lines = "\n".join(f"> {i}. **{s['role']}** — {s['task']}" for i, s in enumerate(queue, 1))
        yield f"> 📋 **Subagent plan — {len(queue)} agent(s) queued:**\n{plan_lines}\n\n"

        # 2. Run subagents sequentially (with one follow-up planning round)
        ran = 0
        followup_done = False
        i = 0
        while i < len(queue) and ran < self.valves.MAX_SUBAGENTS:
            sub = queue[i]
            n = ran + 1
            yield f"> 🚀 **Main model** — starting subagent {n}: *{sub['role']}* — {sub['task'][:120]}\n\n"
            yield f"<details>\n<summary>🤖 Subagent {n} ({sub['role']}) — live trace</summary>\n\n"

            runner = self._run_subagent(n, sub["role"], sub["task"], user_request)
            final = ""
            try:
                while True:
                    yield next(runner)
            except StopIteration as stop:
                final = stop.value or ""

            yield "\n\n</details>\n\n"
            yield f"> ✅ Subagent {n} ({sub['role']}) finished.\n\n"

            # Safety net: agents that forget share_subagent_context still pass
            # their final summary forward.
            if final and not any(c["from"] == f"subagent {n} ({sub['role']})" for c in self.shared_context):
                self.shared_context.append(
                    {"from": f"subagent {n} ({sub['role']})", "context": final}
                )

            ran += 1
            i += 1

            # 3. After the initial queue: one round of "do we need more?"
            if i == len(queue) and not followup_done:
                followup_done = True
                yield "> 🧠 **Main model** — deciding whether more subagents are necessary...\n\n"
                extra = self._plan_followup(user_request)
                if extra:
                    room = self.valves.MAX_SUBAGENTS - ran
                    extra = extra[:room]
                    if extra:
                        lines = "\n".join(f"> +{j}. **{s['role']}** — {s['task']}" for j, s in enumerate(extra, 1))
                        yield f"> ➕ **More subagents queued:**\n{lines}\n\n"
                        queue.extend(extra)
                else:
                    yield "> ✅ No further subagents needed.\n\n"

        # 4. Final synthesis
        yield "> 🧩 **Main model** — composing the final answer from subagent results...\n\n"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the lead assistant. A team of subagents just completed "
                    "work on the user's request. Using their results below, deliver "
                    "the complete final answer. Mention files created (with paths) "
                    "and summarize what was built/found. Do not invent work that "
                    "was not done." + self._image_block()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original request: {user_request}\n\n"
                    f"SUBAGENT RESULTS:\n{self._shared_context_block()}\n\n"
                    "Write the final answer for the user."
                ),
            },
        ]
        for kind, payload in self._stream_turn(self.valves.MAIN_MODEL, messages):
            if kind == "text":
                yield payload
