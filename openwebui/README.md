# Open WebUI Pipes — Vision Router & Subagent Orchestrator

Two Open WebUI **Pipe functions** that run entirely against your local Ollama
server (`localhost:11434` / `127.0.0.1:11434`).

| File | What it does |
|---|---|
| `vision_router_pipe.py` | If the message has an image → caption it with `moondream` → inject caption → answer with `qwen2.5:3b-instruct`. Text-only messages go straight through. |
| `subagent_orchestrator_pipe.py` | Main model plans a pipeline of sequential subagents (research → backend → frontend → ...), each with code-execution tools and a live trace streamed into the chat. Uses the same moondream image pipeline so captions are shared across all agents. |
| `system_prompt.md` | System prompt for the orchestrator's main model. |

## Prerequisites

```bash
ollama pull moondream
ollama pull qwen2.5:3b-instruct   # or your fine-tuned model once merged + converted
ollama serve                       # usually already running as a service
```

## Install (both pipes)

1. Open WebUI → **Admin Panel → Functions → + (New Function)**
2. Paste the contents of the `.py` file, save, and **enable** it.
3. The pipe now appears in the model dropdown like a regular model
   ("Vision Router" / "Subagent Orchestrator").

## Configure (Valves)

Click the gear icon on the function to set valves:

- `OLLAMA_URL` — default `http://127.0.0.1:11434`. **If Open WebUI runs in
  Docker**, the host's Ollama is NOT at 127.0.0.1 from inside the container —
  use `http://host.docker.internal:11434`.
- `MAIN_MODEL` / `SUBAGENT_MODEL` / `TEXT_MODEL` — swap in your fine-tuned
  model's Ollama tag once you've merged the LoRA adapter and created it with
  `ollama create`.
- `WORKSPACE_DIR` — where subagents write files and execute code
  (default `~/openwebui-workspace`).
- `ENABLE_CODE_EXECUTION` — set to `false` to disable `run_python`/`run_shell`.

> ⚠️ **Security note:** the orchestrator executes model-written Python and
> shell commands on the machine running Open WebUI, inside `WORKSPACE_DIR`.
> That is the point (it builds and runs code for you), but only use it on a
> machine/setup where you're comfortable with that, and keep
> `ENABLE_CODE_EXECUTION` off when you don't need it.

## Using the orchestrator

Select **Subagent Orchestrator** in the model dropdown, set the system prompt
from `system_prompt.md` on the main model, and just chat. For multi-part
requests you'll see the flow stream live:

```
> 🧠 Main model — deciding whether subagents are needed...
> 📋 Subagent plan — 3 agent(s) queued:
> 1. research — ...
> 2. backend — ...
> 3. frontend — ...
> 🚀 Main model — starting subagent 1: research ...
  ▸ Subagent 1 (research) — live trace   ← collapsible, streams thinking + tool calls
> ✅ Subagent 1 (research) finished.
...
> 🧠 Main model — deciding whether more subagents are necessary...
> 🧩 Main model — composing the final answer from subagent results...
```

Each subagent passes its findings forward via the `share_subagent_context`
tool, so later agents see everything earlier agents learned/built. Attached
images are captioned once by moondream and the captions are visible to the
main model and every subagent.

## Notes / limits

- `qwen2.5:3b-instruct` supports Ollama tool calling, but a 3B model is a weak
  planner — expect occasional empty plans (falls back to a direct answer) or
  agents that skip `share_subagent_context` (the pipe auto-forwards their final
  summary as a safety net). A 7B+ model as `MAIN_MODEL` plans noticeably better.
- Subagents run sequentially by design so context can flow forward.
- One follow-up planning round runs after the initial queue; `MAX_SUBAGENTS`
  caps the total.
