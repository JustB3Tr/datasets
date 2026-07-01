"""
title: Vision Router (moondream -> qwen2.5:3b-instruct)
author: brady
version: 1.0.0
description: If the user message contains an image, caption it with moondream via
             the local Ollama API, inject the caption as text context, and answer
             with the text-only model. Text-only messages go straight to the text model.
requirements: requests
"""

import json
import requests
from typing import Generator, List, Optional
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        OLLAMA_URL: str = Field(
            default="http://127.0.0.1:11434",
            description="Base URL of the local Ollama server",
        )
        VISION_MODEL: str = Field(
            default="moondream",
            description="Ollama model used to caption images",
        )
        TEXT_MODEL: str = Field(
            default="qwen2.5:3b-instruct",
            description="Ollama model that produces the final answer",
        )
        CAPTION_PROMPT: str = Field(
            default=(
                "Describe this image in thorough detail: objects, people, text, "
                "layout, colors, and anything notable. Be factual and complete."
            ),
            description="Prompt sent to the vision model for each image",
        )
        REQUEST_TIMEOUT: int = Field(default=600, description="HTTP timeout (seconds)")

    def __init__(self):
        self.name = "Vision Router"
        self.valves = self.Valves()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_data_url(url: str) -> Optional[str]:
        """Return raw base64 from a data: URL (or the string itself if already raw)."""
        if not url:
            return None
        if url.startswith("data:"):
            _, _, b64 = url.partition(",")
            return b64 or None
        if url.startswith("http"):
            # Remote image: fetch and re-encode
            import base64
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                return base64.b64encode(resp.content).decode()
            except Exception:
                return None
        return url  # assume raw base64

    def _extract_text_and_images(self, message: dict):
        """Open WebUI content is either a plain string or a list of typed parts."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content, []
        text_parts, images = [], []
        for part in content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                b64 = self._strip_data_url(part.get("image_url", {}).get("url", ""))
                if b64:
                    images.append(b64)
        return "\n".join(text_parts), images

    def _caption_image(self, image_b64: str) -> str:
        """One blocking call to the vision model for a single image."""
        resp = requests.post(
            f"{self.valves.OLLAMA_URL}/api/chat",
            json={
                "model": self.valves.VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": self.valves.CAPTION_PROMPT,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
            },
            timeout=self.valves.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _stream_text_model(self, messages: List[dict]) -> Generator[str, None, None]:
        """Stream the text model's answer chunk by chunk."""
        with requests.post(
            f"{self.valves.OLLAMA_URL}/api/chat",
            json={"model": self.valves.TEXT_MODEL, "messages": messages, "stream": True},
            timeout=self.valves.REQUEST_TIMEOUT,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break

    # ── main entry point ─────────────────────────────────────────────────────

    def pipe(self, body: dict, __event_emitter__=None) -> Generator[str, None, None]:
        incoming = body.get("messages", [])

        # Rebuild history as text-only; caption images wherever they appear.
        out_messages: List[dict] = []
        pending_images: List[str] = []  # images on the LAST user turn

        for i, msg in enumerate(incoming):
            text, images = self._extract_text_and_images(msg)
            is_last_user = (i == len(incoming) - 1) and msg.get("role") == "user"
            if images and is_last_user:
                pending_images = images
            elif images:
                # Older turns: caption inline so history stays coherent.
                captions = []
                for img in images:
                    try:
                        captions.append(self._caption_image(img))
                    except Exception as e:
                        captions.append(f"(image captioning failed: {e})")
                cap_block = "\n".join(f"[Image context: {c}]" for c in captions)
                text = f"{cap_block}\n\n{text}" if text else cap_block
            out_messages.append({"role": msg.get("role", "user"), "content": text})

        # Caption image(s) on the current turn, with a visible status line.
        if pending_images:
            yield f"> 🖼️ Detected {len(pending_images)} image(s) — captioning with {self.valves.VISION_MODEL}...\n\n"
            captions = []
            for idx, img in enumerate(pending_images, 1):
                try:
                    captions.append(self._caption_image(img))
                except Exception as e:
                    captions.append(f"(captioning failed: {e})")
            cap_block = "\n".join(
                f"[Image {i} context: {c}]" for i, c in enumerate(captions, 1)
            )
            last = out_messages[-1]
            last["content"] = (
                f"{cap_block}\n\n{last['content']}" if last["content"] else cap_block
            )

        # System note so the text model knows captions are stand-ins for real images.
        system_note = {
            "role": "system",
            "content": (
                "You are a helpful assistant. When a message includes '[Image context: ...]' "
                "blocks, those are detailed captions of images the user attached, generated "
                "by a vision model. Treat them as if you saw the image yourself; answer the "
                "user's question about the image using that context. Do not mention the "
                "captioning process."
            ),
        }
        if not out_messages or out_messages[0]["role"] != "system":
            out_messages.insert(0, system_note)

        yield from self._stream_text_model(out_messages)
