# Dataset Format Guide

## JSONL (recommended)

Each line is a JSON object. The `messages` field follows the OpenAI chat format.

### Text-only example
```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

### Multimodal (image) example
Images can be local paths, URLs, or base64-encoded strings.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text",  "text": "Describe this image."}
      ]
    },
    {
      "role": "assistant",
      "content": "A photo of a golden retriever sitting in a park."
    }
  ]
}
```

### Multi-turn conversation
```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user",      "content": "And Germany?"},
    {"role": "assistant", "content": "Berlin."}
  ]
}
```

## Hugging Face Hub datasets

Pass the dataset ID (e.g. `username/my-dataset`) and choose the `hf_hub` format.
Your dataset must have a column whose values are lists of message dicts (same structure above).

## CSV

Each row is one training sample. The messages column must contain a JSON string
(the same structure as above, but serialised).

```csv
messages
"[{""role"":""user"",""content"":""Hi""},{""role"":""assistant"",""content"":""Hello!""}]"
```

---

## Notes

- The script masks padding tokens in labels (sets them to `-100`) so loss is only
  computed on non-padded tokens.
- Assistant turns are trained on; user/system turns contribute to the input context
  but the loss is averaged over the whole sequence by default (standard SFT).
- For **image inputs**, `qwen-vl-utils` handles PIL Image objects, local file paths,
  URLs, and base64 strings transparently.
