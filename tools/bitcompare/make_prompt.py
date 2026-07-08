#!/usr/bin/env python3
"""Generate the FIXED, deterministic prompt used by the bit-level regression harness.

Deterministic by construction (no RNG, no time). Re-running produces byte-identical
prompt.txt. Length is chosen so that, with the DSV4 tokenizer, the prompt is:
  * >> KT_PREFILL_STREAM_THRESHOLD (512)  -> triggers streaming prefill + inline depool
  * <  chunked_prefill_size (8192)         -> single chunk, avoids the 坑⑯ NSA crash
The exact token count is printed by capture_client.py at capture time and recorded in
the golden JSON, so any drift is visible.
"""
import pathlib

# A fixed technical passage, repeated with an incrementing marker so the text is long,
# deterministic, and not trivially compressible into a single repeated token.
BASE = (
    "The DeepSeek-V4-Flash model runs on a single Ascend 910C accelerator. "
    "It uses native sparse attention with a compressor over a paged key-value cache, "
    "and a mixture-of-experts feed-forward block whose experts are offloaded to host "
    "memory and streamed back on demand. Section {i}: consider a transformer layer that "
    "routes each token to a small subset of its two hundred fifty-six experts, keeping "
    "thirty-two of them resident on the accelerator while the remainder live in system "
    "DRAM as quantized weights. Explain, step by step, how the streaming prefill path "
    "loads all experts for a layer, runs the fused grouped matrix multiply, and then "
    "selects the hottest experts to keep resident for the decode phase that follows. "
)

def build(n_sections: int = 12) -> str:
    parts = ["Answer the following question in detail.\n\n"]
    for i in range(1, n_sections + 1):
        parts.append(BASE.format(i=i))
    parts.append(
        "\n\nNow summarize the streaming-prefill-to-decode handoff in one paragraph."
    )
    return "".join(parts)

if __name__ == "__main__":
    out = pathlib.Path(__file__).with_name("prompt.txt")
    text = build()
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars)")
