---
topic: What "on par with llama.cpp, nothing missing" actually means — the checklist that decides when v0.0.X LTS ships
status: proposed, awaiting Atur
links: [lts-0-0-0.md, the-big-bang.md, ../research/the-plateau-was-ours-2026-08-10.md]
---

The goal, in Atur's words: **standards-compliant, a person can easily open any
model, best performance against llama.cpp on the criteria, all of its options
and capabilities, nothing left missing — then tag v0.0.X LTS.** Only after that,
20 tok/s.

This node turns that into a checklist that can be ticked, because "nothing left
missing" is otherwise unfalsifiable and would never ship. Every row is either
**done**, **gap**, or **won't** — and a `won't` needs a reason written here, not
a shrug.

## The honest starting position

Bigtea today opens **three** architectures and **one** tokenizer family. That is
the real distance to "any model", and it is far larger than the performance gap.

| | Bigtea | llama.cpp |
|---|---|---|
| architectures | **6** (`deepseek4`, `gemma2`, `llama`, `phi3`, `qwen3`, `qwen3moe`) | ~100 |
| tokenizer families | **2** (`gpt2` BPE, `llama` SPM) | 6 (spm, bpe, wpm, ugm, rwkv, plamo2) |
| quant types | all ggml can decode | same |
| CLI flags | **15** | ~100 |
| chat templates | **9 families** | ~40 |
| backends | CPU | CPU, CUDA, Metal, Vulkan, ROCm, SYCL |

## A. Open any model — the biggest gap, and the cheapest wins in it

**Two hard requirements block the entire Llama family today**, and neither is
deep:

1. `Qwen3Model::required_tensors` demands `attn_q_norm` / `attn_k_norm` on every
   block. Qwen3 has per-head QK norm; **llama, mistral, qwen2, gemma and phi do
   not**, so a container without them is refused before a byte is read.
2. `Tokenizer::from_metadata` refuses anything but `tokenizer.ggml.model ==
   "gpt2"`. The Llama family ships `"llama"` (SentencePiece).

| ticket | what | unlocks | state |
|---|---|---|---|
| **A1** | make QK-norm optional in the dense path | `llama`, `mistral`, `qwen2` structurally | **DONE** |
| **A2** | SPM tokenizer (`tokenizer.ggml.model = "llama"`) | the whole Llama/Mistral family's text | **DONE** — verified on TinyLlama |
| **A3** | accept the `llama` arch name and its metadata aliases | Llama 1/2/3, TinyLlama, CodeLlama, Vicuna, most finetunes | **DONE** — verified on TinyLlama and Llama-3.2 |
| A4 | `gemma`/`gemma2` | Gemma family | **gemma2 DONE 2026-08-10** — post-norms, attention soft-cap (50, into the fused kernel), final soft-cap (30), `sqrt(n_embd)` embedding scaling. Output matches llama.cpp exactly. **Refused past its 4096 sliding window**, which is not implemented |
| A5 | `phi3`, `qwen2` explicit | Phi, Qwen2 | **phi3 DONE 2026-08-10** — fused `attn_qkv` *and* fused `ffn_up` split into views; verified against llama.cpp's own output |
| A6 | WPM + UGM tokenizers | BERT-family, T5-family | **WPM DONE 2026-08-10** — verified token-for-token against `llama-tokenize` on `all-MiniLM-L6-v2`, 13 cases. GGUF stores WordPiece in *SentencePiece* spelling (`▁capital` + bare `ization`), not HuggingFace's `##`, so a textbook implementation `[UNK]`s every ordinary word without erroring: `../research/wordpiece-spelling-2026-08-10.md`. **UGM DONE 2026-08-10** — Viterbi lattice, `USER_DEFINED` scored 0, `f64` path sums; verified on `flan-t5-small`, 5 cases. **Caveat: the precompiled charsmap (NFKC) is not applied**, so input not already in normal form (fullwidth, ligatures) diverges: `../research/unigram-lattice-2026-08-10.md`. **Pre-tokenizers DONE 2026-08-10** — `tokenizer.ggml.pre` was read by nobody, so every BPE container was split with DeepSeek's rule: Qwen's digits grouped in threes instead of singly and `don't` was cut into three pieces on *both* families. `llama-bpe`/`llama3`, `qwen2` and `joyai-llm` now verified against real containers; every other variant is **refused by name** rather than guessed: `../research/pretokenizer-was-ignored-2026-08-10.md` |
| A7 | tied embeddings (`output.weight` absent → reuse `token_embd`) | many small models | **DONE** — Llama-3.2-1B is tied and loads |
| A8 | a clear error naming the *architecture* and what is missing | every unsupported model | **DONE 2026-08-10** — unverified architectures are **refused**; `--force` on the CLI only, never on the server |

**A8 is not cosmetic and should land first.** "Open any model" fails safely only
if the failure says which architecture, which tensor, and whether it is a gap or
a corrupt file. Today an unsupported model reports a missing tensor name.

**A1+A2+A3 together are the single highest-value item in this document**: they
take Bigtea from 3 architectures to the majority of GGUF files people actually
download.

## B. Performance against llama.cpp — the criteria

Parity is not one number. These are the cells that must be **≥ llama.cpp**, each
measured back to back in one session with both command lines recorded.

| criterion | V4-Flash | Qwen3-30B-A3B | Qwen3-4B dense |
|---|---|---|---|
| load / time-to-first-token | 1.25x behind | — | — |
| prefill tok/s | 1.25x behind | **ahead** @565, @2206 | **38.5 vs 111.2 — 2.9x behind** |
| generation tok/s | 0.37 vs 0.39 | 1.07 vs 2.16 | **4.27 vs 5.90 — 1.38x behind** (Llama-3.2-1B: 10.12 vs 12.91, 1.28x) |
| memory footprint at equal speed | **ours, by design** | ours | — |
| long-context generation | untested | untested | untested |

**Dense Qwen3-4B has never been compared to llama.cpp at all**, and it is the
cheapest comparison available — it fits in RAM, so it isolates the compute path
from all the streaming machinery. It should be the first cell closed.

Current ceiling on this machine, measured:
`the-plateau-was-ours-2026-08-10.md` puts a V4-Flash token at ~1.54 s of expert
reads + ~0.6 s of everything else, so with R2 overlap it is **~0.65 tok/s against
0.39** — a real 1.7x lead. That is the performance bar for LTS on this model.

## C. Options and capabilities

| ticket | what | state |
|---|---|---|
| C1 | sampling: temperature, top-k, top-p, min-p, repeat penalty, seed | **DONE 2026-08-10** — 10 unit tests, `--llamacpp-defaults` for like-for-like comparison |
| C2 | chat templates from `tokenizer.chat_template` | **DONE 2026-08-10** — 9 families, detected from the real templates; control tokens encode to single ids |
| C3 | streaming responses (SSE) in `bigtea-serve` | **DONE 2026-08-10** — plus temperature/top_p/top_k/min_p/seed/stop from the request, EOS and stop sequences give `finish_reason: stop` |
| C4 | `-c` context size, `-b` batch, `-t` threads as flags | **DONE 2026-08-10** — llama.cpp's own spellings; the thread default was a hardcoded 12 and is now all cores |
| C5 | stop sequences, `max_tokens`, `n_predict` | **DONE** — `--stop` (repeatable) on the CLI, string-or-array on the server; both match accumulated text, not tokens |
| C6 | grammar / JSON-schema constrained output | **won't for LTS** — large, and not what an agent needs first |
| C7 | LoRA adapters | **won't for LTS** — no user asking |
| C8 | embeddings endpoint | **won't for LTS** — the graph returns logits, not hidden states. Faking it is worse than a 501, and doing it properly means a second output path |
| C9 | quantise/convert tooling | **won't** — llama.cpp owns this and does it well |

**C1 is required, not optional.** Greedy decoding makes every answer
deterministic and flat; no one will judge quality favourably against llama.cpp
without samplers, and it is a day of work.

## D. Standards compliance

| ticket | what | state |
|---|---|---|
| D1 | read every GGUF metadata type incl. arrays and nested | **DONE 2026-08-10** — 16 tests: hand-written malformed corpus plus two sweeps that need no fuzzing crate (every prefix of a valid container; >1,000 single-byte corruptions). Found one real bug: **a duplicate metadata key overwrote silently**, so a file with two `general.architecture` entries loaded as the second one with no error — now `DuplicateKey`, alongside `EmptyKey` and `DuplicateTensor`, matching llama.cpp. See `../research/malformed-containers-2026-08-10.md` |
| D2 | GGUF v2 and v3 | **DONE 2026-08-10** — v2 and v3 proved to parse identically from in-memory headers, v1 and future versions refused, alignment honoured only when a power of two. **The ticket's premise was wrong**: the `u32`→`u64` length change was v1→v2, not v2→v3, and implementing it as written would have mis-read every real v2 container — llama.cpp has no width branch and refuses v1 outright. Also added: a byte-swapped version is now named as an endianness mismatch instead of "unsupported version 50331648". See `../research/gguf-v2-premise-was-wrong-2026-08-10.md` |
| D3 | split containers (`-00001-of-0000N`) | **done** |
| D4 | every ggml quant type ggml can decode | **done — delegated to ggml** |
| D5 | OpenAI API surface: `/v1/chat/completions`, `/v1/models`, `/v1/completions`, `/v1/embeddings` | **3 of 4 + an honest 501.** Chat streams and serves any supported architecture; `/v1/completions` runs the prompt verbatim; **embeddings refuse with 501** rather than returning a logit-derived vector that would look right and behave like noise |
| D6 | refuse an unsupported container clearly rather than producing nonsense | **DONE** — see A8. Gemma-2 was the proof that it was needed |

## The order

1. **A8** — say clearly what is not supported. Everything else is safer after it.
2. **A1 + A3** — QK-norm optional, accept `llama`. Small, structural.
3. **A2** — SPM tokenizer. Needs unit tests against fixtures; a wrong tokenizer
   produces fluent nonsense, never a crash.
4. **C1** — samplers. Cheap, and quality is judged on it.
5. **B: Qwen3-4B dense vs llama.cpp**, both command lines recorded. The cheapest
   uncollected comparison in the project.
6. **C2 + C3** — chat templates and streaming, which is what makes the server
   usable from an editor.
7. **R2 overlap** — the remaining measured 1.4x on V4-Flash.
8. Then A4–A7, C4, C5, C8, D1, D2, D5.

**Then tag v0.0.X LTS.** Then 20 tok/s.

## What this document deliberately does not promise

Feature parity with *all* of llama.cpp is not achievable and not the goal — it
is years of work by hundreds of people across every backend. `won't` rows above
are the honest boundary. **The LTS claim is: "opens the models people actually
run, matches or beats llama.cpp on the models it supports, and tells you the
truth about your machine before you download 144 GB."** Anything wider than that
would be a claim this project cannot defend, and this project has retracted two
claims already.

## One thing that needs Atur

**Testing A1–A3 needs a Llama-architecture GGUF, and there is none on this
machine.** TinyLlama-1.1B Q4_K_M is ~670 MB — the smallest container that
exercises both the `llama` architecture and the SPM tokenizer. Home internet is
limited, so this is a decision, not an assumption: the code can be written and
unit-tested without it, but **it cannot be called supported until a real
container has been opened and its output checked.**
