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
| architectures | 3 (`deepseek4`, `qwen3moe`, `qwen3`) | ~100 |
| tokenizer families | 1 (`gpt2` BPE) | 6 (spm, bpe, wpm, ugm, rwkv, plamo2) |
| quant types | all ggml can decode | same |
| CLI flags | 3 | ~100 |
| chat templates | none | ~40 |
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
| A4 | `gemma`/`gemma2` (post-norm, logit soft-cap, tied embeddings) | Gemma family | gap |
| A5 | `phi3`, `qwen2` explicit | Phi, Qwen2 | gap |
| A6 | WPM + UGM tokenizers | BERT-family, T5-family | gap |
| A7 | tied embeddings (`output.weight` absent → reuse `token_embd`) | many small models | gap |
| A8 | a clear error naming the *architecture* and what is missing | every unsupported model | **partial** — an unverified architecture now says its RoPE layout is a guess |

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
| prefill tok/s | 1.25x behind | **ahead** @565, @2206 | not measured |
| generation tok/s | 0.37 vs 0.39 | 1.07 vs 2.16 | **3.53, not compared** |
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
| C4 | `-c` context size, `-b` batch, `-t` threads as flags | partial |
| C5 | stop sequences, `max_tokens`, `n_predict` | **server DONE** — stop accepted as string or array; CLI still partial |
| C6 | grammar / JSON-schema constrained output | **won't for LTS** — large, and not what an agent needs first |
| C7 | LoRA adapters | **won't for LTS** — no user asking |
| C8 | embeddings endpoint | gap, small |
| C9 | quantise/convert tooling | **won't** — llama.cpp owns this and does it well |

**C1 is required, not optional.** Greedy decoding makes every answer
deterministic and flat; no one will judge quality favourably against llama.cpp
without samplers, and it is a day of work.

## D. Standards compliance

| ticket | what | state |
|---|---|---|
| D1 | read every GGUF metadata type incl. arrays and nested | likely done, **untested against a fuzz corpus** |
| D2 | GGUF v2 and v3 | v3 done; v2 untested |
| D3 | split containers (`-00001-of-0000N`) | **done** |
| D4 | every ggml quant type ggml can decode | **done — delegated to ggml** |
| D5 | OpenAI API surface: `/v1/chat/completions`, `/v1/models`, `/v1/completions`, `/v1/embeddings` | 2 of 4 |
| D6 | refuse an unsupported container clearly rather than producing nonsense | partial — **the most important safety property this runner has** |

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
