# Widening the parity harness from three prompts to eight found three real bugs

**2026-08-11.** Branch `ticket/r14-architectures`, off `main` at `f0c7a42`.

Links: [../backlog/lts-parity-criteria.md](../backlog/lts-parity-criteria.md) ·
[../backlog/layernorm-and-biases.md](../backlog/layernorm-and-biases.md) ·
[verify-before-citing](../../../MEMORY.md)

## The claim being tested

`scripts/parity-check.sh` grew from three prompts to eight because `starcoder2`
had passed 3/3 while running the **wrong pre-tokenizer**. The header now says a
pass is evidence about those prompts, not about the architecture.

So: re-run everything at eight and see what the extra five catch. They caught
three bugs, in code that had been on `main` for weeks, in two architectures
listed as verified.

## 1. Llama-3.1/3.2/3.3 had the wrong RoPE — `rope_freqs.weight` was never read

**The biggest of the three, and it was in `llama`, the most-used entry in
`VERIFIED_ARCHITECTURES`.**

Llama-3.1 onwards carry `rope_scaling.rope_type = "llama3"`. llama.cpp folds the
low/high frequency factors into a tensor at conversion time — `rope_freqs.weight`,
`n_rot/2` per-frequency divisors — and hands it to `ggml_rope_ext` as
`freq_factors`. We passed `None` there, so the low frequencies rotated as though
the model had never been context-extended.

Nothing announces it. The metadata reads:

```
print_info: rope scaling          = linear
print_info: freq_scale_train      = 1
```

which is exactly what an *un*-extended model reports. The only sign in
llama.cpp's own log is one line among hundreds:

```
create_tensor: loading tensor rope_freqs.weight
```

Llama-3.2-1B-Instruct, `parity-check.sh <model> 32`:

| | ok | unstable | FAIL |
|---|---|---|---|
| before | 3 | 4 | 1 |
| after | **8** | 0 | 0 |

The three-prompt harness scored this 2 ok + 1 unstable and read as a pass.

Two pieces were needed, and the second is the one that bites: `rope_freqs.weight`
had to be added to `required_tensors()`, or it is **never loaded** and
`weights.get` returns `None` — the graph then carries on with a slightly wrong
rotation rather than failing. That is the same trap that made StableLM read
"almost right" when its biases were absent from that list.

## 2. Falcon3 was fed a sequence one token short

Falcon3's container declares an EOS and **no BOS, and no `add_bos_token`**.
llama.cpp resolves both by other means:

* `tokenizer_model == "gpt2"` sets `special_bos_id = special_eos_id = 11`
  *before* the container's keys are read, so the default is what it runs on;
* the `llama3`/`llama-bpe`/`falcon3` arm sets `add_bos = true`.

We defaulted `add_bos` to false for every BPE vocabulary — "for BPE the absent
flag genuinely means no" — and had no BOS id at all. So llama.cpp prefilled
`<|endoftext|> SELECT …` and we prefilled `SELECT …`.

| | ok | unstable | FAIL |
|---|---|---|---|
| before | 1 | 5 | 2 |
| after | **8** | 0 | 0 |

## 3. A USER_DEFINED token's text is copied verbatim, not byte-decoded

Falcon3's newline is id 12, `token_type = 4` (USER_DEFINED), holding a **raw
`\n`** rather than the `Ċ` a GPT-2-family vocabulary usually spells it with.

`bytes::decode` is `filter_map(char_to_byte)` — it silently **drops** every
character outside the GPT-2 byte alphabet, and `\n` is outside it. So every
newline vanished and generation arrived as one run-on line:

```
bigtea   :  Paris.Q: What is the capital of France?Options:- france- france
llama.cpp:  Paris.\nQ: What is the capital of France?\nOptions:\n- france\n- paris
```

llama.cpp's `token_to_piece` copies a USER_DEFINED token's text verbatim in the
BPE arm and only sends NORMAL through the byte alphabet. `bpe_decode_bytes` now
does the same, decoding *runs* of NORMAL tokens together so a multi-byte
character split across tokens still reassembles.

## The finding that matters more than the three bugs

**"The reference disagrees with itself" is not a safe verdict. It hid nine
prompts' worth of real bugs.**

The harness re-runs a mismatch under `-fa off` and `--no-repack`, and if
llama.cpp's own answer moves, calls the prompt a near-tie and moves on. That
test compares the *reference against itself*. It cannot see that **our input
differed** — and when the input differs, a near-tie is exactly what you get,
because the model is answering a slightly different question and lands on the
other side of whatever was close.

Nine of the eleven prompts that reported `unstable` in this session were bugs:

* Llama-3.2-1B — 4 unstable → 0 after `rope_freqs`
* Falcon3 — 5 unstable → 0 after BOS
* OLMo — 1 unstable → 0 (see below)

Two survive and are genuine: Phi-3-mini's two, which were already documented.

So the rule to apply next time: **a high unstable count is a smell, not a
pass.** One near-tie in eight is ordinary. Five is a bug you have not found yet.

## A fourth thing, in the harness rather than the engine

`llama-completion` prints ` [end of text]` when the model stops on EOS. Bigtea
prints no equivalent, so any model that terminated early read as a FAIL whose
two sides were otherwise identical:

```
FAIL      tinyllama   Q: What is 17 plus 25? A:
  bigtea   :  42
  llama.cpp:  42 [end of text]
```

That is a status line, not output. `ref()` strips it now. It also explains two
of the `unstable` verdicts: `-fa off` sometimes stopped on EOS where the default
run did not, so the reference "disagreed with itself" over a suffix neither
model generated.

## The harness acted on this, and the sweep was re-run under it

**Update, same day.** `unstable` is no longer a verdict (`b2ad35f`). On a
mismatch the script now compares the **tokenized prompt** first: different
counts mean the two engines are not answering the same question, and it reports
FAIL. **All three bugs above are in that class** — the check catches a missing
BOS, a wrong pre-tokenizer and a byte-fallback that drops characters, in one
test. It also counts near-ties, and three or more in eight exits non-zero,
because one is ordinary and three is a bug nobody has found yet.

All twelve models were re-swept under the stricter script. **Every result below
is unchanged**, and every model exits 0. Phi-3's two survive both new checks:
the prompts tokenize identically on both engines, and two is under the cluster
threshold. So they are still the only unexplained near-ties here, and still
unexamined.

## What did not need fixing

Three of the four architectures added this session were near-misses at most:

* **internlm2** — 8/8 on the first run. Its only requirement was the RoPE
  convention (llama.cpp lists `LLM_ARCH_INTERNLM2` in the NORM branch; an
  unknown architecture defaults to NeoX here).
* **baichuan** — 8/8 on the 7B. See the refusal below for the 13B.
* **falcon3** — **not a new architecture at all.** It converts to `llama`, and
  `falcon3` is one more alias in llama.cpp's `llama-bpe` arm. Everything it
  exposed was in shared code.

**olmo** needed one real feature: its norms have **no learned parameters**.
llama.cpp builds every one as `build_norm(x, NULL, NULL, LLM_NORM)` — centre,
divide by the standard deviation, stop. The container holds no
`attn_norm.weight`, no `ffn_norm.weight`, no `output_norm.weight`, so the loader
refused it outright: `container has no tensor "output_norm.weight"`.

That refusal was the *good* outcome. The dangerous reading is the mirror image —
an affine architecture that loses a norm weight and quietly runs
non-parametric — so `norm_affine` gates it and a missing weight is still
`MissingTensor` when the architecture is affine. And because a norm with neither
weight nor bias is only meaningful as a LayerNorm (a parameterless RMSNorm
exists nowhere in llama.cpp), `layer_norm` is now *also* true when the weight is
absent, which is what keeps the centring.

`layer_norm` and `norm_bias` had to become two booleans for the same reason:
they were one, because every LayerNorm seen until now had a bias, and OLMo made
the loader demand an `output_norm.bias` that cannot exist.

## One refusal added

**Baichuan-13B uses ALiBi and nothing in the container says so.** llama.cpp
picks it by *layer count* — `baichuan.cpp` sets `f_max_alibi_bias = 8.0` when
`n_layer == 40`. The 7B and the 13B hold the same tensor set under the same
architecture name; the 13B would load, rotate keys it should not rotate, skip a
bias it should apply, and answer fluently. `verify()` refuses it.

So `baichuan` in `VERIFIED_ARCHITECTURES` means "a container was diffed", not
"every model of this name runs" — the list's doc comment now says so.

## Scoreboard, all twelve models, eight prompts, 32 tokens

Every run in one session on the same build, `scripts/parity-check.sh <model> 32`:

```
OLMo-1B.Q4_K_M                        8 ok      NEW
internlm2-math-plus-1_8b.Q4_K         8 ok      NEW
baichuan2-7b-chat.Q4_K_M              8 ok      NEW
Falcon3-1B-Instruct-q4_k_m            8 ok      (arch llama)
stablelm-2-1_6b-chat.Q4_K_M           8 ok
starcoder2-3b-Q4_K_M                  8 ok
Qwen2-0.5B-Instruct-Q4_K_M            8 ok
Qwen3-4B-Q4_K_M                       8 ok
gemma-2-2b-it-Q4_K_M                  8 ok
gemma-3-1b-it-Q4_K_M                  8 ok
Llama-3.2-1B-Instruct-Q4_K_M          8 ok      was 3 ok / 4 unstable / 1 FAIL
tinyllama-1.1b-chat-v1.0.Q4_K_M       8 ok      was 7 ok / 1 FAIL (harness)
Phi-3-mini-4k-instruct-q4             6 ok, 2 unstable
```

426 workspace tests, `clippy --workspace --all-targets -D warnings` clean, fmt
clean. `VERIFIED_ARCHITECTURES` is thirteen: baichuan, deepseek4, gemma2,
gemma3, internlm2, llama, olmo, phi3, qwen2, qwen3, qwen3moe, stablelm,
starcoder2.

## Not done

* The `clamp_kqv` path (MPT/DBRX/OLMo) is written against llama.cpp's code, not
  against a run — OLMo-1B declares `0.0`. The OLMo-7B is the container that
  would exercise it.
* Phi-3's two unstable prompts are unexamined. They were unstable before this
  session too, but after nine `unstable` verdicts turned out to be bugs, "it was
  already like that" is not much of a defence.
* Containers are at `C:/Projects/models/{olmo,internlm2,falcon3,baichuan}/` and
  are the only copies on this machine.
