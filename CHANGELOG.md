# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, anything may change in a minor release.

## [Unreleased]

## [0.0.12] — 2026-08-20

### The app updates itself

Atur: *"users can get the most updated release when they connect to the internet
from the app — an updating flow, not every time go and download a new setup. For
all apps and exports we need."*

- **Chaos asks once at startup whether a newer release exists**, and says nothing
  unless one does. If one does, it offers to fetch the installer and hand over.
  `Help ▸ Check for updates` asks on demand and always answers; `Help ▸ Install
  update…` does the download. `CHAOS_NO_UPDATE_CHECK=1` turns the automatic
  check off.
- **`chaos-run --update` does the same from a terminal**, because the window is
  one of twelve binaries a release ships and a CLI user should not have to open
  a GUI to learn that a release exists.
- **One update updates everything.** The installer carries the whole payload —
  the window, `chaos-run`, `chaos-serve`, all of it — so there is nothing to do
  per binary. Your models are not touched.
- The decision logic is in `chaos_model::release` and is tested against
  **GitHub's actual response**, kept as a fixture: `0.0.9` must not look newer
  than `0.0.11`, and a macOS tarball must not be downloaded onto Windows.

### Five builds, eight assets

Two platforms that had nothing at all:

- **Intel Macs.** `macos-latest` is Apple Silicon, so every macOS release so far
  was arm64 only.
- **arm64 Linux** — a Raspberry Pi, an Ampere box, an arm64 VM.

The updater picks by **architecture as well as operating system** now. An arm64
tarball on an Intel Mac is the same silent failure as the wrong OS one level
down: it downloads, it unpacks, and then nothing runs. A test pins the five
names against the five the release workflow builds, because the updater matches
by string equality — a matrix entry renamed in one place and not the other is an
updater that says "no installer for this platform" forever, on that platform
only, with nothing in a log.

### Images ship

`chaos-draw` is a **binary** now, not an example held back behind a comment that
said it would become one "when there is evidence, not before". The evidence:
the autoencoder round-trips real photographs at **36.09–40.89 dB** with the check
ablated three ways first, the text encoder answers `" Paris"` at **logit 22.58**,
the denoiser scores **0.85 velocity cosine** against a real latent, and at
1024×1024 the output is photorealistic — skin texture, individual hairs,
catchlights.

```
chaos-pull ideogram-4 ideogram-4-uncond qwen3-vl-8b flux2-vae
chaos-draw "a red apple on a white table" --grid 64 --steps 20 -o apple.png
```

**What is still imperfect is object form**, and `--help` says so rather than
leaving it to be discovered after two hours: colour and scene follow the prompt,
a named object may come out the wrong shape. Structured, JSON-shaped prompts
condition about three times as strongly as a bare phrase, which is what these
models were trained on.

It ships in every artifact, so a release now carries twelve binaries.

### The icon, drawn from the vector at every size

`make-ico.py` renders `assets/logo.svg` at each of the nine sizes Windows asks
for. It now does so with **8 subsamples instead of 3** — that is the number of
grey levels an antialiased edge can take, and this mark is two dozen rays about
one pixel wide, so nine steps was what "blocky" looked like — and with a **4%
inset rather than 8%**, which at 16px is the difference between 13 and 15 pixels
of drawing.

### The window, after Atur used it

Six reports, each one a real defect:

- **The rail lit every page at once.** Invalidating the parent window does not
  repaint owner-drawn children, so each rail item lit itself when clicked and
  nothing ever un-lit the last one — click all four and all four are
  highlighted. `View ▸` and `Ctrl+1..4` were worse: they changed the page and
  left the rail pointing at the old one.
- **Drop-downs opened too narrow to read.** A Win32 drop-down is exactly as wide
  as its box unless told otherwise, so *"Processor (the GPU is not used here
  yet)"* opened as *"Processor (the GPU is not used her…"*. The open list is now
  measured against its longest option, capped to the screen.
- **BROWSE sat on top of the sentence explaining the models folder.** Two
  layout walkers stepped by different amounts; they now share one.
- **The mark in the rail was rough.** It is a sun of two dozen fine rays around
  an eye, drawn at 44px with 16 levels of antialiasing — about one level per ray
  edge. Now 64px at 64 levels, scan-converted from outlines and cached.
- The models list gives the name its own column, so a narrow window eats a
  measurement rather than the end of `Qwen3-VL-8B-Instruct-Q4_K_M`. The
  **AVAILABLE** tab has columns too, and it was worse there — drawn as one
  string, "needs 16.5 GB - slow, re-reads" came out "needs 16.5 GB - sl…",
  cutting the one word the row exists to say. The list also got wider, and the
  resident requirement is now printed **only when it differs from the download
  size**: on a dense model they are the same number, so twenty rows out of
  twenty-seven were spending a column to repeat themselves.
- The window is DPI-aware (per-monitor v2) and opens centred in the work area
  rather than off the corner of a scaled display.

## [0.0.11] — 2026-08-20

### Chaos draws images

The whole path now runs: **Qwen3-VL-8B** turns the prompt into conditioning,
two copies of **Ideogram 4** — the conditional model and a separately trained
unconditional twin, 5.26 GiB each — denoise a latent, and the **FLUX.2**
autoencoder turns it into pixels. Four files, 16.7 GB.

**It follows the prompt's colour and scene and gets an object's form wrong.**
Said plainly, because a diffusion pipeline that is subtly wrong produces a
plausible picture: at 512×512 *"a red apple on a white table"* drew red on white
surfaces beside a wooden shelf, and not an apple. The foundation is verified;
the pictures are not good yet.

What *is* verified, and how:

- **The text encoder** answers *" Paris"* after "The capital of France is" at
  logit 22.58 — so its attention, rotary positions, per-head QK norm,
  grouped-query broadcasting and causal mask are all right.
- **The denoiser** is scored without looking at a picture. A rectified-flow
  model at noise level `sigma` must predict `noise - latent`, and **both terms
  are known** because the autoencoder's encoder — verified to 36 dB by round trip
  in v0.0.10 — turns a real photograph into a real latent. Cosine against the
  truth is **0.85 at 512×512**.

Scoring against each half separately — "can it see the noise" against "can it
see the image" — is what found three bugs a picture could not have told apart.
**Two of them are in the reference implementation**: `stable-diffusion.cpp`
reads `1.f / 128.f` as an attention scale when it is an F16 overflow guard that
cancels exactly, and it never applies the autoencoder's own latent
normalisation. Being faithful to a reference and being right are not the same
thing.

### 1024×1024 stops being impossible

Chaos runs models larger than memory by streaming them; the image path did not.
Both halves built each graph in a context whose arena allocates every tensor and
frees none, so they paid for every intermediate ever written while the live set
is a handful. Decoding 768×768 wanted **29.5 GiB** and aborted the process after
an hour of denoising.

`ggml_gallocr` plans a graph and gives the same buffer to tensors whose
lifetimes do not overlap. The *device* path here always did this; the host path
never did.

| | before | after |
|---|---|---|
| decode 256×256 | 3.69 GiB | **0.20 GiB** |
| decode 1024×1024 | 52 GiB | **3.4 GiB** |
| denoise 1024×1024, per layer | 14.6 GiB | **2.0 GiB** |

**Bit-identical, not merely close**: 0 of 196,608 pixels differ, and the
denoiser's velocity score is unchanged to four decimals.

### The app says how far a model has loaded

It said "loading — a large model takes a while" and then nothing, which on a
144 GB container is minutes of a window that looks broken. Now:

```
loading qwen3-4b  62%  ·  1.7 GB of 2.8 GB resident  ·  340 MB/s  ·  3s left
```

Measured from the server's working set against the catalogue's resident figure,
the same way the download bar is measured from bytes on disk — no protocol, no
parsing, and it survives a server started with no console. **The bar stops at 99
until the server answers**, because the tail of a load is not weights and a full
bar over a busy app is a lie. A model the catalogue does not know gets bytes and
**no percentage**: a denominator taken from the file size would report V4-Flash
as 5% loaded for its whole load.

### Fixed

- **`chaos-pull` accepted a corrupt download.** A resume the CDN did not honour
  made curl append the whole file to the partial one; the result was 478,535,680
  bytes **too large**, `saturating_sub` floored the remainder to zero, and it
  reported "already complete". The container passed every structural check —
  being too *big* means every tensor offset is readable — and produced NaN at
  block 31 of 36. Now refused by name, and the size is verified after
  downloading.
- An image too large to decode is refused **in 34 milliseconds, before anything
  is loaded**, naming the largest size that works — rather than aborting inside
  ggml after an hour of denoising.


## [0.0.10] — 2026-08-20

### Chaos decodes images, and the autoencoder is verified rather than eyeballed

`chaos-image::vae` builds the FLUX.2 autoencoder as a ggml graph — both halves,
248 tensors, and **no transposes anywhere**, because a contiguous PyTorch
`[OC, IC, KH, KW]` convolution weight *is* ggml's `[KW, KH, IC, OC]` read
backwards. The decoder turns a latent into pixels; the **encoder exists so the
decoder could be checked without a reference implementation**.

Encode a real photograph, take the distribution's mean, decode it, compare. The
two halves are separately trained weights over one shared latent space, so
neither can compensate for a bug in the other. Four 256×256 photographs:
**36.09, 36.29, 36.49 and 40.89 dB**.

That is only worth printing because the check was ablated first. Against the
same input, baseline 36.09 dB:

| deliberate error | PSNR |
|---|---|
| `group_norm` without its per-channel scale and shift | 16.77 dB |
| downsampler padded symmetrically instead of `(0,1,0,1)` | 14.60 dB |
| mid-block attention skipped | 31.93 dB |
| convolution kernels not dimension-reversed | ggml aborts |

Three of the four still produce a recognisable picture, and all three would have
passed "it looks like a photograph" — which is this project's oldest hazard.

**The denoiser is not started**, so Chaos cannot yet generate an image from a
prompt; it can decode a latent into one. The text encoder and the unconditional
twin are fetchable but not yet fetched.

New ggml bindings, each exercised in `try-vae-ops` against a hand-computed
answer before going near the suite: `new_f32_4d`, `upscale_nearest`, `pad`,
`cont_2d`, and a `Tensor::ne` accessor. `ggml_pad` puts its zeros at the far end
of each dimension only, which is exactly diffusers' asymmetric downsample
padding.

### The file has 251 tensors and the graphs name 248

The remainder is a BatchNorm — `bn.running_mean`, `bn.running_var`,
`bn.num_batches_tracked` — holding the **latent normalisation**, which is what
earlier autoencoders did with a scalar `scaling_factor`. It is **128-wide, not
32**: the patchified channel count, 32 latent channels times a 2×2 patch, which
is exactly what the denoiser consumes.

A round trip never touches it, because encode and decode are inverses whatever
the normalisation is — which is what makes the round trip a fair test of this
port and *not* a test of the interface to the denoiser. Recorded because leaving
it out is the next chance to produce a confident, plausible, wrong image.

### `usable without ggml` had been red, and was reported as green

The check failed from the commit that added `try-vae-ops` onwards: `cargo clippy
--all-targets` builds examples, and that example names `Context`, which is
`#[cfg(have_ggml)]`. Four of five checks green was read as green. The gated
*tests* in that crate use `#![cfg(have_ggml)]` on the whole file; an example
cannot, because a file with no `main` does not compile at all, so the cfg goes on
the function with a stub that says what to set.

### Fixed

- The new autoencoder tests panicked when the 336 MB file was absent, which is
  what they were written to do and which turned three CI jobs red — CI runs
  `cargo test --workspace -- --ignored` on three platforms that do not have it.
  They skip now, the way the V4-Flash set does. Because that is the hazard this
  project has already been burned by, the skip prints **"SKIPPED, NOTHING WAS
  CHECKED"** and names the file, and `CHAOS_REQUIRE_VAE=1` turns absence back
  into a failure.

### Note on 0.0.9

There is no v0.0.9 release. The version was bumped in `Cargo.toml` and the
installer was exercised locally, but the tag was never pushed, so v0.0.8 was the
newest release until this one. Everything that section would have contained ships
here.


### Qwen3.6-27B loads, generates, and is wrong — in llama.cpp too

Found while checking that every installed model runs. It exits 0 and prints
`ทัน ทัน ทัน ทัน ทัน ทัน`. Qwen3.5-0.8B — the same architecture at 24 layers
instead of 64 — is byte-identical to llama.cpp at three prompt lengths, so
`qwen35` is verified on the shape it was diffed against and not on this one.

llama.cpp fails on the same file too — `333333` — and the per-layer sums agree to
five significant figures through layer 5 before both overflow to NaN. **That is a
clue, not a defence**: reproducing a wrong answer precisely is still a wrong
answer, and this engine is judged on whether it replies correctly. What the
agreement buys is knowing where not to look. Chaos warns before generating, and
the warning names no other project.

The first sweep of all twelve models recorded "twelve of twelve" because every
one exited 0 and the outputs were never read. That is this project's own
documented hazard walked into by a test that checked liveness instead of
correctness. The sweep now requires the word `Paris` in the continuation of
"The capital of France is".

Ruled out by measurement, not reasoning: the key-head broadcast (the 27B has 16
key heads and 48 value heads where the 0.8B has 16 and 16, so a missing
broadcast would be invisible there — a new test calls the fused op at a 2:6
ratio and proves it broadcasts on its own), and every tensor shape in both
containers against what the config computes. Open, with the next three
experiments written down, in `backlog/qwen35-27b-is-wrong.md`.

### Loading a second model sent your messages to the first one

The window never stopped the running server before starting the next. So
loading a second model spawned a second `chaos-serve` while the first still
held the port: the new process died with `os error 10048`, the readiness check
took its 200 from the **old** server, and the window said the new model was
ready while every message went to the old weights. The dead child's handle
overwrote the live one, so nothing could kill the orphan either — it kept its
memory, which on a 15.7 GiB machine running a 144 GB model is the difference
between 0.45 tok/s and nothing at all.

Both halves were reproduced before the fix and re-checked after. The window now
stops the server first, waits for the port to be released, watches the process
rather than only the port, and shows the server's own reason when one exits —
that sentence used to be written to a console the app does not have.

### Four settings that did nothing

`chaos-serve`'s argument parser swallowed anything it did not recognise. It had
no `-ngl`, no `-c`, no `--auto` and no `--force`, and the app sent all four — so
the GPU list, the context list and the AUTO and FORCE toggles were decorations
for three releases.

- An unknown flag is an error now, in `chaos-serve`, `chaos-probe`,
  `chaos-pull` and `chaos-model-info`. A flag is a promise.
- `-c` works, and only ever *lowers* the limit: a ceiling above what the engine
  can hold is a promise broken mid-request.
- `--force` works, and says on every start that the architecture is unverified.
- `--auto` sizes the expert cache from the memory actually free.
- `-ngl` is refused **by name, with the reason**: the server binds weights
  straight into host memory rather than through the runner's device loader. The
  GPU list now says the window runs on the processor and that `chaos-run -ngl
  99` uses the card. One true option beats three that do nothing.

### The GPU on V4-Flash was accepted and ignored

`--device` and `-ngl` on the streaming expert path changed nothing and printed
nothing, so "I turned the GPU on and it does no work" was an accurate
description of the program. `chaos-run` now says so, with the reason. Not built
rather than not wanted: V4-Flash's always-read set is 7.38 GiB against 5.11 GiB
free on this card, and the only measured figure for a streaming MoE on this
device is **4.3x slower** than the host path.

### The uninstall could not finish

Three causes, each sufficient on its own:

- **A dialog outlived the helper's patience.** An uninstall started from inside
  the install folder stages a copy of itself in `%TEMP%`, because a running
  executable keeps its own file open. The parent then showed a modal and only
  quit once dismissed — while the helper retried for ten seconds and gave up.
  The helper now waits for its parent by process handle and reports the result
  itself.
- **`version.txt` was never removed.** The install wrote it; the uninstall
  removed the manifest and the log and not this, so the folder could never be
  deleted and a stale version file was left claiming Chaos was installed. Found
  on a real machine, not in a test. Both ends now read one list.
- **Add/Remove Programs opened the installer.** `UninstallString` carried no
  arguments, so Windows Settings showed the welcome screen with INSTALL as its
  primary button. It now asks once and removes. `QuietUninstallString` is there
  too, for `winget uninstall`.

### An update looked exactly like a first install

The upgrade line existed and was only ever written into the report — *after* the
install had run. The welcome screen now reads the chosen folder first and says
"Chaos 0.0.8 is installed here. This updates it to 0.0.9.", with the button
relabelled UPDATE, or REINSTALL for the same version. It re-reads when the path
is edited.

### Eight of ten binaries had no icon, and Explorer kept showing the old one

`chaos-app` and `chaos-setup` each carried a private copy of the same `windres`
work, so the other four crates that build binaries had nothing to include:
`chaos-run.exe`, `chaos-serve.exe`, `chaos-pull.exe`, `chaos-probe.exe`,
`gguf-info.exe` and the benches all shipped with the blank Windows default. One
`chaos-build` crate now holds it — a build dependency, linked into nothing.
Verified by extracting the icon from all ten executables: one hash, ten
binaries.

And the file being right was not enough. Explorer caches an icon by path and
does not re-read a file overwritten in place, so an upgrade kept showing the
previous version's icon. The installer now tells the shell otherwise, which is
what every installer does at the end of a copy.

### The mark was too small to read

The icon's art filled the middle 68% of its tile; at 32px that leaves the sun's
rays one pixel wide and the eye invisible. Now 84%. The app's rail mark went
from 32px to 44 in a 208px rail, and the installer's welcome mark from 68px to
96 in a 900px window, where it had less presence than the wordmark beneath it.
Nothing was resampled to get there — the art is filtered from the 256px master
at whatever size is asked for.

### The logo lost its eyes

**An SVG path is one region, however many subpaths it has.** With no
`fill-rule` the default is nonzero, so a subpath winding the other way is a
*hole* — and one path in this mark is a near-white shape whose hole is what the
eyes show through. The generator emitted every subpath as its own shape and
filled each independently, so the hole was filled too. 43 paths producing 44
polygons was the tell, and `assets/logo.png` — rendered by a different tool that
collects a path's edges before scanning — had always shown the eyes. Two
renderers of the same art disagreeing was the whole bug.

### Release assets are named one way

v0.0.8 shipped a mix of `chaos-` and `Chaos-`, and the AppImage was the only
asset that did not say which platform it was for. Now one prefix, one version
format, and platform and architecture on everything — with the `.deb` keeping
Debian's mandated `name_version_arch.deb`, because an installer that will not
install is not tidier. The README lists all five downloads with what to do with
each, and says plainly that the window is Windows-only.

### Mojo and xtool, asked and answered

Both in `research/mojo-and-macos-packaging-2026-08-19.md`.

**Mojo** is genuinely open source now — Apache 2.0, compiler included, CUDA,
HIP and Metal. It is also a language, not a memory manager: deciding which of
256 experts stay resident is policy, and that policy is what this project *is*.
What it could replace is ggml's kernels, and the measured numbers say that is
not where the time goes — the routed expert arithmetic is under 5% of a
V4-Flash token, which is 1.56 s of disk read. It would also be the first
external toolchain here, and Modular is not accepting compiler contributions
until the end of 2026.

**`xtool`** builds *iOS* apps with SwiftPM. This is a Rust project and the ask
was macOS, so it is the wrong tool rather than a tool used wrongly. A macOS
build needs the Apple SDK on Apple hardware — which the release workflow
already has, on a `macos-latest` runner. The `.deb` and the AppImage already
existed too, verified in CI with a real `dpkg -i` and a PATH check.

### The Qwen3.5 forward pass is implemented and diffed against llama.cpp

`qwen35` — Qwen3.5, Qwen3.6 **and Qwen3.8**, all one architecture — now runs
end to end. Developed against `Qwen3.5-0.8B-Q8_0` because it is the same
architecture at 24 layers instead of 64, so a wrong step shows up in a second
rather than a minute.

**Verified, not asserted.** Every layer's output compared against
`llama-eval-callback` on the same container and prompt, by value *and* by sum
over all five prompt tokens:

```
l_out-0   llama -0.3452   chaos -0.345155    sum -4.384898 / -4.384897
l_out-3   llama -0.2049   chaos -0.204932    (the first attention layer)
l_out-23  llama  0.3155   chaos  0.315493    sum 26.697807 / 26.697754
```

**All 24 layers agree**, and the sampled token ids are llama.cpp's exactly:
`" Paris"`, `"."`, `"
"`, `"The"`, `" capital"`, `" of"`.

The sums matter as much as the values: the delta net is recurrent, so a state
carried wrongly leaves token 0 perfect and every token after it wrong. Comparing
first rows alone said the port was correct while the answer was still garbage.

What it took:

- **The gated delta net** (`crates/chaos-arch/src/qwen35.rs`) — 18 of the 0.8B's
  24 layers, 48 of the 27B's 64. Projections, a rolling depthwise convolution,
  `l2_norm` on q and k, `ggml_gated_delta_net`, a gated norm, an output
  projection. The state is host-side like the KV cache, so no in-graph cache
  writes.
- **mRoPE** for the attention layers, and **four position values per token** —
  ggml asserts `ne[2] * 4 == b->ne[0]` and aborts otherwise.
- **The fused query/gate projection**, interleaved *per head*: head 0's query,
  head 0's gate, head 1's query. Two contiguous halves would take the queries of
  the first heads and the gates of the last, silently.
- **The `qwen35` pre-tokenizer** — one digit at a time, and combining marks
  belonging to the word rather than the punctuation beside it. Checked against
  `llama-tokenize`.

### Three detections that read a familiar name and got the wrong architecture

All three found by the layer diff, all three now testing for what they mean
rather than for one tensor name:

| flag | what went wrong |
|---|---|
| `parallel_residual` | absence of `ffn_norm` meant "one norm, parallel block". `qwen35` has two norms and calls the second `post_attention_norm`, so the FFN consumed `attn_norm(x)` — the value attention had already seen. The symptom was `attn_post_norm-0` coming out **identical to llama.cpp's `attn_norm-0`**, which is what named the bug. |
| `post_norms` | presence of `post_attention_norm` meant Gemma. Gemma's post-norms are a *pair*; demanding the missing `post_ffw_norm` refused a loadable model. |
| `fused_qkv` | presence of `attn_qkv` meant Phi-3. On `qwen35` that tensor is the delta net's input projection, and attention layers have no such tensor at all. |

### The bug is found: a tensor read back that was never computed

The attention layers' **output gate** is a *sibling* view of the `attn_q` matmul,
not an ancestor of q, k or v — so a graph rooted at those three never computed
it, and `to_vec_f32` returned whatever the reused scratch arena held. The
previous layer's leftovers.

That explains every symptom:

- Layers 0–2 are recurrent and have no gate, so they matched llama.cpp exactly;
  layer 3, the first attention layer, was wrong.
- Any extra compute anywhere changed the leftovers and so changed the answer —
  which is why `CHAOS_DUMP_LAYERS=1` looked like it *fixed* the model.
- Bisecting the debug computes pointed at the dense-FFN pass, three phases away
  from the actual fault. The trigger and the bug were in different places.

The fix is one line of intent: **the gate joins q, k and v as a root of both
`realize_graph` and `run`.** `qwen35.rs` carries a textual regression test for
it, because the failure is structural — it lives in which tensors `stream.rs`
hands to the compute.

### Verified, with the instrumentation off

| check | result |
|---|---|
| all 24 layers, by value and by sum | match `llama-eval-callback` |
| 1-token prompt `"Paris"` | `", France.\nThe first time I"` — identical |
| 5-token prompt | `" Paris.\nThe capital of France is"` — identical |
| 22-token prompt | `"</think>\n\n The sky appears"` — identical |
| debug dump on vs off | **same output** |

The three prompt lengths are not decoration: the fused delta rule takes a
different path at one token than at many, so both regimes are covered.

**`qwen35` is in `VERIFIED_ARCHITECTURES` and `RUNNABLE_ARCHS`** — Qwen3.5,
Qwen3.6 and Qwen3.8 all report it, and `chaos-run` and `chaos-serve` take them
without `--force`. `qwen35moe` stays out: no MoE container of the family has been
run, and its refusal now names what is *untested* rather than what is
unimplemented.

### Three detections that read a familiar name and got the wrong architecture

All three found by the layer diff, all three now testing for what they mean
rather than for one tensor name:

| flag | what went wrong |
|---|---|
| `parallel_residual` | absence of `ffn_norm` meant "one norm, parallel block". `qwen35` has two norms and calls the second `post_attention_norm`, so the FFN consumed `attn_norm(x)` — the value attention had already seen. The symptom was `attn_post_norm-0` coming out **identical to llama.cpp's `attn_norm-0`**, which is what named the bug. |
| `post_norms` | presence of `post_attention_norm` meant Gemma. Gemma's post-norms are a *pair*; demanding the missing `post_ffw_norm` refused a loadable model. |
| `fused_qkv` | presence of `attn_qkv` meant Phi-3. On `qwen35` that tensor is the delta net's input projection, and attention layers have no such tensor at all. |

### Not finished, and the reason is worth reading

**`CHAOS_DUMP_LAYERS=1` changes the answer.** With it set the sampled ids are
llama.cpp's; without it they are wrong. Same binary, same prompt, `--temp 0`.

So the layer diff above is real *and* insufficient: the debug pass computes extra
tensors in the same arenas, which changes when a buffer is written, and the plain
path reads one at the wrong moment. **A model whose output depends on whether it
is being observed is not verified**, so `qwen35` is out of
`VERIFIED_ARCHITECTURES` and `RUNNABLE_ARCHS` — it was briefly added during this
work and has been taken back out.

Ruled out by measurement: thread count (1, 2 and 4 all identically wrong),
`--force`, computing the layer twice, `scores` as an explicit graph root, `cont`
copies of both carried states, and a `cont` copy of the convolution's input. The
delta net's *own* debug pass makes no difference either — so the trigger is one
of the three extra computes in the **attention** and dense-FFN phases, and the
first suspect is the strided fused query/gate views. The bisect and the next step
are written down in `docs/graph/backlog/qwen35-gated-delta-net.md`.

`CHAOS_DUMP_LAYERS=1` prints the residual stream per layer in
`llama-eval-callback`'s own format and names, which is how all of the above was
found.

### Qwen3.8-27B is in the catalogue, and it is Qwen3.6's architecture

Read from the container, not inferred: `general.architecture qwen35`, 866
tensors, 51 metadata keys, 17.9 GB dense. Upstream it is
`Qwen3_5ForConditionalGeneration` with a vision tower and a separate 928 MB
`mmproj`. Asking for the newer model does not route around the gated delta net —
it adds to it. A test asserts 3.6 and 3.8 agree on the architecture.

### The delta rule is one ggml op, and it is bound

`ggml_gated_delta_net` performs the whole chunked delta scan and returns the
carried state with the scores. It is present in the pinned `ggml-base.a`, and it
is now bound along with `ggml_ssm_conv`, `ggml_l2_norm`, `ggml_rope_multi` and
the 4-D view/reshape/cont/repeat helpers.

**Three numeric tests rather than a compile check**, because a wrong FFI
declaration mis-reads its arguments and returns confident numbers: `l2_norm`
takes four 2s to 0.5 where `rms_norm` would give 1.0; `ssm_conv` sums its rolling
window to 10 then 14; `gated_delta_net` returns `S*H*T*N + S*S*H*N` finite values
with the state moved off zero.

**Qwen3.8 does not run yet** — the layer wiring, the recurrent state cache and
the llama.cpp diff are still to come.

## [0.0.8] — 2026-08-19

Every dropdown on the settings page was two pixels tall. A model in its own
folder was invisible. A download that stopped half way looked finished. All
three were reported by Atur in one message, all three are fixed, and all three
now have a test.

### The settings dropdowns open

**Measured, on this machine: the dropped list was 32 pixels tall in 0.0.7 and
is 238 in 0.0.8.** A combo box in Win32 is sized by the height of its *dropped
list*, not of its closed box, and `layout` was passing it the row height. The
options were all there — `CB_GETCOUNT` reported three to six of them — and the
list they opened into had two pixels of room after the closed control took its
share. Clicking one looked exactly like a control that ignores the mouse.

`metric::COMBO_VISIBLE` now sizes them for eight rows, and the constant carries
the reason so nobody sets it back.

### Models are found where models are actually kept

- **The `models folder` setting is read by the engine.** It was written to
  `settings.txt` by the app and consulted by nothing — the worst of the three
  possible behaviours, because it looked like it worked. `chaos_model::find`
  reads it now, so `chaos-run` and `chaos-serve` honour it too.
- **Several folders, separated by `;`.** A 144 GB container does not live beside
  a 2 GB one. Both `CHAOS_MODELS` and the setting take a list, in the platform's
  own separator.
- **One level of subfolders is searched.** A five-shard model lives in its own
  directory; a scan that stopped at the top level reported "no models installed"
  with 145 GB plainly there. This is how DeepSeek-V4-Flash became visible in the
  app without moving a byte.
- The search order holds each directory exactly once. `Vec::dedup` removes only
  *adjacent* duplicates, so the cache directory used to appear twice.

### An unfinished download says so

A truncated `.gguf` has a valid header — the header is written first — so it
listed beside models that work and failed several seconds into a load, in the
engine's words. Three of the models on Atur's machine were in that state.

`chaos_gguf::Gguf::expected_file_bytes` reads the container's own tensor index
and returns where the last tensor ends; a file shorter than that is *provably*
truncated, with no catalogue, no network and no guess involved. Every shard is
checked separately.

- **MODELS** marks such a row `(unfinished)`.
- **LOAD** refuses, naming the shortfall, and points at DOWNLOAD — which
  resumes.
- `chaos-run` and `chaos-serve` refuse the same way, before anything is bound:
  `the download did not finish -- phi-4-Q4_K_M.gguf is 2.48 GB short of the
  8.89 GB its own index requires (72% written)`.

### Ideogram 4 is in the catalogue, and listed as an image model

It is open-weight — a 9.3B diffusion transformer, GGUF conversions and all — so
"we cannot get it" was never true. What is true is that generating an image
needs a sampler loop, a separate text encoder and a VAE, and Chaos is a token
loop with none of the three. The container says as much itself: **458 tensors
and zero metadata keys**, so there is no `general.architecture` to dispatch on.

Listed, sized from the repository, and refused with the reason. The route in, if
it is ever wanted, is `docs/graph/backlog/image-generation-ideogram-4.md`.

### Why Qwen3.5/3.6 cannot run, stated properly

The old refusal blamed interleaved multimodal RoPE. That is real and it is the
smaller half. Read from the container itself: `full_attention_interval 4` with
64 blocks means **48 of the 64 layers are a gated delta net with recurrent
state**, carrying `ssm_conv1d`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_dt.bias`
and `ssm_norm` — and a KV cache cannot stand in for a carried state.

The refusal now says that, and `docs/graph/backlog/qwen35-gated-delta-net.md`
writes down the four pieces of work and how they get verified.

### Phi-4 runs: the `dbrx` pre-tokenizer

Phi-4 downloaded completely and then refused to load — `tokenizer.ggml.pre =
"dbrx" is not implemented`. Its expression is `llama3`'s byte for byte, and
llama.cpp says so in a comment above it.

**It is still a separate variant rather than another name in the `llama-bpe`
arm**, because llama.cpp's `llama3` branch also sets `ignore_merges` and
`add_bos` and its `dbrx` branch sets neither. Phi-4 declares no
`tokenizer.ggml.add_bos_token`, so aliasing it would have prepended a BOS from
the default and shifted every position by one — silently, the way tokenizer bugs
always fail here.

Checked against `llama-tokenize` on the container: six cases, exact ids, and
eleven tokens with no BOS for the test sentence. `chaos-run phi-4 --chat` then
answers *"The capital of France is Paris."*

**`smaug-bpe` shares llama.cpp's arm and is still refused**, because there is no
Smaug container here to check it against. Identical in the source is not the
same as verified.

### Verified by running them

Every model on Atur's machine was run through the **installed** v0.0.8 — what
he has, not what the repository has.

**Ten of the eleven generate correct text**: Llama-3.2 1B and 3B,
Qwen2.5-Coder-7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, Gemma-3 4B, 12B and 27B, and
Phi-4. **DeepSeek-V4-Flash generates too** — *" Paris."*, 0.340 tok/s, 5 prompt
tokens in 8.5 s — from `C:\Projects\models\v4flash`, without a byte being
moved.

The eleventh is Qwen3.6-27B, refused by architecture with the reason. Nothing
crashed and nothing hung.

## [0.0.7] — 2026-08-19

The window redesigned around four pages, an installer that looks like one, and
a mark drawn from vectors. Every resident figure in the catalogue is now
measured rather than estimated — including the one this project's headline
claim rests on.

### The app is four pages, not one panel

Atur's verdict on 0.0.6 was exact: *"why is all click in one slot"*. Everything
had been on one screen because the window grew a control at a time.

- **CHAT · MODELS · MONITOR · SETTINGS**, reachable from a navigation rail, a
  real **menu bar**, or `Ctrl+1`..`Ctrl+4`. One page owns the screen.
- A **strip on every page** carries what is running, its endpoint, its
  throughput and STOP.
- **A page per model** rather than a row: status, endpoint, context, threads,
  cache, uptime, tokens served.
- **SETTINGS exposes all nine fields** the file holds; the old window showed
  three. The five that need knowledge are **dropdowns generated from your
  machine** — core count, free memory, whether a GPU exists — each option
  carrying one line on what it costs, and "Measured" always first.
- Light and dark, following Hermes' desktop palette with Atur's `#0000F2`.

### The installer looks like an installer

Navy `#0d2f86`, cream `#ffe6cb`, a tracked display-serif wordmark and one
bordered action — Hermes' own installer values, read from its source. The
install is **twenty named steps on a worker thread** with a progress bar, ticks
and per-step timings, instead of a frozen window that returned a paragraph when
it was over. It also has a panic hook and writes `%TEMP%\chaos-setup.log`.

### The mark is geometry

`assets/logo.svg` ships as **44 polygons, 11,094 points**, filled by a scanline
rasteriser at whatever size is wanted. No bitmap anywhere. The app icon is a
real rounded square with an alpha channel — it was RGB before, so a rounded
corner had to be filled with something and any fill read as a square.

### Added

- **An API key that is actually checked.** `chaos-serve --api-key` requires
  `Authorization: Bearer` on `/v1/*` and answers 401 in the shape an OpenAI
  client expects. Off by default; `/health` is never gated. Generated from the
  system CSPRNG, not the clock.
- **Model → Test the connection** makes the three requests an agent makes and
  reports each. `docs/AGENTS.md` carries recipes for Hermes, aider and the
  OpenAI-compatible provider every editor extension has.
- **Download progress**: percent, bytes, rate and time left, measured from the
  files on disk so a resumed fetch reports correctly.
- **Linux packages**: a `.deb` and an AppImage, built from coreutils alone and
  verified in CI by installing the deb, running `chaos-run` off PATH, and
  executing `chaos-probe` out of the AppImage.
- **`tools/gguf-always-read.py`** reads a container's tensor table over HTTP and
  computes its resident set exactly, without downloading the model.

### Fixed

- **A model the engine cannot run was started anyway.** Qwen3.6 showed a green
  dot while the server had already exited, so the next message failed with
  "connection actively refused". The architecture is read from the container's
  header before a server is started, and the refusal says why.
- **A dead engine went on being reported as running.**
- **The window flickered once a second** — the timer repaint covered the child
  controls. `WS_CLIPCHILDREN`, on both windows.
- **`DrawTextW` with an empty string killed the installer.** `Vec::as_ptr` on an
  empty vector is dangling and Windows dereferences it; the report has blank
  lines in it.
- A worker thread cannot show a message box owned by the UI thread — it did
  nothing at all.

### Corrected

- **V4-Flash's resident set was `7_925_000_000`, a round guess.** Measured
  across all five shards: **7,920,157,020**. The estimate was lucky to 0.06%,
  and it is the number the project's headline rests on.
- Qwen3-30B-A3B was `1_000_000_000`; measured **997,554,176**. A test now
  rejects any Mixture-of-Experts resident figure that looks rounded.

### Known limits

- **Qwen 3.5/3.6 (`qwen35`, `qwen35moe`) cannot run.** They are hybrid
  architectures — recurrent state plus multi-token prediction — not plain
  transformers. They are listed in the catalogue and refuse with the reason.
- The menu bar stays light in dark mode; Windows draws it and
  `SetPreferredAppMode` measurably does nothing on 10.0.26200.
- MONITOR cannot show streamed bytes or cache residency; the engine measures
  them but does not report them over the socket.

## [0.0.6] — 2026-08-18

The app in 0.0.5 could not be used. This release is that, fixed, plus what a
model runner has to show.

### Fixed — the app died on the first click

**Clicking INSTALLED or AVAILABLE killed the process instantly**, with no
window, no message and no log.

`WM_CTLCOLORLISTBOX` borrows the window's state, and the code that filled the
list held a mutable borrow while calling `SendMessageW` — which dispatches that
message *synchronously*. A `RefCell` double borrow, and `panic = "abort"` turns
a double borrow into immediate process death. It could never have worked.

Six places had that shape. Three were found by a test written afterwards rather
than by clicking, and one of them was **UNLOAD** — the button that frees the
model's memory.

Two more, in the same family:

- **The GUI uninstall removed nothing.** It launched the helper and stayed open,
  so the helper could not delete the folder the window was running from and gave
  up. The window now exits after reporting.
- **Closing the window left the engine running**, holding every resident byte —
  7 GiB for V4-Flash — with nothing left to stop it from. Closing Chaos now
  stops the model.

**A crash now says something.** Under `abort` with no console there was nothing
at all; the app writes `%TEMP%\chaos-app-crash.log` and shows a message box
naming it.

### Added — icons

`chaos-setup.exe`, `chaos-app.exe` and the window itself carry the logo, at
**nine sizes from 16 to 256**, each rendered from `assets/logo.svg` at its own
resolution. Windows downsamples one large icon badly and this mark is thin
radiating lines.

### Added — what the app was missing

- **The endpoint.** `running <model> -> http://127.0.0.1:8231/v1`, so a coding
  agent can be pointed at it. There is no API key: the server binds localhost
  only.
- **DELETE**, which removes *every shard*. Deleting one file of a five-shard
  container would leave 120 GB of unusable data and report success.
- **Live memory**, free of total.
- **Settings that persist**, in `%USERPROFILE%\.chaos\settings.txt` — outside
  the install, so upgrading or uninstalling never takes them. Unknown keys are
  preserved, so an older build cannot discard a newer one's preferences.
- **A sidebar that scales with the window.** At a fixed width the model rows
  were clipped mid-word and the fit verdict — the number that decides whether a
  model runs — could not be read.

### Added — the installer says what it did

Install and uninstall end with a report rather than the window closing: what was
written, where, what was removed, and what was deliberately kept. Running a newer
setup over an older install names the upgrade.

### Added — `docs/APP.md`

A manual for the window, including a section on what it does not do yet.

### Changed

- CI starts `chaos-app`, waits, and fails if it exited or left a crash log —
  and checks no `chaos-serve` survives it. An app that only *builds* is what
  shipped 0.0.5.
- The documented test count is checked against the suite that actually ran.

### Known

**Windows SmartScreen will warn** that the publisher is unknown. That is what
Windows says about every unsigned application; choose *More info → Run anyway*.
Signing needs a certificate, which is a purchase, not a patch.

One model runs at a time; there is no per-model window yet.


## [0.0.5] — 2026-08-18

### Fixed — uninstalling actually uninstalled

**In 0.0.4, uninstalling from Add/Remove Programs removed nothing and reported
success.** A running executable cannot delete the directory it lives in, and the
installer copies itself into `bin` so Windows has something to launch — so the
uninstaller ran from inside the very folder it was trying to remove, the delete
failed, and the code said "uninstalled" anyway.

Three things were wrong and all three are fixed:

- The uninstaller now **re-runs from a copy in the temp directory**, spawned
  **detached** so the original can exit. Waiting for it was the first attempt and
  it failed the same way: the parent stayed alive holding the file open for the
  whole of the child's run.
- The staged copy **retries for ten seconds** rather than trying once, because
  the process that asked for the uninstall may still be exiting.
- It **checks before reporting.** If files remain it says how many and where,
  instead of claiming success over a full directory.

Also: the silent log no longer recreates the folder it just deleted. A clean
uninstall used to leave an empty `Chaos` directory containing one file that said
it had uninstalled.

Verified from the default prefix, the path that was broken: install 17 files,
uninstall from the installed copy, everything gone in one second, PATH and
registry clean.


## [0.0.4] — 2026-08-18

### Added — one file to install everything

**`Chaos-0.0.4-Setup.exe`.** Download it, run it, press INSTALL. 23 MB with every
binary inside: no archive to unpack, no PowerShell, no toolchain, no network, no
administrator rights. It installs per-user, puts Chaos on your PATH, creates the
models folder, adds a Start Menu entry and registers in Add/Remove Programs.

Built without NSIS, WiX, Inno or MSI tooling, because every one of them would
have to be installed on the build machine before a release could be cut, and this
project has no dependencies. A Windows install turns out to be a window, a file
copy, a PATH entry, a shortcut and one registry key.

**Uninstalling never touches your models.** They live outside the install folder
on purpose, and a test enforces it — the failure mode is deleting a 155 GB
download.

Silent mode for scripting and CI: `/S`, `/S --uninstall`, `--prefix <dir>`.
**Wait for it** — the installer is a window-subsystem binary, so a shell gets
control back immediately and never sees an exit code:

```powershell
$p = Start-Process Chaos-0.0.4-Setup.exe -ArgumentList '/S' -Wait -PassThru
$p.ExitCode
```

### Added — Chaos as a window

**`chaos-app`**, a native Win32 application. Not a browser in a frame: a real
window, drawn with GDI, in two colours and nothing between them. Pick a model,
LOAD it, chat with it, UNLOAD it — and unloading genuinely frees the memory,
because the engine runs as a child process rather than inside the window.

INSTALLED and AVAILABLE tabs, a DOWNLOAD button, and settings for cache, threads
and port.

### Added — a browser interface for `chaos-serve`

`GET /` now serves a chat page, self-contained in the binary: no CDN, no font, no
script fetched from anywhere. An offline machine gets the whole interface.

### Added — 13 models to fetch, up from 2

Qwen3 4B/8B/14B/32B, Gemma-3 4B/12B/27B, Llama-3.2 1B/3B, Qwen2.5-Coder-7B and
Phi-4, alongside DeepSeek-V4-Flash and Qwen3-30B-A3B. Every repository, filename
and byte count was read from the Hugging Face API and verified to resolve before
being added.

**Each entry states what must stay resident, not just what it downloads**, and
that is the number the fit verdict uses:

```
v4flash    155.1 GB    7.38 GiB resident   -> streams on a 16 GB machine
qwen3-32b   19.8 GB   18.40 GiB resident   -> does not
```

A dense model has no routed experts, so nothing streams and the whole file has to
fit. Sorting by download size would have called the 155 GB model impossible and
the 20 GB one easy, which is backwards.

### Changed

- `chaos-iobench` and `chaos-gpubench` now ship in the release, so the
  measurements this project publishes can be reproduced by anyone who downloads
  it.
- The release workflow builds and tests the installer, and fails if it embedded
  nothing.

### Fixed

- The documented test count is now checked by CI against the suite that actually
  ran. It had gone stale three times — 566, 570, 575 — each caught only by
  someone noticing.


## [0.0.3] — 2026-08-16

### Changed — the project is now called `chaos`

`bigtea-run` is `chaos-run`, and so are the other ten binaries. Crate names,
`--help` text, info lines, environment variables (`BIGTEA_*` → `CHAOS_*`), the
workflows and all 105 documents were renamed in one pass — 1,623 occurrences
across 184 files, none left. The git remote is deliberately untouched; the
`repository`/`homepage` URLs point at the new name and start resolving when the
repository is renamed.

**This is a breaking change for anything scripted against the old names**, which
is what a `0.0.x` minor is for.

### Added — running it no longer starts with a path

**`chaos-run <name>`.** Every command used to begin with an absolute path to a
`.gguf` file, which on Windows means something like
`C:\Users\you\.chaos\models\Qwen3-30B-A3B-Q4_K_M.gguf` typed by hand, and for a
five-shard container it means knowing which shard to name. Now any unique part
of a name resolves:

```
chaos-run                                   # lists the models you have
chaos-run qwen3 "The capital of France is"  # runs Qwen3-30B-A3B-Q4_K_M.gguf
chaos-run deepseek "..."                    # opens shard 1 of 5, automatically
```

An existing path still wins, so nothing that worked before changes. An ambiguous
name lists the candidates rather than guessing a 144 GB read, and an unknown one
lists what *is* available rather than leaving the user to go looking.
`chaos-serve` has the same lookup from the same code, so the two cannot disagree
about where models live.

Searched in order: `CHAOS_MODELS`, `~/.chaos/models` (which `install.ps1`
creates), the download cache `chaos-pull` writes to, and `./models`. **Two of
those already existed and pointed at different places** — where a model lived
depended on how it had arrived — which is exactly the kind of thing a first-time
user should never have to learn.

### Added

- **A startup logo**: the name, then the logo centred beneath it, then the
  version. Rasterised offline into 3 KB of committed luminance bytes and printed
  with Unicode half-blocks, two pixels to a cell. Cropped to the artwork rather
  than the SVG's canvas, which had been carrying a wide white margin into every
  render. No SVG parser, no
  image decoder, no build script, no dependency — the workspace still has zero.
  It sizes itself to the terminal and is skipped for `NO_COLOR`,
  `CHAOS_NO_BANNER`, `--log-disable`, a terminal too small, and any stdout or
  stderr that is not a terminal.
- **`scripts/install.ps1`** — Windows install and in-place upgrade. Copies the
  binaries to `%LOCALAPPDATA%\Chaos\bin`, adds it to the *user* PATH exactly
  once, and creates `%USERPROFILE%\.chaos\models`. Re-running upgrades and
  removes binaries the new version no longer ships; `-Uninstall` reverses it and
  never touches the models directory. It refuses rather than half-upgrading when
  a binary is running. Shipped inside the Windows archive and smoke-tested in
  the release workflow, on the unpacked archive, in the shape a user meets it.
- `chaos-serve` and `chaos-pull` are now in the release archives. They were
  built and never packaged.

### Corrected

**V4-Flash is at parity with llama.cpp, not far behind it.** The published
figures — prefill 1.62x behind, generation 3-4x behind — date from 2026-08-07 and
no longer reproduce. Three alternating pairs in one session, both engines at
their defaults:

| DeepSeek-V4-Flash | Chaos | llama.cpp |
|---|---:|---:|
| prefill, ms per prompt token | **1640** | 1679 |
| generation, tok/s | **0.394** | 0.39 |

The warm-up run, discarded, read llama.cpp at 0.23 tok/s — which would have made
this a 1.7x lead. It is not one. See
[`where-we-stand-vs-llamacpp-2026-08-16.md`](docs/graph/research/where-we-stand-vs-llamacpp-2026-08-16.md).

**The parallel-experts optimisation does not transfer to V4-Flash, and the
ceiling is measured: the entire routed expert arithmetic is under 5% of a
token.** A token is 67% expert-slice read, 17% block compute, 16% routing. The
block's single `compute` had been folded into the residual of the phase table,
which is why that split had never been written down.

### Corrected — earlier

**The hot expert set is per-prompt, so it cannot be pinned.** v0.0.2's routing
figures were all scored *in-sample on a single prompt*. Re-measured on eight
prompts across four subjects, with the token-id-routed layers 0-2 excluded and a
uniform-router null at matched sample size:

| published in 0.0.2 | measured |
|---|---|
| top-64 = 97.8% of selections | **90.5%** in-sample, **53.7%** on a prompt the set was not chosen from |
| 33.6 tok/s disk floor at 34.27 GiB | **1.60 tok/s** |
| 20 tok/s needs a ~48 GiB desktop | unsupported — needs a 96.3% hit rate; a pinned cache gives 76.7% at 68.5 GiB |
| chi-square 7805 | not a valid statistic — generation re-runs prefill per token, so the prompt was counted once per pass (1282 → 5464 → 11469 for 1, 4, 8 passes, with coverage unmoved) |

The skew itself is real and reproduced on every prompt: top-8 of 256 takes
34.6–52.0% of selections against a uniform null of 6.8–7.4%. What does not hold
is *transfer* — across subjects a pinned hot set scores 37.5% against 25.0% for
caching at random. See
[`routing-skew-is-per-prompt-2026-08-08.md`](docs/graph/research/routing-skew-is-per-prompt-2026-08-08.md).

### Added

- `CHAOS_ROUTING_DUMP=<path>` writes raw `pass,layer,expert,count` rows, so two
  runs can be compared offline and passes are not conflated.
- `tools/routing/` — the prompts, capture script and analysis behind the above.
- `STATUS.md` — one canonical statement of where the project stands and what
  remains, so any session can resume without reconstructing it.

### Planned

Everything the previous list named is done — the KV cache, the downloader, the
OpenAI-compatible server, quant selection from the probe and prebuilt binaries.
What replaces it, in the order the measurements justify:

- **The tok/s-versus-RAM frontier for a 144 GB model.** Nobody has published it,
  and only an engine that owns residency can sweep it — `mmap` cannot be told to
  use exactly N GiB.
- **Verify the GPU tier.** `--device`, `-ngl`, `-ot` and `--op-offload` all work
  on Vulkan, and the device path fails 1 of 8 parity prompts where the CPU path
  fails none. Shown to be arithmetic rather than wiring, but unproven either way.
- **More architectures.** 13 of llama.cpp's 141 have been diffed against it.
- **Not** 20 tok/s on V4-Flash. That is closed by measurement rather than
  deferred: it needs 79 MB/token and the model reads 3288.

## [0.0.2] — 2026-08-07

Findings, a retraction, and the measurement that changes the project's direction.

> **⚠ Superseded 2026-08-08.** Every routing figure in this entry was scored
> in-sample on one prompt and four of them are wrong — see **Corrected** under
> [Unreleased](#unreleased). The entry is left as released rather than rewritten.

### Added

- `CHAOS_ROUTING=1` prints how often each expert of each layer is actually
  selected, and what the hot set would cost to keep resident.

### Discovered

**DeepSeek-V4-Flash's router is violently skewed.** Every speed estimate this
project ever made assumed it spread evenly over 256 experts:

| top-N per layer | share of selections | resident cost |
|---:|---:|---:|
| 1 | 12.1% | 0.54 GiB |
| 8 | 52.9% | 4.28 GiB |
| 16 | 70.4% | 8.57 GiB |
| 64 | **97.8%** | 34.27 GiB |

Uniform routing would give top-16 = 6.2%; measured 70.4%, chi-square 7805 against
uniform's ~255. With a hot-set cache, bytes read per token fall from 3.21 GiB to
**72 MiB** — a 33.6 tok/s disk floor, against a 27 tok/s compute floor.

**20 tok/s for a 144 GB model is a cache-sizing problem, not a physics
violation**, and it needs roughly a **48 GiB desktop** rather than the ~150 GiB
previously claimed. On a 15.7 GiB laptop the same arithmetic implies ~1.3 tok/s,
about 4x llama.cpp. Neither is measured yet; both are arithmetic on measurements
that are. See
[`routing-skew-changes-everything.md`](docs/graph/research/routing-skew-changes-everything.md).

### Retracted

**v0.0.1's claim that Chaos leads llama.cpp on DeepSeek-V4-Flash.** It claimed
3.0x faster load and 1.20x faster prefill. Both were false: Chaos's numbers were
measured fresh and llama.cpp's were copied from a two-day-old document taken under
different free-RAM conditions, so the engines were never run back to back. Run
back to back, twice:

| | Chaos | llama.cpp |
|---|---:|---:|
| load | 10.0s | 10.5s |
| prefill, per prompt token | 2440 ms | **1503 ms** |
| generation | 0.064 tok/s | **0.21–0.31 tok/s** |

**Chaos leads on nothing on this model.** It remains ahead on Qwen3-30B-A3B
prefill at 565 and 2206 tokens, measured back to back.

## [0.0.1] — 2026-08-07

Performance. DeepSeek-V4-Flash prefill is **2.2x** faster than v0.0.0 and
generation **1.83x**, with every one of the 14 oracle tests still matching
llama.cpp's element sums.

### Changed

- **One graph evaluation per block instead of 24.** `Context::compute` evaluates
  a tensor's *entire ancestor graph*, so calling it on every intermediate does
  not merely dispatch more work — it **re-does** the work, once per call, and
  pays a graph build and a threadpool cycle each time. A value is now computed
  only where the CPU must read it. Worth **1.9x**, and invisible on a long
  prefill because the matmuls there are large enough to bury it.
- **A layer's three expert tensors are read in one parallel batch.** Four
  readers, jobs distributed one slice at a time so each reader gets an equal
  share of the bytes. Parallel reads had been tried and reverted twice before;
  the difference is batch size — per-tensor groups are 6 slices at generation
  time, and the thread spawns cost more than the queue depth buys.
- `CHAOS_THREADS` selects the thread count per graph evaluation, and
  `CHAOS_BLOCK_TIMING` now reports each phase of a block separately.

### Performance

DeepSeek-V4-Flash, same machine, both engines' command lines and outputs in
[`v4flash-vs-llamacpp-2026-08-07.md`](docs/graph/research/v4flash-vs-llamacpp-2026-08-07.md):

> **⚠ Retracted the same day.** This section originally claimed 3.0x faster load
> and 1.20x faster prefill. Both were wrong: Chaos's numbers were fresh and
> llama.cpp's were copied from a two-day-old document taken under different
> free-RAM conditions, so the two engines were never run back to back. Corrected
> figures, measured back to back twice:

| | Chaos | llama.cpp | |
|---|---:|---:|:--|
| load | 10.0s | 10.5s | parity |
| prefill, per prompt token | 2440 ms | **1503 ms** | llama.cpp 1.62x faster |
| generation | 0.064 tok/s | **0.21-0.31 tok/s** | llama.cpp 3-4x faster |

**Chaos leads on nothing on this model.** The speedups below are real and
measured against Chaos's own previous version; they simply did not close the gap.

A single-token forward pass costs **4.0s**. That is what one step of a KV-cached
loop will cost — 0.25 tok/s — and it is the number to plan against, because the
0.077 above is an artefact of re-running the whole sequence for each token.

### Fixed

- macOS: process enumeration read `/proc`, which does not exist there, so the
  "close these apps to free RAM" advice silently did nothing. Falls back to `ps`.
- macOS: Accelerate framework was never linked, though ggml's cmake enables it by
  default and calls vDSP.
- macOS: OpenMP was demanded unconditionally; AppleClang ships none.
  `CHAOS_GGML_OPENMP` overrides the per-platform default.
- The documented `cmake` line built **shared** ggml libraries, so a new user
  following the README got no `.a` archives at all.
- `chaos-arch` now fails with one actionable message when ggml is missing,
  instead of a wall of unresolved imports.
- Declared MSRV was 1.74 while the code used a 1.82 API. Now 1.82.

## [0.0.0] — 2026-08-07

First public release. The engine works and is verified; the product around it is
not built yet. See [README](README.md#status) for what is and is not there.

### Added

- **Runs Mixture-of-Experts models several times larger than RAM, on CPU.**
  Always-read weights stay resident; routed experts stream from disk as routing
  selects them.
- **DeepSeek-V4-Flash (`deepseek4`) support** — 43 blocks, hyper-connections,
  three kinds of compressed attention (raw, compressed-sparse, heavily
  compressed), hash routing on the first three layers and biased top-k routing on
  the other 40. Verified element-by-element against llama.cpp on all 43 blocks
  plus the output head.
- **Qwen3 and Qwen3-MoE support**, with a frequency-gated expert cache.
- **Cache-bypassing direct I/O** (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`), falling
  back to buffered reads and *reporting* that it did rather than pretending.
- **Zero-copy expert reads.** `SkewedBuf` deliberately misaligns the destination
  buffer to match the file's sector residue, because GGUF pads tensor data to 32
  bytes rather than to a disk sector — so a conventionally aligned buffer can
  never receive a direct transfer. 0.80 → 1.58 GiB/s, with 0.09% of bytes copied
  instead of 300%.
- **Residency with a hard budget**, which reports what did not fit, what
  re-reading it costs per token, and which processes to close to fix it.
- `chaos-run` — prefill and generation.
- `chaos-probe` — RAM, disk, GPU, and what to close.
- `chaos-model-info` — fit prediction and tok/s estimate before running.
- `chaos-meta`, `gguf-info`, `chaos-loadbench` — container and I/O inspection.
- 157 unit tests and 16 container-backed tests.

### Performance

Measured on one machine (15.7 GiB RAM, NVMe at 2.55 GB/s, 20 threads). Both
engines produce identical, correct output; llama.cpp is measured with a warm page
cache. Full command lines and outputs in
[`head-to-head-llamacpp-2026-08-05.md`](docs/graph/research/head-to-head-llamacpp-2026-08-05.md).

Qwen3-30B-A3B Q4_K_M prefill, Chaos / llama.cpp:

| tokens | Chaos | llama.cpp |
|---:|---:|---:|
| 565 | **27.64** | 23.55 |
| 2206 | **36.60** | 33.59 |
| 4395 | 38.40 | 40.25 |
| 8775 | 34.88 | 35.01 |
| 4395 (`-b 4096`) | **43.61** | 40.25 |

### Known limitations

- **Generation is slower than llama.cpp.** DeepSeek-V4-Flash: 0.077 tok/s against
  0.45, because the V4-Flash path has no KV cache yet and each token re-runs the
  whole sequence. Qwen3-30B-A3B: 1.07 against 2.16, about 2x. On V4-Flash
  **Chaos leads on nothing** — see the retraction above. It is ahead only on
  Qwen3-30B-A3B prefill at 565 and 2206 tokens.
- **Linux and macOS build and pass the unit tests in CI, but no model has been
  run on either.** macOS additionally has no direct-I/O path — `F_NOCACHE` needs
  an `fcntl` after opening and is not written yet — so it falls back to buffered
  reads and the page-cache problems this design exists to avoid.
- No model downloader; bring your own `.gguf`.
- No server or API.
- No prebuilt binaries; ggml must be built first and `GGML_LIB_DIR` set.
- Only `qwen3`, `qwen3moe` and `deepseek4` architectures run. Others parse as
  containers but will not execute.

### Retracted

- **"llama.cpp cannot run models larger than RAM."** It can, with `--no-repack`.
  This claim survived several days on a misattributed error string because nobody
  ran the opposing command. It is retracted in writing, and the project now
  requires a competitor's exact command line and output before any competitive
  claim is citable.

[Unreleased]: https://github.com/aturzone/Chaos/compare/v0.0.12...HEAD
[0.0.12]: https://github.com/aturzone/Chaos/releases/tag/v0.0.12
[0.0.11]: https://github.com/aturzone/Chaos/releases/tag/v0.0.11
[0.0.10]: https://github.com/aturzone/Chaos/releases/tag/v0.0.10
[0.0.8]: https://github.com/aturzone/Chaos/releases/tag/v0.0.8
[0.0.7]: https://github.com/aturzone/Chaos/releases/tag/v0.0.7
[0.0.6]: https://github.com/aturzone/Chaos/releases/tag/v0.0.6
[0.0.5]: https://github.com/aturzone/Chaos/releases/tag/v0.0.5
[0.0.4]: https://github.com/aturzone/Chaos/releases/tag/v0.0.4
[0.0.3]: https://github.com/aturzone/Chaos/releases/tag/v0.0.3
[0.0.2]: https://github.com/aturzone/Chaos/releases/tag/v0.0.2
[0.0.1]: https://github.com/aturzone/Chaos/releases/tag/v0.0.1
[0.0.0]: https://github.com/aturzone/Chaos/releases/tag/v0.0.0
