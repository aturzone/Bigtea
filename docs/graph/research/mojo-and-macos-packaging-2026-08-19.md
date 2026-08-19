# Mojo, and how a macOS build would actually be made — 2026-08-19

Two questions Atur asked, answered from the sources rather than from
impressions. Both answers are **no, and here is the specific reason**, which is
more useful than a maybe.

Links: [`hard-won-facts`](../reference/hard-won-facts.md) ·
[`gpu-does-not-help-streaming-moe-2026-08-16`](gpu-does-not-help-streaming-moe-2026-08-16.md) ·
[`v4flash-ram-frontier-2026-08-16`](v4flash-ram-frontier-2026-08-16.md)

---

## 1. Mojo: real, and not what this bottleneck is

### What is actually true about it

- **Fully open source under Apache 2.0**, compiler and toolchain included, after
  reaching 1.0 with source stability. Earlier phases opened the standard library
  (2024) and the kernel code; the compiler came last.
- **GPU across three vendors**: CUDA (NVIDIA), HIP (AMD) and Metal (Apple).
- **CPU targets**: x86-64 from Intel and AMD, AWS Graviton, RISC-V.
- **Outside contributions to the compiler and tools are not being accepted
  yet** — Modular says by the end of 2026. The standard library has taken them
  since 2024.
- Modular was acquired by Qualcomm, which is the occasion for the release.

That is a serious piece of engineering and the vendor coverage is genuinely
better than this project's Vulkan path.

### What it would not do

The ask was "sync all this RAM, VRAM, GPU, CPU for a model when it runs". **A
language does not do that.** Deciding which of 256 experts stay resident, what
gets re-read per token, and what the KV cache may have is *policy*, and it is
already the thing this project is — `chaos-plan`, `ResidentSet`, the residency
report. Mojo would compile that policy faster; it would not supply one.

What it could genuinely replace is **ggml's kernels**. Chaos borrows ggml for
arithmetic and owns everything else, so the arithmetic is exactly the swappable
part, and one portable kernel set across NVIDIA, AMD and Apple is worth more
than a Vulkan binding that is verified on one card.

### Why not now, in numbers already measured here

- **The bottleneck is the disk, not the kernels.** On V4-Flash a token is 1.56 s
  of expert read plus 0.84 s that never touches the disk, and the routed expert
  arithmetic is **under 5% of a token**. A kernel twice as fast moves a token by
  less than 2.5%.
- **The dense path is at 1.20–1.27x of llama.cpp when both are hand-tuned.** A
  kernel rewrite chases at most that gap, against a target that is also written
  in hand-tuned SIMD.
- **It would be the first external toolchain in the project.** Today a release
  needs Rust, MinGW and a prebuilt ggml. The no-dependencies rule is why a
  release binary starts on a machine with no runtime, and it is load-bearing.
- **A compiler bug would not be fixable by us until contributions open.** For a
  component that decides whether the arithmetic is correct — and this codebase's
  central hazard is that wrong arithmetic produces fluent nonsense, never a
  crash — that is the wrong kind of dependency to take on.

**The honest condition under which it becomes interesting**: a machine where the
model is resident, so the workload is compute-bound rather than disk-bound, and
a second GPU vendor to support. Both are already open items. Until then it buys
under 5% of a token and costs a toolchain.

---

## 2. `xtool` builds iOS apps, and this is a Rust project

`xtool.sh` was suggested for exporting to macOS from this Windows machine. Read
against its own README, it does not do that:

- It builds **iOS** apps with **SwiftPM**. Chaos is Rust and the target asked
  for is **macOS**, which is neither half.
- Its hosts are Linux, macOS and Windows **through WSL** — so even the platform
  claim needs WSL rather than Windows itself.
- It expects an **Apple Developer account**, because its purpose is replacing
  Xcode's signing and deployment for iOS.

So it is the wrong tool, not a tool used wrongly.

### What a macOS build actually requires

Cross-compiling Rust to `aarch64-apple-darwin` needs the **Apple SDK**, and
Apple licenses that for use on Apple hardware. There is no legitimate path from
this Windows machine. The real route is a **macOS runner** — `macos-latest` in
the existing release workflow, which already builds a matrix.

And it would ship **less than the Windows release does**: `chaos-app` is Win32
against libraries only Windows has, so a macOS artefact is the CLI —
`chaos-run`, `chaos-serve`, `chaos-probe`, `chaos-pull`, `gguf-info`. Worth
saying up front rather than shipping something called a macOS build that has no
window in it.

### Linux packaging, which is buildable here

`.deb` is an `ar` archive holding two tarballs and a control file — every part of
it is in the Python standard library, so it needs no packaging tool and breaks no
rule. **AppImage is different**: it needs the AppImage runtime binary appended to
a squashfs image, and that runtime is a download. A `.tar.gz` of the same tree
gives the same "one file, no install" property with nothing fetched.

Neither can be *tested* on Windows, which is the reason to build them in CI on
`ubuntu-latest` where `dpkg-deb` exists, rather than shipping an artefact whose
only evidence is that a script ran.

---

**Sources**: Phoronix, "Modular's Mojo Language Now Open-Source Following
Qualcomm Acquisition"; `modular.com/blog/mojo-open-source`;
`docs.modular.com/mojo/requirements`; `docs.modular.com/mojo/manual/gpu/fundamentals`;
`github.com/xtool-org/xtool` README.
