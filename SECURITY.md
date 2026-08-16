# Security Policy

## Supported versions

Chaos is at **v0.0.0**, a preview. Only the latest release and `main` receive
fixes. There is no long-term support branch yet, despite the LTS ambition in the
roadmap.

| version | supported |
|---|---|
| `main` | ✅ |
| 0.0.0 | ✅ |
| anything earlier | ❌ (none exists) |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private reporting:
[Security → Report a vulnerability](https://github.com/aturzone/Chaos/security/advisories/new).

If that is unavailable to you, contact the maintainer through their GitHub
profile ([@aturzone](https://github.com/aturzone)) and ask for a private channel
before sending details.

Expect an acknowledgement within a week. This is a solo project, so please be
patient — and please do give it a reasonable disclosure window before publishing.

## What is in scope

Chaos parses **untrusted binary input**: a GGUF container is a file that may
have come from anywhere on the internet, and its header declares offsets, sizes,
dimensions and types that the parser must not trust. The interesting bug classes
are therefore:

- Memory-safety issues reachable by parsing a malformed or hostile `.gguf`
  (out-of-bounds reads, integer overflow in offset or size arithmetic,
  allocation of an attacker-chosen size).
- Path traversal or unintended file access through shard discovery, which
  derives sibling file names from the path you pass it.
- Any way to make Chaos write outside paths the user explicitly named. Chaos
  opens model files **read-only** and should never write to them.

`crates/chaos-ggml` contains `unsafe` FFI by necessity, and
`crates/chaos-io`'s aligned buffers use raw allocation. Both are the places to
look.

## What is out of scope

- **Model behaviour.** What a model generates is not a Chaos vulnerability.
  Chaos runs weights; it does not endorse their output.
- **Resource exhaustion from a model you chose to run.** Asking Chaos to load a
  144 GB model and running out of memory is the documented behaviour, not a
  denial of service. Chaos reports what will not fit rather than failing
  silently, and that is the intended handling.
- **ggml itself.** Report those upstream to
  [ggml-org/ggml](https://github.com/ggml-org/ggml). If a ggml issue is reachable
  specifically through how *Chaos* calls it, that is in scope here.
- Anything requiring an attacker to already have local code execution as the
  user running Chaos.

## Known and deliberate

Chaos binds weights **zero-copy**: `ggml` tensors point directly into buffers
Chaos owns, and the safety of that arrangement rests on `WeightSet` outliving
every tensor that points into it. This is enforced by the borrow checker and
documented in `crates/chaos-ggml/src/weights.rs`. If you find a way to defeat
it from safe code, that is a genuine finding and I want to hear about it.
