# A mixed host/device context builds correctly and then segfaults

**2026-08-15.** Split out of
[phase-a-device-prefill-2026-08-15.md](phase-a-device-prefill-2026-08-15.md)
because it is the obvious thing to try and it costs a day.

Links: [phase-a-device-prefill-2026-08-15.md](phase-a-device-prefill-2026-08-15.md) ·
[gpu-tier-smallest-honest-slice-2026-08-11.md](gpu-tier-smallest-honest-slice-2026-08-11.md)

## The obvious thing

Phase C is dense weights resident on the card with routed experts streaming from
disk. So: bind some tensors host-side (zero-copy, as this engine always has) and
some device-side, put them in one context, and run the graph. No scheduler, no
mixed-device machinery — just let ggml sort it out.

Every step of that works until the last one.

| step | outcome |
|---|---|
| bind one host tensor, one device tensor | fine |
| build a graph over both | fine |
| allocate device memory for the context | **fine, and correct** |
| `ggml_backend_graph_compute` | **STATUS_ACCESS_VIOLATION** |

## What is actually correct here

`ggml_backend_alloc_ctx_tensors_from_buft` **skips a tensor that already carries
a host pointer.** That is not a workaround; it is the documented behaviour and it
is what makes a mixed context expressible at all. Measured: binding one 3x4 host
tensor and one 4x2 device tensor and allocating gives

```
report.tensors == 1
report.bytes   == 32     // exactly the device tensor, k*n*4
```

The host tensor is untouched. The split is exactly what was asked for, the
allocation is exactly right, and nothing warns.

## What is not

The Vulkan backend then dereferences the host pointer as device memory. The
process dies with `STATUS_ACCESS_VIOLATION` — no `ggml_status` error, no
refusal, no fallback to a slower path, and no Rust backtrace, because the fault
is inside the driver.

**So `ggml_backend_sched` is mandatory for Phase C, not an optimisation.** The
scheduler exists precisely to insert the copies this path assumes are unnecessary.

## Why this node exists separately

Three of this session's crashes were the same fault wearing different clothes,
and the third one is the reason to write it down rather than remember it:

1. The experiment above — a host pointer read as device memory.
2. `pos.set_i32` in the QKV builder — a tensor written before its context was
   realized, which on a device is a null pointer.
3. `mask.set_bytes` **inside** `attention_flash` — same fault, but unreachable
   from the call site, because the function both built graph nodes and wrote
   into them.

(3) generalises past this project:

> **A function that both builds graph nodes and writes into them cannot be used
> on a device.** Those two operations must be separated by an allocation the
> function does not control, so a builder has to *return* its input tensors
> rather than fill them.

`attention_flash` now returns the mask tensor unwritten. Any future builder must
do the same, and the failure mode if it does not is a segfault with no
diagnostic — not a compile error, not a wrong number, just a dead process.

## The test that stayed

`crates/chaos-ggml/tests/device_arithmetic.rs` asserts the half that must keep
working — that the split happens and a host-bound tensor is **never** uploaded,
since the zero-copy path is the whole memory design. It does **not** execute the
compute step: an access violation takes the entire test binary down and loses
every other result, which is the same reason the V4-Flash suite serialises its
heavy tests.

A finding that kills the process is written down, not re-run.
