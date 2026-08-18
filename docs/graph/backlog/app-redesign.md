---
topic: the app's second design — navigation, per-model pages, monitoring
status: proposed 2026-08-18, awaiting Atur
links:
  - app-to-production.md
  - ../../APP.md
---

# The redesign

Atur's verdict on the current window: *"too messy and not user friendly … why
is all click in one slot … where is settings, where is model management, where
is the menu, where are the windows, where are the options."*

He is right, and the reason is structural rather than cosmetic. **Everything is
on one screen because the window was grown one button at a time.** Every feature
added a control to the same sidebar, so a model list, a download catalogue, four
actions and three settings all compete for one 380px column, and none of them
has room to say anything.

## What Hermes does that this does not

Captured from `Hermes-Setup.exe` on this machine (installer only — nothing was
installed):

- **One idea per screen.** The first frame is a wordmark, one sentence, and one
  action. Nothing else is on it.
- **Enormous typographic range.** The wordmark is perhaps 60px against 15px body
  text. The current app runs everything at 14-15px, so nothing leads and the eye
  has nowhere to start.
- **A single primary action**, marked out — `[ INSTALL ]` — rather than six
  buttons of equal weight.
- **Space is the main material.** Wide margins, a full-bleed field, and the
  content sitting in the middle third.

The two-colour rule stays: that was a deliberate choice and it is not what makes
the window messy. Hierarchy comes from **size, weight and space**, which a
two-value palette supports perfectly well.

## The shape to build

**Navigation down the left, one page at a time on the right.** Four
destinations, not one panel with everything in it:

```
+--------------+-----------------------------------------------+
|  CHAOS       |                                               |
|              |   the page                                    |
|  ▸ CHAT      |                                               |
|    MODELS    |                                               |
|    MONITOR   |                                               |
|    SETTINGS  |                                               |
|              |                                               |
|  ------------|                                               |
|  ● qwen3-4b  |   <- what is running, always visible          |
|    16.9 tok/s|                                               |
+--------------+-----------------------------------------------+
```

### CHAT
The conversation, full width. Nothing else on the page. The running model and
its endpoint sit in the persistent strip, not in the middle of the transcript.

### MODELS
Two lists — installed and available — with **a page per model** rather than a
row. Selecting one opens it:

```
Llama-3.2-1B-Instruct-Q4_K_M                        ● RUNNING
808 MB on disk · 808 MB resident · llama · verified

  [ STOP ]   [ DELETE ]

  endpoint    http://127.0.0.1:8231/v1
  context     2048 tokens        threads    4
  cache       (measured)         port       8231
  started     4 minutes ago      served     103 tokens
```

That answers *"a model needs three buttons to use it"* and *"where is the status
of model active or not"* — the status is on the model's own page and echoed in
the persistent strip.

### MONITOR
What the machine is doing while a model runs, which currently does not exist
beyond one line: memory in use and free, bytes streamed from disk, read rate,
cache hit rate, tokens per second over time, and which process is holding what.
The engine already prints all of this; none of it reaches the window.

### SETTINGS
The settings file has nine fields and the window shows three. All of them
belong here, grouped — **model defaults**, **performance**, **paths**,
**server** — each with what it does and what "empty" means, and a *Reset to
measured* action.

## Rules for the rebuild

1. **One page owns the screen.** No page may borrow space from another.
2. **One primary action per page**, visually heavier than the rest.
3. **Type scale, not more controls**: a display size for the page title, a body
   size, and a small size for units and hints. Three sizes, no more.
4. **The persistent strip is the only thing on every page** — what is running,
   its throughput, and STOP.
5. **Every number carries its unit and its meaning.** `808 MB resident` rather
   than `808 MB`, because the resident figure is the one that decides whether a
   model runs.
6. **Nothing may be discovered by clicking.** If an action can fail or take
   minutes, the page says so before it is pressed.

## What this costs

This is a rewrite of `main.rs`'s window layer, not an edit. The current file
builds every control up front and positions them in one `layout` function; a
paged design needs controls created and destroyed per page, a navigation model,
and a persistent strip that survives page changes. The logic underneath —
`settings`, `catalog`, `models`, `client` — is already separate and tested, and
none of it needs to change.

**Order**, so each step is usable on its own:

1. The navigation shell and the persistent strip, with CHAT as the only page.
   Nothing is lost, everything still works.
2. MODELS with per-model pages, replacing the current sidebar list.
3. SETTINGS, exposing what the file already holds.
4. MONITOR, which needs the engine to report over its socket rather than to a
   log.

Step 1 is the one that changes how the app feels; steps 2-4 are then additive.
