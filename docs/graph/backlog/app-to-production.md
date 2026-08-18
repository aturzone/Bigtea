---
topic: everything wrong with chaos-app and chaos-setup, and the order to fix it
status: in progress, opened 2026-08-18
links:
  - ../reference/hard-won-facts.md
  - bigger-machine-prompt.md
---

# Getting the app to production

**Atur's verdict on v0.0.5 was "still acts like trash", and he is right.** The
installer's file mechanics were verified and the application itself was never
used. Opening a window was treated as evidence that it worked. That is the same
mistake this codebase has a standing rule about — *loading is not evidence* —
committed in a new place.

This is the complete list, nothing dropped. Items are ordered so that each one
is testable when it lands.

## The root cause of "it crashes when I click anything"

`WM_CTLCOLORLISTBOX`, `WM_CTLCOLORBTN` and `WM_CTLCOLOREDIT` all do
`UI.borrow()`. `rescan()` holds `UI.borrow_mut()` and, while holding it, calls
`SendMessageW(list, LB_RESETCONTENT | LB_ADDSTRING | LB_SETCURSEL)` and
`EnableWindow`.

**Those messages are dispatched synchronously.** The listbox asks its parent for
colours *during* `LB_ADDSTRING`, the parent's handler calls `borrow()`, the
`borrow_mut()` is still live, `RefCell` panics — and `panic = "abort"` in the
release profile turns that into instant process death with no window, no message
and no log.

So clicking INSTALLED or AVAILABLE could never have worked. The rule this
breaks, and the one to hold to from now on:

> **Never hold a `RefCell` borrow across a Win32 call that can dispatch a
> message.** Copy what is needed out of the borrow, drop it, then call.

## A — blockers, nothing works until these land

- [x] **A1. No borrow may span a Win32 call.** Restructure every `UI.with` site
      to take a short borrow, copy out handles and values, drop, then call.
      ~20 sites.
- [x] **A2. A panic must be visible.** `panic = "abort"` means every future
      mistake is a silent disappearance. Install a panic hook that writes a log
      beside the executable and shows a message box naming the file, so the next
      failure is reportable instead of invisible.
- [x] **A3. GUI uninstall does nothing.** `uninstall_to` spawns the detached
      helper and returns, but the window stays open, so the helper cannot delete
      the directory the window is running from and gives up after ten seconds.
      The GUI must exit after spawning.
- [ ] **A4. Every button verified by clicking it**, not by reading the code.

## B — the installer

- [ ] **B1. SmartScreen.** "Windows protected your PC / Unknown publisher" on
      every download. Options, honestly costed: an OV/EV certificate (money,
      annual, and OV still needs reputation to accumulate); publishing hashes
      and a documented "More info -> Run anyway" path (free, still a warning);
      submitting the binary to Microsoft for analysis (free, reduces it over
      time). **Reputation is per-signature and per-file, so an unsigned
      installer that changes every release will always warn.** Decide with Atur;
      do not pretend a code change removes this.
- [x] **B2. Icons.** `chaos-setup.exe`, `chaos-app.exe` and every other binary
      ship with the default blank Windows icon, and the window has none. Needs a
      real `.ico` generated from `assets/logo.svg` at several sizes, embedded via
      a resource, plus `WM_SETICON` for the title bar and taskbar.
- [x] **B3. Upgrade in place.** Running a newer setup over an older install must
      detect the existing version, stop anything running from it, replace the
      files, and say what it upgraded from and to.
- [x] **B4. A completion report.** Install and uninstall currently end by the
      window simply closing. It must stay open and say what happened: what was
      written, where, what to do next — and for uninstall, what was removed and
      what was deliberately left.
- [ ] **B5. Uninstall from Add/Remove Programs works end to end**, verified by
      going through the Windows UI rather than a flag.

## C — the application

- [x] **C1. Activate / deactivate a model**, with the state visible in the list.
- [x] **C2. Download and delete models**, with real progress, cancel, and disk
      space checked before starting.
- [x] **C3. Resource usage, live**: RAM resident, what is streaming, disk read
      rate, VRAM if a card is used — the numbers the engine already prints, in
      the window rather than in a log.
- [x] **C4. Background running and a real quit.** Closing the window must not
      leave `chaos-serve` running, and quitting from the taskbar must stop
      everything. A tray icon with an explicit Quit.
- [x] **C5. API key and endpoint URL per running model**, shown and copyable,
      so a coding agent can be pointed at it.
- [ ] **C6. A management view per running model** — its own panel: status,
      throughput, context, stop.
- [ ] **C7. A settings page**: models directory, default cache, threads,
      generation and prefill, context, port, device, and where they persist.
- [x] **C8. Responsive layout.** The window must reflow rather than clip, and
      look deliberate at any size.
- [ ] **C9. Every option a model runner needs**, reachable without the CLI.

## D — what "LTS" has to mean here

- [ ] **D1. Tests for UI logic**, since Win32 painting cannot be tested: state
      transitions, list contents, enable/disable rules, settings round-trip.
- [ ] **D2. No silent failure anywhere.** Every path that can fail reports.
- [ ] **D3. A written manual** for the app, not only the CLI.
- [ ] **D4. The release exercises the app**, not just the binaries' `--help`.

## The rule that would have caught all of this

A GUI cannot be verified by building it. Every item above closes only when it
has been **driven** — clicked, in a window, on a clean machine — and the
evidence is in the pull request.

## Where it stands, 2026-08-18

**Done and verified by driving the window**: A1-A3, B2-B4, C1-C5, C8.

`chaos-app` no longer aborts on a click, carries the logo everywhere, shows
the endpoint and live memory, deletes models with every shard, and stops the
engine when the window closes. The installer has an icon, detects and names an
upgrade, and reports what it did instead of vanishing.

**Two more instances of the crash class were found by audit, not by clicking**:
`load_model` held a borrow across `SetWindowTextW`. LOAD would have died the
same way the tabs did. That is the argument for D1: this class is invisible
until the exact control is touched, so it needs a rule enforced rather than a
person remembering.

**Still open**: B1 (SmartScreen -- needs a decision, not a commit), B5, C6, C7,
C9 and all of D.
