# Chaos, the Windows app

`chaos-app` is a native window: pick a model, run it, talk to it, point a coding
agent at it, and watch what the machine is doing while it works. Everything it
does is also possible from the command line — the app is the shorter route, not
a different engine.

## Installing

Download **`Chaos-Setup.exe`** from
[Releases](https://github.com/aturzone/Chaos/releases) and run it. One file,
everything inside it, no administrator rights. It installs to
`%LOCALAPPDATA%\Chaos`, adds that to your PATH, creates the models folder and
puts Chaos in the Start Menu.

Running a newer setup over an older install upgrades in place and tells you what
it replaced. Uninstall from **Settings → Apps**, or by running the setup again
and pressing UNINSTALL.

> **Windows will warn you** that it "protected your PC" and the publisher is
> unknown. That is what Windows says about every unsigned application, and it
> will keep saying it until the binary is signed with a certificate. Choose
> **More info → Run anyway** if you trust the download. There is no code change
> that removes this.

## The window

```
+----------------+----------------------------------------------+
| ✳ CHAOS        |  Chat                                        |
|   v0.0.12      |  Talk to the running model, or point a       |
|                |  coding agent at its endpoint.               |
| ▎CHAT          |                                              |
|  MODELS        |  +----------------------------------------+  |
|  MONITOR       |  |  the conversation                      |  |
|  SETTINGS      |  +----------------------------------------+  |
|                |  [ what you type      ]  [ SEND ]  CLEAR     |
+----------------+----------------------------------------------+
| ● qwen3-4b                        15.4 tok/s        [ STOP ]   |
|   http://127.0.0.1:8231/v1        up 3m · 812 tokens           |
+----------------------------------------------------------------+
```

**Four pages, and one owns the screen at a time.** Reach them from the rail, from
**View** in the menu, or with `Ctrl+1` … `Ctrl+4`.

**The strip along the bottom is on every page.** Whatever you are looking at, it
says whether a model is up, where to reach it, and how fast it is going.

### CHAT

The conversation, full width. `Ctrl+Enter` sends — plain Enter makes a
paragraph, because a prompt is often more than one line. **CLEAR** empties the
transcript and the history behind it; until then the model sees the whole
conversation.

### MODELS

**INSTALLED** lists what is on this machine. **AVAILABLE** lists what Chaos can
fetch. Selecting a model opens **that model's own page** beside the list:

```
Llama-3.2-1B-Instruct-Q4_K_M                            ● RUNNING

  [ LOAD ]   [ STOP ]   [ COPY ENDPOINT ]
  [ DOWNLOAD ]   [ DELETE ]

  on disk        808 MB
  endpoint       http://127.0.0.1:8231/v1
  context        the model's own limit
  threads        measured
  expert cache   measured
  uptime         1m 18s
  served         20 tokens
```

For a model you have not downloaded yet, the same page shows the two numbers
that decide whether it will run here:

```
  download            155 GB
  stays resident      7.92 GB -- this is the number that decides
```

**Read the second one.** The first is the download; the second is what has to
stay in memory. A 155 GB Mixture-of-Experts model *streams* on a 16 GB machine
because only the always-read weights are resident. A 20 GB dense model does not,
because a dense container has no routed experts to leave on disk. Sorting by
download size gets this exactly backwards.

**LOAD** starts the engine. Large models take a while; the strip says so until
the server answers. **STOP** frees the memory, and so does closing the window —
the model runs as a child process and Chaos stops it on the way out.

**COPY ENDPOINT** puts `http://127.0.0.1:8231/v1` on the clipboard, which is the
string you paste into `aider`, `Cline`, `Continue`, or anything else that takes
a base URL. With a key set it copies both.

### The API key

**Model → Require an API key** turns one on. Chaos generates 24 bytes from the
system's own random generator, shows it, copies it, and stores it; from then on
`chaos-serve` refuses any `/v1/*` request without
`Authorization: Bearer <key>` and answers `401` in the shape an OpenAI client
expects.

It is **off by default**, deliberately. The server binds `127.0.0.1` only and
never listens on the network, so a key is not what keeps a stranger out -- what
keeps them out is that there is no route in. Switching it on by default would
also start refusing every agent already pointed at an existing install. It is
here because many clients insist on sending a key, and because a shared machine
is a real thing.

`/health` and the browser page stay open either way, so the window can still
tell when a model is up. With no key set, any value a client sends is accepted.
The key is on the model's page and on MONITOR while a model runs.

**DELETE** removes an installed model and *every shard* of it, after telling you
how many files and how many bytes. It refuses while that model is running.

### An unfinished download

A row marked `(unfinished)` is a container that stopped part way. Chaos knows
because a GGUF states its own length: the tensor index says where the last
tensor ends, and a file shorter than that is truncated — no catalogue and no
network involved. **LOAD refuses it and says how much is missing**; **DOWNLOAD**
finishes it, resuming from what is already on disk.

This matters because the failure is otherwise invisible. A half-written `.gguf`
has a perfectly valid header — the header is written first — so it sits in the
list looking exactly like a model that works.

### MONITOR

What the machine is doing: memory free and in use with a bar, the running
model's endpoint, uptime, last measured rate and tokens served, and what is on
disk.

Streamed bytes, expert read rate and cache residency are measured inside the
engine and printed to its log. **They are not on this page**, because nothing
carries them over the socket yet — and the page says so rather than leaving a
gap to be noticed.

### SETTINGS

Every setting the file holds — nine of them — grouped, each saying what it does
and what leaving it empty will do:

| | |
|---|---|
| **Model defaults** | `context`, `GPU layers`, `measure this machine`, `allow unverified architectures` |
| **Performance** | `expert cache`, `generation threads`, `prefill threads` |
| **Server** | `port` |
| **Paths** | `models folder` |

**`models folder` takes more than one.** Separate them with `;` and all of them
are searched, which is how a 144 GB container on another drive appears beside a
2 GB one. The *first* is where downloads go. The setting is read by the engine
too, so `chaos-run` and `chaos-serve` see the same models the window does.

**Empty means measured.** Chaos reads the machine and picks a value; typing one
overrides it. **SAVE** writes the file and reports where. **RESET** returns every
engine setting to measured, and leaves the theme alone.

> **`allow unverified architectures`** runs a model that has never been diffed
> against llama.cpp. Such a model can produce fluent nonsense rather than an
> error. The setting is honest about that, and it is honoured — turning it off
> makes `chaos-serve` refuse by name.

### The menu

**File** — rescan (`F5`), open the models folder, exit.
**Model** — load (`Ctrl+L`), stop, download, delete, copy endpoint (`Ctrl+E`).
**View** — the four pages, and light or dark.
**Help** — manual, check for updates, install update, releases, crash log, about.

Commands that cannot be run right now are greyed rather than left to be tried.

## Updating

**Chaos tells you when there is a newer release.** It asks GitHub once, shortly
after the window opens, and says nothing at all unless something newer exists —
if one does, it offers to fetch the installer and hand over to it.

| | |
|---|---|
| `Help ▸ Check for updates` | asks now, and always answers, even to say you are current |
| `Help ▸ Install update…` | downloads this platform's installer and starts it |
| `chaos-run --update` | the same, from a terminal |

**Chaos closes when the installer starts.** Windows keeps a running
executable's file open, so the installer cannot replace `chaos-app.exe` while
the window is up — it would stop with *"cannot write chaos-app.exe. Close Chaos
and run this again."* Letting it hand over cleanly is the whole reason for the
step.

**One update updates everything.** The installer carries all twelve binaries —
the window, `chaos-run`, `chaos-serve`, `chaos-pull` and the rest — so there is
nothing to update per binary and no version skew to manage.

**Your models are never touched.** They live outside the install prefix
(`%USERPROFILE%\.chaos\models` by default) and neither the installer nor the
uninstaller goes near them. Your settings survive too.

> **To turn the automatic check off**, set `CHAOS_NO_UPDATE_CHECK=1` in the
> environment. The menu items still work; nothing is asked on startup.

If the download fails, Chaos says so and gives you the URL — no update is ever
applied from a file it could not verify the size of, because `curl` reports
success after saving an error page just as happily as after saving an installer.

## Light and dark

Chaos opens light, which is what Hermes' desktop does and what this design was
drawn against. **View → Dark** switches, and the choice persists.

> **The menu bar stays light in dark mode.** It is drawn by Windows outside the
> client area and does not follow the dark title-bar attribute.
> `SetPreferredAppMode` — the undocumented `uxtheme` ordinal every dark Win32
> app calls — was tried both before and after window creation, with
> `FlushMenuThemes`. The ordinals resolve on Windows 10.0.26200 and the bar
> still measures `#FFFFFF`. Owner-drawing the entire menu is the only route
> left, and it needs another undocumented message to paint the bar's own
> background. The scrollbars *are* fixed, by naming the control's dark theme
> class — measured going from `#F0F0F0` to `#171717`.

## Where things live

| | |
|---|---|
| the app and binaries | `%LOCALAPPDATA%\Chaos\bin` |
| models | `%USERPROFILE%\.chaos\models`, plus anything in `models folder` |
| settings | `%USERPROFILE%\.chaos\settings.txt` |
| a crash report | `%TEMP%\chaos-app-crash.log` |

**Models are never inside the install.** Uninstalling cannot delete them, and an
upgrade never touches them.

**A model in its own folder is found.** Each search folder is scanned, and so is
each of its immediate subfolders — one level, not a walk, because a models
folder pointed at a whole drive would otherwise read every directory on it. A
five-shard container in `models/v4flash/` appears as one entry, its first shard.

The settings file is plain text and safe to edit by hand. Keys it does not
recognise are preserved, so running an older build once will not silently
discard a newer one's preferences.

## When something goes wrong

If the app closes unexpectedly it writes `%TEMP%\chaos-app-crash.log` and shows
a message box naming it. **Help → Open crash log** finds it. That file says what
failed and where — please send it.

A model that will not load is nearly always one of four things: the download did
not finish (the row says `(unfinished)`; press DOWNLOAD), the always-read set
does not fit (the model's page says `too big for this machine`), the port is
already taken (change it in SETTINGS), or the architecture has never been diffed
against llama.cpp and `allow unverified architectures` is off.

## What the app does not do yet

Named plainly rather than left to be discovered:

- One model runs at a time.
- Download progress is measured from the bytes on disk, so a paused or
  restarted fetch is still tracked; it cannot show which *shard* of a
  five-part container is in flight.
- No tray icon — closing the window is the way to quit.
- MONITOR cannot show streamed bytes or cache residency; the engine measures
  them but does not report them over the socket.
- The menu bar does not follow dark mode. See above for what was tried.

`docs/graph/backlog/app-to-production.md` tracks these.
