# Chaos, the Windows app

`chaos-app` is a native window: pick a model, run it, talk to it, point a coding
agent at it. Everything it does is also possible from the command line — the app
is the shorter route, not a different engine.

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
+---------------------------+--------------------------------------+
|  logo                     |                                      |
|  CHAOS                    |  the conversation                    |
|  [INSTALLED] [AVAILABLE]  |                                      |
|                           |                                      |
|  model list               |                                      |
|                           +--------------------------------------+
|  [LOAD]    [UNLOAD]       |  what you type              [SEND]   |
|  [DOWNLOAD][DELETE]       +--------------------------------------+
|  [RESCAN]                 |  running <model> -> http://...       |
|  cache  threads  port     |  memory  6.4 GB free of 16.9 GB      |
+---------------------------+--------------------------------------+
```

**INSTALLED** lists what is on this machine. **AVAILABLE** lists what Chaos can
fetch, with two numbers per row:

```
v4flash UD-Q4_K_XL   155 GB [5 files]   needs 7.92 GB - streams
qwen3-32b Q4_K_M     19.8 GB            needs 19.8 GB - too big
```

**Read the second number.** The first is the download; the second is what has to
stay in memory. A 155 GB Mixture-of-Experts model *streams* on a 16 GB machine
because only the always-read weights are resident. A 20 GB dense model does not,
because a dense container has no routed experts to leave on disk. Sorting by
download size gets this exactly backwards.

## Using it

**LOAD** starts the engine on the selected model. Large models take a while; the
status line says `loading` until the server answers, then `ready`.

**The endpoint appears at the bottom** once a model is up:

```
running qwen3-4b -> http://127.0.0.1:8231/v1   (no API key needed, localhost only)
```

That is an OpenAI-compatible endpoint. Point `aider`, `Cline`, `Continue` or
anything else that takes a base URL at it. **There is no API key** — the server
binds `127.0.0.1` only and never listens on the network, so there is nothing to
authenticate. If a client demands a key, give it any string.

**UNLOAD** stops the engine and frees the memory. So does closing the window:
the model runs as a child process and Chaos stops it on the way out.

**DOWNLOAD** fetches the selected AVAILABLE model with `chaos-pull`, in the
background. **DELETE** removes an installed model and *every shard* of it, after
telling you how many files and how many bytes. It refuses while that model is
loaded.

## Settings

The three boxes are the settings that matter most, and they persist to
`%USERPROFILE%\.chaos\settings.txt`:

| | |
|---|---|
| `cache GiB` | expert cache budget. Empty means the engine measures your machine |
| `threads` | generation threads. Empty means measured — generation wants 2-4, not all of them |
| `port` | where the server listens, and what the endpoint line shows |

The file holds more than the window exposes — `threads_batch`, `context`, `ngl`,
`models_dir`, `auto`, `force` — and it is plain text, safe to edit by hand.
Unknown keys are preserved, so an older build will not silently discard a newer
one's preferences.

## Where things live

| | |
|---|---|
| the app and binaries | `%LOCALAPPDATA%\Chaos\bin` |
| models | `%USERPROFILE%\.chaos\models` |
| settings | `%USERPROFILE%\.chaos\settings.txt` |
| a crash report | `%TEMP%\chaos-app-crash.log` |

**Models are never inside the install.** Uninstalling cannot delete them, and an
upgrade never touches them.

## When something goes wrong

If the app closes unexpectedly it writes `%TEMP%\chaos-app-crash.log` and shows
a message box naming it. That file says what failed and where — please send it.

A model that will not load is nearly always one of three things: the always-read
set does not fit (the AVAILABLE row says `too big`), the port is already taken
(change it), or the architecture has never been diffed against llama.cpp. The
app passes `--force` for the last case, because refusing to run what you have is
not useful in a window — but be aware that an unverified architecture can produce
fluent nonsense rather than an error.

## What the app does not do yet

Named plainly rather than left to be discovered:

- No per-model window; one model runs at a time.
- Download progress shows start and finish, not a percentage.
- No tray icon — closing the window is the way to quit.
- The GPU settings (`ngl`, device) are in the settings file but not in the
  window.

`docs/graph/backlog/app-to-production.md` tracks these.
