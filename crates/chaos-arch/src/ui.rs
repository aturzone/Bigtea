//! The browser interface `chaos-serve` hands out at `GET /`.
//!
//! **Why a page and not a window.** This workspace has zero external
//! dependencies on purpose, and every native GUI toolkit is a large one plus a
//! platform-specific build. A served page needs neither: the browser is already
//! installed, it renders the same on all three platforms, and the transport is
//! the `/v1/chat/completions` SSE endpoint that agents already use and that the
//! tests already cover. So the UI exercises the same path a real client does
//! rather than a second one written for it.
//!
//! Everything is inline — no CDN, no font, no image, no build step. The page is
//! one `&str` compiled into the binary, so an offline machine with no network at
//! all still gets the whole interface. That is the same constraint the rest of
//! the project runs under: Chaos downloads nothing on its own.

/// The single page. Served verbatim; `{model}` is substituted at request time.
pub const PAGE: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chaos</title>
<style>
  :root {
    --bg: #ffffff; --fg: #17171a; --dim: #6b6b75; --line: #e3e3e8;
    --panel: #f7f7f9; --accent: #d1500f; --mine: #eef1f6; --code: #f2f2f5;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #131316; --fg: #e9e9ec; --dim: #9a9aa4; --line: #2b2b31;
      --panel: #1b1b1f; --accent: #ff7a33; --mine: #232329; --code: #202026;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg); color: var(--fg);
    font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column;
  }
  header {
    border-bottom: 1px solid var(--line); padding: 10px 16px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  .mark { font-weight: 700; letter-spacing: .18em; font-size: 13px; }
  .model { color: var(--dim); font-size: 13px; font-family: ui-monospace, monospace; }
  .spacer { flex: 1; }
  .stat { color: var(--dim); font-size: 12px; font-family: ui-monospace, monospace; }
  main { flex: 1; overflow-y: auto; padding: 20px 16px; }
  .wrap { max-width: 760px; margin: 0 auto; }
  .msg { margin-bottom: 18px; }
  .who {
    font-size: 11px; text-transform: uppercase; letter-spacing: .1em;
    color: var(--dim); margin-bottom: 5px;
  }
  .body { white-space: pre-wrap; word-wrap: break-word; }
  .msg.me .body {
    background: var(--mine); padding: 10px 13px; border-radius: 10px;
  }
  .msg.err .body { color: var(--accent); }
  .cursor::after {
    content: "\258b"; color: var(--accent);
    animation: blink 1.1s steps(2, start) infinite;
  }
  @keyframes blink { to { visibility: hidden; } }
  footer { border-top: 1px solid var(--line); padding: 12px 16px; }
  form { max-width: 760px; margin: 0 auto; display: flex; gap: 8px; align-items: flex-end; }
  textarea {
    flex: 1; resize: none; font: inherit; color: var(--fg);
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 9px; padding: 10px 12px; min-height: 44px; max-height: 190px;
  }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button {
    font: inherit; font-weight: 600; cursor: pointer; color: var(--bg);
    background: var(--fg); border: 0; border-radius: 9px; padding: 11px 18px;
  }
  button:disabled { opacity: .45; cursor: default; }
  .hint { max-width: 760px; margin: 8px auto 0; color: var(--dim); font-size: 12px; }
  .empty { color: var(--dim); text-align: center; margin-top: 12vh; }
  .empty h2 { font-weight: 600; font-size: 17px; color: var(--fg); margin: 0 0 6px; }
</style>
</head>
<body>
<header>
  <span class="mark">C H A O S</span>
  <span class="model" id="model">&mdash;</span>
  <span class="spacer"></span>
  <span class="stat" id="stat"></span>
</header>

<main id="main">
  <div class="wrap" id="log">
    <div class="empty" id="empty">
      <h2>Ask it something.</h2>
      <div>Running locally. Nothing leaves this machine.</div>
    </div>
  </div>
</main>

<footer>
  <form id="form">
    <textarea id="input" rows="1" placeholder="Send a message&hellip;" autofocus></textarea>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="hint">Enter sends &middot; Shift+Enter for a new line</div>
</footer>

<script>
const log = document.getElementById('log');
const input = document.getElementById('input');
const form = document.getElementById('form');
const send = document.getElementById('send');
const stat = document.getElementById('stat');
const main = document.getElementById('main');
const empty = document.getElementById('empty');

// The whole conversation, resent each turn. This server holds no session
// state -- an engine that streams experts per token has nothing cheap to keep
// between requests -- so the transcript lives here and travels with the call.
let history = [];
let busy = false;

fetch('/v1/models').then(r => r.json()).then(d => {
  const id = d && d.data && d.data[0] && d.data[0].id;
  if (id) document.getElementById('model').textContent = id;
}).catch(() => {});

function add(role, text) {
  empty.style.display = 'none';
  const el = document.createElement('div');
  el.className = 'msg' + (role === 'you' ? ' me' : role === 'error' ? ' err' : '');
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = role;
  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = text;
  el.appendChild(who);
  el.appendChild(body);
  log.appendChild(el);
  main.scrollTop = main.scrollHeight;
  return body;
}

function autosize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 190) + 'px';
}
input.addEventListener('input', autosize);

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});

form.addEventListener('submit', async e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || busy) return;

  busy = true; send.disabled = true;
  input.value = ''; autosize();
  add('you', text);
  history.push({ role: 'user', content: text });

  const out = add('chaos', '');
  out.classList.add('cursor');
  const t0 = performance.now();
  let produced = 0, answer = '';

  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history, stream: true, max_tokens: 512 })
    });
    if (!res.ok) throw new Error('server returned ' + res.status);

    // Server-sent events, parsed by hand: a chunk boundary can fall anywhere,
    // including mid-event, so hold the remainder until a blank line completes.
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        for (const line of part.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;
          let piece = '';
          try {
            const j = JSON.parse(payload);
            piece = (j.choices && j.choices[0] && j.choices[0].delta &&
                     j.choices[0].delta.content) || '';
          } catch (_) { continue; }
          if (!piece) continue;
          answer += piece; produced++;
          out.textContent = answer;
          const s = (performance.now() - t0) / 1000;
          // ASCII only, deliberately: the page is served with a Content-Length
          // counted in bytes, so a multi-byte separator here makes the header
          // disagree with the body. A test pins this.
          stat.textContent = produced + ' tokens, ' + (produced / s).toFixed(2) + ' tok/s';
          main.scrollTop = main.scrollHeight;
        }
      }
    }
    if (answer) history.push({ role: 'assistant', content: answer });
    else out.textContent = '(no output)';
  } catch (err) {
    out.parentElement.className = 'msg err';
    out.textContent = String(err && err.message ? err.message : err) +
      '\n\nIs chaos-serve still running? Its terminal window shows every request.';
    history.pop();
  } finally {
    out.classList.remove('cursor');
    busy = false; send.disabled = false; input.focus();
  }
});
</script>
</body>
</html>
"##;
