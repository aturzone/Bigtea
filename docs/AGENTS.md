# Pointing a coding agent at Chaos

Chaos serves an **OpenAI-compatible** endpoint, so anything that takes a base
URL works. Load a model, then use **Model → Test the connection** — it makes the
same three requests an agent makes and writes the result into the transcript,
including the base URL and key to paste.

```
[ ok ]  server is up
    127.0.0.1:8231 answered /health

[ ok ]  model list
    /v1/models offers Llama-3.2-1B-Instruct

[ ok ]  a real completion
    the model replied "OK"
```

**Two values are all any client needs:**

| | |
|---|---|
| Base URL | `http://127.0.0.1:8231/v1` |
| API key | none by default — send any string. **Model → Require an API key** sets a real one |

The port comes from SETTINGS, and the endpoint line always shows the port the
server was actually given.

## What Chaos answers

Verified against the running server, not inferred from the code:

| route | what an agent uses it for |
|---|---|
| `GET /v1/models` | discovering the model's name — returns `{"object":"list","data":[{"id":…}]}` |
| `POST /v1/chat/completions` | the request that matters, with `usage` counts |
| `POST /v1/chat/completions` with `"stream": true` | server-sent events, `chat.completion.chunk` deltas |
| `POST /v1/completions` | the older text API |
| `POST /v1/embeddings` | vectors |
| `GET /health` | readiness — **never gated by the API key**, so a launcher can poll it |

One request at a time, no TLS, `127.0.0.1` only.

## Hermes

Hermes takes any OpenAI-compatible endpoint through its `custom` provider. In
`cli-config.yaml`:

```yaml
model:
  provider: "custom"
  base_url: "http://127.0.0.1:8231/v1"
  default: "Llama-3.2-1B-Instruct"   # whatever /v1/models reports
```

If you have set a key, put it where Hermes reads one for a custom provider (an
`OPENAI_API_KEY`-style variable in its environment); with no key set, any value
is accepted, so a placeholder is fine.

**Use the name `/v1/models` reports** as `default`. Chaos names the model after
its container, and a client that sends a name Chaos does not know is answered
anyway — but the two disagreeing is confusing later.

## aider

```bash
export OPENAI_API_BASE=http://127.0.0.1:8231/v1
export OPENAI_API_KEY=chaos          # any value, unless you set a real one
aider --model openai/Llama-3.2-1B-Instruct
```

## Cline, Continue, and most VS Code extensions

Choose the **OpenAI Compatible** provider and give it:

- Base URL `http://127.0.0.1:8231/v1`
- API key: anything, or the key from **Model → Require an API key**
- Model: whatever `/v1/models` reports

## When it does not connect

Run **Model → Test the connection** first. It fails on the first thing that is
wrong, which is nearly always one of:

- **`nothing answered on 127.0.0.1:8231`** — no model is loaded, or the port in
  SETTINGS is not the port the running server was started with. The port only
  takes effect on the next load.
- **`refused: the API key is wrong`** — a key is required and the client is
  sending a different one, or none.
- **`the model returned nothing`** — the model loaded but produced no text. An
  architecture Chaos has not verified can do this; the model's page says whether
  it is one.

Anything that reaches a real completion in the test will work from an agent,
because the test *is* what an agent does.

## A note on the key

There is no key by default. The server binds `127.0.0.1` and never listens on
the network, so a key is not what keeps a stranger out — what keeps them out is
that there is no route in. The key exists because many clients insist on sending
one, and because a shared machine is a real thing. When it is on, `/v1/*`
requires `Authorization: Bearer <key>` and answers `401` in the shape an OpenAI
client expects.
