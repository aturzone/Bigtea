---
topic: R10.1 — GBNF and JSON-schema constrained decoding, and the two bugs only llama.cpp's own grammars could find
status: resolved
links: [../backlog/llamacpp-flag-audit.md, ../backlog/lts-parity-criteria.md]
---

`crates/chaos-grammar` parses GBNF, compiles it to a stack matcher, and turns
the bytes generated so far into the set of token ids that may legally come next.
Four of llama.cpp's 182 flags depend on it: `--grammar`, `--grammar-file`,
`--json-schema`, `--json-schema-file`.

It was previously marked **won't for LTS**. Reopened because an agent calling a
local model wants its answer to *parse* more than it wants most of the rest of
the list: sampling freely and hoping is how you get a trailing comma, and a
retry costs a whole generation.

## The engine hook is one function

```rust
let grammar = Grammar::from_json_schema(schema)?;
let constraint = Constraint::new(grammar, &vocab);   // vocab: token id -> bytes
constraint.allowed(generated_so_far).apply(&mut logits);
```

`allowed` returns a bitset; `apply` sets every disallowed logit to `-inf`. The
crate has **no dependencies at all** — not on ggml, not even on the tokenizer,
because the vocabulary arrives as bytes the caller already has. It is in the CI
job that proves the ggml-free crates build on a machine with no C toolchain.

Deliberately **not** wired into `sample.rs` or `chaos-run.rs`: another session
owns those files.

## What C3 gets wrong if you write the tests yourself

**A grammar that accepts everything passes every test that only checks
acceptance.** `root ::= .*` constrains nothing, and neither does a parser that
quietly turned a rule it did not understand into an empty one. So every case
here checks a rejection that is a one-character edit of an acceptance, and the
accepted text is **llama.cpp's own output** rather than a string invented to
match the implementation.

```
llama-completion -m Llama-3.2-1B-Instruct-Q4_K_M.gguf --temp 0 -no-cnv \
  -n 60 -c 512 --no-warmup --grammar-file grammars/json.gbnf \
  -p "JSON object describing a person:"

{"name":"John","age":30,"city":"New York","country":"USA"," occupation":
"Software Engineer","address":{"street":"123 Main St","city":"New York",
"state":"NY","zip":"10001","country":"USA"}}    [end of text]
```

That is accepted and **complete**. The near-misses — trailing comma, unquoted
key, single quotes, leading zero, invalid escape, a raw newline inside a string,
an array where the root demands an object — are all rejected.

Same for the schema path, against llama.cpp's `--json-schema`:

| schema | llama.cpp's output |
|---|---|
| `{name: string, age: integer}` | `{"name":"John","age":30}` |
| `{city: string, scores: integer[] minItems 2}` | `{"city": "New York", "scores": [1, 2, 3, 4, 5] }` |

The second is the one that mattered. llama.cpp put a space after every `:` and
`,` and before the closing `}`. A converter that emitted the separators without
optional whitespace around them would reject llama.cpp's own valid output, and
**no hand-written test would have used that spacing** — every example anyone
writes by hand is minified.

## Two bugs, and what found each

**1. Only the first alternative of the root rule was ever explored.** A rule is
entered through a `RuleRef`, which fans out over its alternatives; the root has
no `RuleRef` pointing at it, so starting at element 0 explored one alternative
and no more. `root ::= "cat" | "car"` accepted `cat` and rejected `car`. Found
by a two-line unit test, and it would have made almost every real grammar
subtly wrong — with no error, just a token masked that should not have been.

**2. Three of the eight grammars llama.cpp ships did not parse.** `json.gbnf`,
`json_arr.gbnf` and `c.gbnf` all write the rule body on the line *after* `::=`.
Newlines end a rule body, so the parser saw an empty rule and then tried to read
`"{" ws (` as a rule name. A newline is allowed between `::=` and the body and
nowhere else in the sequence.

That second one is the argument for the test that walks the whole `grammars/`
directory. It found the bug on its first run, and it is the thing that notices
when upstream adds a construct this parser does not handle.

## What is refused, and why refusing is the safe direction

| refused | reason |
|---|---|
| `<think>`, `<[1000]>`, `!<...>` | token literals match *tokenizer tokens*, so resolving one needs the vocabulary at parse time |
| `allOf`, `not`, `if`/`then`/`else`, `pattern`, `patternProperties`, `prefixItems`, `uniqueItems`, `minLength`/`maxLength`, `minimum`/`maximum`, `multipleOf` | not implemented |
| `additionalProperties` other than `false` | would admit keys the schema never mentioned |
| a rule referenced but never defined | would become an empty rule |

Every one is refused **by name**, never ignored. Ignoring a constraint yields a
grammar *looser* than the schema asked for — output that satisfies the grammar
and violates the schema, which nothing downstream can detect. A refusal is a
message; a silent loosening is a bug in someone else's parser three days later.
The same reasoning already governs `VERIFIED_ARCHITECTURES` and
`tokenizer.ggml.pre`.

An undefined rule deserves its own note: left alone it is an empty rule that
matches nothing, so **every token is masked and generation simply stops**,
looking exactly like a well-behaved end of sequence. `TokenMask::is_empty`
exists so a caller can tell those apart.

## Details that are not obvious and are not arbitrary

The JSON primitive rules are copied character-for-character from
`common/json-schema-to-grammar.cpp`. `char` excludes `\x7F` and `\x00-\x1F`
because JSON forbids raw control characters in strings; `integral-part` is
`[0] | [1-9] [0-9]{0,15}` rather than `[0-9]+` because JSON forbids leading
zeros. Rewriting them "more clearly" is how you get a grammar that accepts `01`.

**A token may end half way through a character.** Terminals are code points, but
tokens are byte strings, and an emoji is four byte-fallback tokens under
SentencePiece. The trailing partial sequence is judged on the range of code
points it could still become. Treating it as a failure would mask every
byte-fallback token and make non-ASCII output impossible under any grammar.

## Cost

54 unit tests, 11 against llama.cpp, 1 doc test. `clippy -D warnings` and `fmt`
clean; 255 tests pass in the ggml-free job.
