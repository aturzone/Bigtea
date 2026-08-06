## What this changes

<!-- One paragraph. What problem does this solve? -->

## What was measured

<!-- Required for any performance change.
     Include the command line, the hardware, and the before/after numbers.
     "Should be faster" is not a result — this project is nought for four on
     predicting performance without measuring it. -->

## How correctness was checked

<!-- A wrong forward pass, tokenizer, mask or cache produces fluent nonsense
     rather than an error. What test would fail if the numbers were wrong? -->

## Checklist

- [ ] `cargo test --release` passes
- [ ] `cargo clippy --all-targets -- -D warnings` is clean
- [ ] `cargo fmt --check` is clean
- [ ] Comments explain *why*, not *what*
- [ ] If a claim about another engine is made, its exact command line and output are in a doc
