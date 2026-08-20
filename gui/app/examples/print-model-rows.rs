//! Print the MODELS page's rows, through the code the window uses.
//!
//! **Because driving the GUI to read them was not working.** Two attempts to
//! click MODELS with `SendKeys` and `mouse_event` both captured the CHAT page
//! instead, and a screenshot that shows the wrong page proves nothing. The rows
//! come from `catalog::row`, so calling it directly is the same evidence without
//! the window in the way — and it can be diffed by eye at any memory size.
//!
//! Takes the free-memory figure to use, in GiB, so all three verdicts can be
//! seen without owning three machines:
//!
//! ```text
//! cargo run -p chaos-app --example print-model-rows -- 64
//! cargo run -p chaos-app --example print-model-rows -- 16
//! cargo run -p chaos-app --example print-model-rows -- 4
//! ```

fn main() {
    let gib: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16.0);
    let free = (gib * 1024.0 * 1024.0 * 1024.0) as u64;
    println!("as the window would show them with {gib} GiB free:\n");
    for o in chaos_app::catalog::offers() {
        println!("  {}", chaos_app::catalog::row(&o, free));
    }
    println!();
    println!("Three verdicts, three speeds. None of them is a refusal:");
    println!("  fits            everything resident");
    println!("  streams         the always-read set fits; experts come off disk");
    println!("  slow, re-reads  it does not fit and it still runs");
}
