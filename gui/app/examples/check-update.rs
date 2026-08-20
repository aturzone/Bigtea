//! What the update check makes of a real releases feed.
//!
//! The unit tests parse a feed this file wrote, which proves the parser agrees
//! with itself. This one runs it against **GitHub's actual answer**, which is
//! the shape that can change without warning.
//!
//! ```text
//! cargo run --release -p chaos-app --example check-update
//! cargo run --release -p chaos-app --example check-update -- some-saved.json
//! ```

fn main() {
    let arg = std::env::args().nth(1);
    let json = match &arg {
        Some(p) => std::fs::read_to_string(p).expect("read the saved feed"),
        None => {
            let out = std::process::Command::new("curl")
                .args([
                    "-sS",
                    "-L",
                    "--fail",
                    "--max-time",
                    "20",
                    "-H",
                    "User-Agent: Chaos",
                    "-H",
                    "Accept: application/vnd.github+json",
                    chaos_app::update::LATEST_URL,
                ])
                .output()
                .expect("run curl");
            if !out.status.success() {
                eprintln!("curl: {}", String::from_utf8_lossy(&out.stderr));
                std::process::exit(1);
            }
            String::from_utf8_lossy(&out.stdout).into_owned()
        }
    };

    let running = chaos_app::update::running();
    println!("running   {}", running.text());
    match chaos_app::update::parse_latest(&json) {
        Some(r) => {
            println!("latest    {}", r.version.text());
            println!("assets    {}", r.assets.len());
            for (n, u) in &r.assets {
                println!("  {n}\n    {u}");
            }
            match r.asset_url() {
                Some(u) => println!("\nthis platform wants:\n  {u}"),
                None => println!("\nNO ASSET FOR THIS PLATFORM"),
            }
            println!("\n{}", chaos_app::update::decide(Some(r), running).line());
        }
        None => println!("the feed did not parse -- its shape has changed"),
    }
}
