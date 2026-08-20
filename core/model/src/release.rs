//! Which Chaos release is newest, and which file this platform needs.
//!
//! Atur: *"users can get the most updated release when they connect to the
//! internet from the app -- an updating flow, not every time go and download a
//! new setup. For all apps and exports we need."*
//!
//! Here rather than in `chaos-app` because of that last sentence. The window is
//! one of twelve binaries a release ships, and a CLI user who never opens it
//! needs the same answer -- `chaos-run --update`. `chaos-model` is the crate
//! both already depend on, and the crate that already owns the other thing that
//! talks to the network and downloads a file (`chaos-pull`).
//!
//! # What is here and what is not
//!
//! **The decisions are pure and tested; the one call that reaches the internet
//! is a `curl` invocation kept to a single function.** Everything else is a
//! function of a version string or a blob of JSON, so all of it runs in CI --
//! the comparison that decides whether to offer an update, the asset name for
//! the platform, and the parse of the releases feed.
//!
//! That split is deliberate. The interesting failures are "0.0.9 looked newer
//! than 0.0.11" and "downloaded the macOS tarball onto Windows", and neither of
//! those needs a socket to reproduce.
//!
//! # What an update is
//!
//! Chaos ships one installer per platform, and it carries **every** binary --
//! the window, `chaos-run`, `chaos-serve` and the rest. So one update updates
//! all of them, and the flow is *fetch the right asset and run it* rather than
//! a bespoke patching mechanism. The Windows installer already knows how to
//! upgrade in place: it reports "Reinstalling" or "UPDATE", and it leaves the
//! models directory alone.

/// A released version, compared the way people expect.
///
/// **String comparison is wrong and looks right for a long time.** `"0.0.9"` is
/// greater than `"0.0.11"` alphabetically, so a user on 0.0.9 would be told
/// they were up to date, and one on 0.0.11 would be offered a downgrade —
/// silently, and only once the tenth release existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u32, pub u32, pub u32);

impl Version {
    /// Parse `1.2.3`, or `v1.2.3` as a tag carries it.
    ///
    /// Anything trailing the third number is ignored, so `0.0.12-rc1` parses as
    /// 0.0.12: a pre-release is not a different number, and treating it as
    /// unparseable would silently stop offering updates.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().trim_start_matches(['v', 'V']);
        let mut it = s.split('.');
        let major = it.next()?.parse().ok()?;
        let minor = it.next()?.parse().ok()?;
        let patch_part = it.next().unwrap_or("0");
        let end = patch_part
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(patch_part.len());
        let patch = patch_part[..end].parse().ok()?;
        Some(Version(major, minor, patch))
    }

    pub fn text(&self) -> String {
        format!("{}.{}.{}", self.0, self.1, self.2)
    }
}

/// The version this binary was built as.
pub fn running() -> Version {
    Version::parse(env!("CARGO_PKG_VERSION")).unwrap_or(Version(0, 0, 0))
}

/// Where the newest release is described.
///
/// The API rather than the HTML page: it names the assets and their sizes, and
/// it does not change shape when the site is redesigned.
pub const LATEST_URL: &str = "https://api.github.com/repos/aturzone/Chaos/releases/latest";

/// The name of the release asset built for the machine this is running on.
///
/// **Both halves matter, and getting either wrong is silent.** A macOS tarball
/// saves perfectly well onto Windows and then does nothing, which reads as a
/// broken updater rather than as the wrong file — and an arm64 binary on an
/// Intel Mac is the same failure one level down: it downloads, it unpacks, and
/// then nothing runs.
///
/// So the architecture is read from the build, not assumed. A release ships
/// five: Windows x86_64, macOS on both arm64 and x86_64, and Linux on both
/// x86_64 and arm64.
pub fn asset_for_platform(version: &Version) -> String {
    let v = version.text();
    // The names the release workflow's matrix produces. `arm64` rather than
    // Rust's `aarch64`, because the asset names are what a person reads on the
    // releases page.
    let arm = cfg!(target_arch = "aarch64");
    if cfg!(windows) {
        // One Windows build. When an arm64 one exists this needs the same
        // branch the others have.
        format!("Chaos-v{v}-windows-x86_64-Setup.exe")
    } else if cfg!(target_os = "macos") {
        let arch = if arm { "arm64" } else { "x86_64" };
        format!("Chaos-v{v}-macos-{arch}.tar.gz")
    } else {
        let arch = if arm { "arm64" } else { "x86_64" };
        format!("Chaos-v{v}-linux-{arch}.tar.gz")
    }
}

/// What the releases feed said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub version: Version,
    /// `(file name, download URL)` for every asset.
    pub assets: Vec<(String, String)>,
}

impl Release {
    /// The download URL for this platform's installer, if the release has one.
    pub fn asset_url(&self) -> Option<&str> {
        let want = asset_for_platform(&self.version);
        self.assets
            .iter()
            .find(|(n, _)| *n == want)
            .map(|(_, u)| u.as_str())
    }
}

/// Read the tag and assets out of the releases API's answer.
///
/// Hand-scanned rather than parsed into a document: the answer is 30 kB of
/// fields this does not care about, and the three it does are unambiguous
/// strings. A JSON parser here would be a dependency edge for `tag_name`.
pub fn parse_latest(json: &str) -> Option<Release> {
    let tag = field(json, "\"tag_name\"")?;
    let version = Version::parse(&tag)?;
    let mut assets = Vec::new();
    // Each asset object carries a "name" and a "browser_download_url"; walking
    // pairwise keeps them associated even though the scan is flat.
    let mut rest = json;
    while let Some(i) = rest.find("\"browser_download_url\"") {
        // The name precedes the URL inside the same object.
        let before = &rest[..i];
        let name = before
            .rfind("\"name\"")
            .and_then(|j| field(&before[j..], "\"name\""));
        let url = field(&rest[i..], "\"browser_download_url\"");
        if let (Some(n), Some(u)) = (name, url) {
            assets.push((n, u));
        }
        rest = &rest[i + 22..];
    }
    Some(Release { version, assets })
}

/// The string value of `key` in `json`, starting from the first occurrence.
fn field(json: &str, key: &str) -> Option<String> {
    let i = json.find(key)? + key.len();
    let rest = &json[i..];
    let c = rest.find(':')? + 1;
    let rest = &rest[c..];
    let open = rest.find('"')? + 1;
    let rest = &rest[open..];
    // No escape handling: none of the three fields read here can contain a
    // quote, and inventing an unescaper for a version tag would be the more
    // likely bug.
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// What the window should say and offer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Nothing newer exists.
    UpToDate(Version),
    /// A newer release, and where its installer is.
    Available { version: Version, url: String },
    /// Newer, but with no installer for this platform.
    NoAssetForPlatform(Version),
    /// The check itself did not work.
    Failed(String),
}

impl Outcome {
    /// One line for the status bar.
    pub fn line(&self) -> String {
        match self {
            Outcome::UpToDate(v) => format!("Chaos {} is the newest release", v.text()),
            Outcome::Available { version, .. } => {
                format!(
                    "Chaos {} is available -- Help > Install update",
                    version.text()
                )
            }
            Outcome::NoAssetForPlatform(v) => format!(
                "Chaos {} is available, but has no installer for this platform",
                v.text()
            ),
            Outcome::Failed(why) => format!("could not check for updates: {why}"),
        }
    }
}

/// Compare a fetched release against what is running.
pub fn decide(latest: Option<Release>, running: Version) -> Outcome {
    let Some(r) = latest else {
        return Outcome::Failed("the release feed could not be read".into());
    };
    if r.version <= running {
        return Outcome::UpToDate(running);
    }
    match r.asset_url() {
        Some(u) => Outcome::Available {
            version: r.version,
            url: u.to_string(),
        },
        None => Outcome::NoAssetForPlatform(r.version),
    }
}

/// The arguments that fetch the release feed.
///
/// One list, used by the window and by `chaos-run --update`, because the
/// interesting one is easy to leave out: **the GitHub API rejects a request
/// with no `User-Agent` outright**, and the failure is a 403 rather than
/// anything that mentions headers.
pub fn feed_curl_args() -> Vec<&'static str> {
    vec![
        "-L",
        "--fail",
        "-sS",
        "--max-time",
        "20",
        "-H",
        "User-Agent: Chaos",
        "-H",
        "Accept: application/vnd.github+json",
        LATEST_URL,
    ]
}

/// Ask GitHub what the newest release is.
///
/// Through `curl` for the same reason `chaos-pull` does: the workspace has no
/// external Rust dependencies, and an HTTPS client is not worth being the thing
/// that ends that.
///
/// The window does not call this -- it needs `CREATE_NO_WINDOW` on the command
/// so no console flashes, which is a Windows-only extension to `Command`. It
/// builds the same call from [`feed_curl_args`].
pub fn fetch_latest_json() -> Result<String, String> {
    let out = std::process::Command::new("curl")
        .args(feed_curl_args())
        .output()
        .map_err(|e| format!("curl could not be run ({e})"))?;
    if out.status.success() {
        return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
    }
    let why = String::from_utf8_lossy(&out.stderr);
    let first = why.lines().next().unwrap_or("").trim().to_string();
    Err(if first.is_empty() {
        format!("curl exited {}", out.status)
    } else {
        first
    })
}

/// The arguments that fetch a release asset. The caller appends `-o <path>`
/// and the URL.
///
/// `-L` follows the redirect to the asset host, and `--fail` makes an HTTP
/// error an error rather than a saved error page -- which is how a 401 becomes
/// a corrupt download that passes every other check.
pub fn asset_curl_args() -> Vec<&'static str> {
    vec!["-L", "--fail", "-sS", "--retry", "3"]
}

/// Below this, whatever was saved is not an installer.
///
/// **Exit zero is not a file.** `curl` reports success after saving a redirect
/// to an error page, which is the trap `chaos-pull` documents at length and the
/// one that put a corrupt .gguf on this machine. No installer this project has
/// ever built is under a megabyte, so a short file is a failed download wearing
/// the right name.
pub const MIN_INSTALLER_BYTES: u64 = 1 << 20;

/// Check, without needing anything from the caller.
pub fn check() -> Outcome {
    match fetch_latest_json() {
        Ok(json) => decide(parse_latest(&json), running()),
        Err(why) => Outcome::Failed(why),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Versions compare as numbers. Alphabetically 0.0.9 beats 0.0.11, which
    /// would have told everyone on 0.0.9 they were current.
    #[test]
    fn versions_compare_as_numbers_not_as_text() {
        assert!(Version::parse("0.0.11").unwrap() > Version::parse("0.0.9").unwrap());
        assert!("0.0.11" < "0.0.9", "which is exactly the trap");
        assert!(Version::parse("1.0.0").unwrap() > Version::parse("0.9.9").unwrap());
        assert_eq!(Version::parse("v0.0.12"), Some(Version(0, 0, 12)));
        assert_eq!(Version::parse("0.0.12"), Some(Version(0, 0, 12)));
        // A pre-release is that version, not an unreadable one -- returning
        // None here would quietly stop offering updates.
        assert_eq!(Version::parse("0.0.12-rc1"), Some(Version(0, 0, 12)));
        assert_eq!(Version::parse("1.2"), Some(Version(1, 2, 0)));
        assert_eq!(Version::parse("banana"), None);
        assert_eq!(Version::parse(""), None);
    }

    /// The five assets a release ships, shaped as GitHub returns them: the
    /// download URL ends in the asset's own filename, which is what makes a
    /// mis-pairing detectable rather than plausible.
    fn feed(tag: &str) -> String {
        let a =
            |n: &str| format!(r#"{{"name":"{n}","browser_download_url":"https://example/{n}"}}"#);
        let assets = [
            a(&format!("Chaos-{tag}-windows-x86_64-Setup.exe")),
            a(&format!("Chaos-{tag}-linux-x86_64.tar.gz")),
            a(&format!("Chaos-{tag}-linux-arm64.tar.gz")),
            a(&format!("Chaos-{tag}-macos-arm64.tar.gz")),
            a(&format!("Chaos-{tag}-macos-x86_64.tar.gz")),
        ]
        .join(",");
        format!(r#"{{"tag_name":"{tag}","name":"{tag}","assets":[{assets}]}}"#)
    }

    #[test]
    fn the_feed_gives_up_its_tag_and_assets() {
        let r = parse_latest(&feed("v0.0.12")).expect("parsed");
        assert_eq!(r.version, Version(0, 0, 12));
        assert_eq!(r.assets.len(), 5, "an asset was lost or invented");
        // The name and the URL of an asset stay together -- for every one of
        // them, not only the first. Pairing by scanning back from each URL is
        // the part that can go wrong.
        assert!(r.assets[0].0.contains("Setup.exe"));
        for (name, url) in &r.assets {
            assert!(
                url.ends_with(name.as_str()),
                "{name} was paired with {url}, which is a different build"
            );
        }
        // Nonsense in gives nothing out rather than a wrong version.
        assert!(parse_latest("not json at all").is_none());
        assert!(parse_latest("{}").is_none());
    }

    /// The platform's own installer is chosen, never another one.
    #[test]
    fn the_asset_matches_the_platform() {
        let r = parse_latest(&feed("v0.0.12")).expect("parsed");
        let url = r.asset_url().expect("an asset for this platform");
        // **The architecture as well as the operating system.** An arm64
        // tarball on an Intel Mac downloads, unpacks and then runs nothing --
        // the same silent failure as the wrong OS, one level down.
        let want = format!("https://example/{}", asset_for_platform(&r.version));
        assert_eq!(url, want, "picked the wrong build for this machine");
        // And that name really is this machine's: OS and architecture both.
        let os = if cfg!(windows) {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            "linux"
        };
        let arch = if cfg!(target_arch = "aarch64") && !cfg!(windows) {
            "arm64"
        } else {
            "x86_64"
        };
        assert!(url.contains(os) && url.contains(arch), "{url}");
    }

    /// The five names are exactly the five the release workflow produces.
    ///
    /// **A name is the whole contract here.** The updater finds its download by
    /// string equality against the feed, so a matrix entry renamed in
    /// `release.yml` and not here is an updater that reports "no installer for
    /// this platform" forever, on that platform only, with nothing in a log.
    #[test]
    fn the_asset_names_are_the_ones_the_release_builds() {
        let v = Version(0, 0, 12);
        let mine = asset_for_platform(&v);
        let shipped = [
            "Chaos-v0.0.12-windows-x86_64-Setup.exe",
            "Chaos-v0.0.12-linux-x86_64.tar.gz",
            "Chaos-v0.0.12-linux-arm64.tar.gz",
            "Chaos-v0.0.12-macos-arm64.tar.gz",
            "Chaos-v0.0.12-macos-x86_64.tar.gz",
        ];
        assert!(
            shipped.contains(&mine.as_str()),
            "{mine} is not one of the assets a release builds"
        );
    }

    #[test]
    fn a_newer_release_is_offered_and_an_older_one_is_not() {
        let newer = parse_latest(&feed("v0.0.12"));
        match decide(newer.clone(), Version(0, 0, 11)) {
            Outcome::Available { version, url } => {
                assert_eq!(version, Version(0, 0, 12));
                assert!(url.starts_with("https://"));
            }
            other => panic!("expected an offer, got {other:?}"),
        }
        // The same version is not an update, and neither is an older one.
        assert_eq!(
            decide(newer.clone(), Version(0, 0, 12)),
            Outcome::UpToDate(Version(0, 0, 12))
        );
        assert_eq!(
            decide(newer, Version(1, 0, 0)),
            Outcome::UpToDate(Version(1, 0, 0))
        );
        // A failed fetch says so rather than claiming to be current, which
        // would be a lie that hides a broken updater forever.
        assert!(matches!(
            decide(None, Version(0, 0, 11)),
            Outcome::Failed(_)
        ));
    }

    /// A release with nothing for this platform says so, rather than offering a
    /// download that would do nothing.
    #[test]
    fn a_release_without_this_platforms_installer_is_named_as_such() {
        let json = r#"{"tag_name":"v9.9.9","assets":[
            {"name":"Chaos-v9.9.9-freebsd.tar.gz","browser_download_url":"https://example/bsd.tgz"}]}"#;
        let r = parse_latest(json).expect("parsed");
        assert_eq!(
            decide(Some(r), Version(0, 0, 11)),
            Outcome::NoAssetForPlatform(Version(9, 9, 9))
        );
    }

    /// Every outcome says something a person can act on.
    #[test]
    fn every_outcome_reads_as_a_sentence() {
        assert!(Outcome::UpToDate(Version(0, 0, 11))
            .line()
            .contains("newest"));
        assert!(Outcome::Available {
            version: Version(0, 0, 12),
            url: "x".into()
        }
        .line()
        .contains("0.0.12"));
        assert!(Outcome::Failed("no network".into())
            .line()
            .contains("no network"));
    }
}
