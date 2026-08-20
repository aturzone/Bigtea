//! The update check, against GitHub's actual answer.
//!
//! The unit tests in `release.rs` parse a feed the same file wrote, which
//! proves only that it agrees with itself. `github-release.json` beside this
//! file is the **real response** for v0.0.11, trimmed to two assets but
//! otherwise untouched: real key order, the `uploader` object nested inside
//! each asset, `"label":null`, all of it.

use chaos_model::release::{self, Outcome, Version};

const FEED: &str = include_str!("github-release.json");

/// The invariant this pins is narrow and load-bearing.
///
/// Assets are found by scanning back from each `browser_download_url` to the
/// nearest `"name"`, and that is correct only because nothing between them
/// carries a `"name"` of its own -- the uploader object has `login`, not
/// `name`, and the release's own `"name"` sits before the first asset rather
/// than inside it. Scanning *forward* instead labels every asset `v0.0.11`,
/// which is what this catches.
#[test]
fn the_real_github_response_parses() {
    let r = release::parse_latest(FEED).expect("the real feed did not parse");
    assert_eq!(r.version, Version(0, 0, 11));
    assert_eq!(r.assets.len(), 2, "an asset was lost or invented");
    for (name, url) in &r.assets {
        assert!(
            name.starts_with("Chaos-v0.0.11-"),
            "{name:?} is not an asset name -- the scan found the wrong field"
        );
        assert!(
            url.ends_with(name.as_str()),
            "{name} was paired with {url}, which is a different file"
        );
    }
}

/// The platform's own installer comes back out of it, and a build of that
/// version is not offered an update to itself.
#[test]
fn the_feed_yields_this_platforms_installer() {
    let r = release::parse_latest(FEED).expect("parsed");
    let want = release::asset_for_platform(&r.version);
    // The fixture carries the Windows and macOS assets; a Linux CI run has
    // nothing to match, and saying so is the correct answer there.
    match r.asset_url() {
        Some(u) => assert!(u.ends_with(&want), "picked the wrong file: {u}"),
        None => assert!(
            want.contains("linux"),
            "no asset matched {want}, and the fixture has one"
        ),
    }
    assert_eq!(
        release::decide(Some(r), Version(0, 0, 11)),
        Outcome::UpToDate(Version(0, 0, 11))
    );
}

/// A feed one version ahead is offered.
///
/// The same fixture, with the tag moved forward: the only difference that
/// should matter is the number.
#[test]
fn a_newer_tag_in_the_real_shape_is_offered() {
    let ahead = FEED.replace("\"tag_name\":\"v0.0.11\"", "\"tag_name\":\"v9.9.9\"");
    assert_ne!(ahead, FEED, "the tag was not where it was expected");
    let r = release::parse_latest(&ahead).expect("parsed");
    assert_eq!(r.version, Version(9, 9, 9));
    // The assets still say v0.0.11, so no installer matches the new version --
    // and the honest answer is to say so rather than to offer the old file
    // under the new number.
    assert_eq!(
        release::decide(Some(r), Version(0, 0, 11)),
        Outcome::NoAssetForPlatform(Version(9, 9, 9))
    );
}
