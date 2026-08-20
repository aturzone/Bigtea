//! Verify direct I/O against a real file on the real device.
//!
//! Unit tests use temp files, which can live on a filesystem that quietly
//! refuses `O_DIRECT` / `FILE_FLAG_NO_BUFFERING`. The assumption that the fast
//! path actually engages on the machine's real storage is load-bearing — if it
//! silently falls back to buffered, every bandwidth number we report is the
//! page cache's rather than the disk's. So it gets checked against a real file.
//!
//! Skips cleanly when the model is absent.

use std::path::PathBuf;

use chaos_io::{DirectFile, IoMode};

const DEFAULT_PATH: &str =
    r"C:\Projects\models\v4flash\DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf";

fn container() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CHAOS_TEST_GGUF") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(DEFAULT_PATH);
    p.exists().then_some(p)
}

#[test]
fn direct_io_engages_on_real_storage() {
    let Some(path) = container() else {
        eprintln!("skipping: no container present");
        return;
    };
    let f = DirectFile::open(&path).expect("open");
    assert_eq!(
        f.mode(),
        IoMode::Direct,
        "cache-bypassing I/O did not engage on this device; every bandwidth \
         measurement taken through it would be the page cache's, not the disk's"
    );
    assert!(!f.is_empty());
}

#[test]
fn direct_reads_match_buffered_reads_on_a_real_file() {
    // The fast path must never disagree with the slow one, on real data with
    // real offsets -- including a GGUF header, which starts unaligned almost
    // everywhere after the magic.
    let Some(path) = container() else {
        eprintln!("skipping: no container present");
        return;
    };
    let direct = DirectFile::open(&path).expect("open direct");
    let buffered = DirectFile::open_buffered(&path).expect("open buffered");
    let len = direct.len();

    let cases: &[(u64, usize)] = &[
        (0, 4),       // GGUF magic
        (0, 4096),    // first sector
        (13, 977),    // deliberately awkward
        (4095, 8194), // straddles three sectors
        (len - 1, 1), // final byte
    ];
    for &(offset, want) in cases {
        if offset + want as u64 > len {
            continue;
        }
        let a = direct.read_at(offset, want).expect("direct read");
        let b = buffered.read_at(offset, want).expect("buffered read");
        assert_eq!(a, b, "direct and buffered disagree at {offset}+{want}");
        assert_eq!(a.len(), want);
    }
}

#[test]
fn the_gguf_magic_is_readable_through_direct_io() {
    // End-to-end sanity: the bytes we read really are the file's, not a
    // zeroed buffer that happens to be the right length.
    let Some(path) = container() else {
        eprintln!("skipping: no container present");
        return;
    };
    let f = DirectFile::open(&path).expect("open");
    let magic = f.read_at(0, 4).expect("read magic");
    assert_eq!(&magic, b"GGUF", "first four bytes should be the GGUF magic");
}
