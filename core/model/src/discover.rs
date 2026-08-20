//! Find every shard of a split model, given any one of them.
//!
//! llama.cpp names split containers `<stem>-00001-of-00005.gguf`. Users
//! reasonably point a tool at "the model" — usually shard one — and expect it
//! to find the rest, so that is what this does.

use std::path::{Path, PathBuf};

/// Split `name` into `(prefix, index, total, suffix)` when it looks like a
/// shard: `foo-00002-of-00005.gguf` -> `("foo-", 2, 5, ".gguf")`.
fn parse_shard_name(file_name: &str) -> Option<(&str, u32, u32, &str)> {
    let (head, tail) = file_name.rsplit_once("-of-")?;
    // `head` ends with the shard index, `tail` starts with the total.
    let (prefix, index) = {
        let digits_start = head.len() - head.chars().rev().take_while(char::is_ascii_digit).count();
        if digits_start == head.len() {
            return None; // no digits before "-of-"
        }
        (
            &head[..digits_start],
            head[digits_start..].parse::<u32>().ok()?,
        )
    };
    let digits_len = tail.chars().take_while(char::is_ascii_digit).count();
    if digits_len == 0 {
        return None;
    }
    let total = tail[..digits_len].parse::<u32>().ok()?;
    let suffix = &tail[digits_len..];
    Some((prefix, index, total, suffix))
}

/// All shards belonging to the same split as `path`, in shard order.
///
/// Returns just `path` when it is not a split container, or when siblings are
/// missing — a caller working with a partial download should still get what
/// exists rather than an error.
pub fn discover_shards(path: &Path) -> Vec<PathBuf> {
    let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
        return vec![path.to_path_buf()];
    };
    let Some((prefix, _index, total, suffix)) = parse_shard_name(file_name) else {
        return vec![path.to_path_buf()];
    };
    let dir = path.parent().unwrap_or(Path::new("."));

    // Shard numbers are zero-padded to the width they were written with.
    let width = file_name
        .rsplit_once("-of-")
        .map(|(head, _)| head.chars().rev().take_while(char::is_ascii_digit).count())
        .unwrap_or(5);

    let mut found = Vec::new();
    for n in 1..=total {
        let candidate = dir.join(format!("{prefix}{n:0width$}-of-{total:0width$}{suffix}"));
        if candidate.exists() {
            found.push(candidate);
        }
    }
    if found.is_empty() {
        vec![path.to_path_buf()]
    } else {
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_cpp_split_names() {
        let (prefix, index, total, suffix) =
            parse_shard_name("DeepSeek-V4-Flash-UD-Q4_K_XL-00002-of-00005.gguf").unwrap();
        assert_eq!(prefix, "DeepSeek-V4-Flash-UD-Q4_K_XL-");
        assert_eq!(index, 2);
        assert_eq!(total, 5);
        assert_eq!(suffix, ".gguf");
    }

    #[test]
    fn rejects_names_that_are_not_splits() {
        assert!(parse_shard_name("model.gguf").is_none());
        assert!(parse_shard_name("weird-of-thing.gguf").is_none());
        assert!(parse_shard_name("-of-00005.gguf").is_none());
    }

    #[test]
    fn a_single_file_discovers_only_itself() {
        let p = Path::new("some/dir/model.gguf");
        assert_eq!(discover_shards(p), vec![p.to_path_buf()]);
    }

    #[test]
    fn missing_siblings_do_not_fail() {
        // Nothing on disk: the caller still gets a usable path back.
        let p = Path::new("nowhere/model-00001-of-00005.gguf");
        let found = discover_shards(p);
        assert_eq!(found, vec![p.to_path_buf()]);
    }

    #[test]
    fn shard_numbers_keep_their_padding() {
        // Naive formatting would look for "model-1-of-5.gguf" and find nothing.
        let (prefix, _, total, suffix) = parse_shard_name("model-00001-of-00012.gguf").unwrap();
        assert_eq!(
            format!("{prefix}{:05}-of-{total:05}{suffix}", 7),
            "model-00007-of-00012.gguf"
        );
    }
}
