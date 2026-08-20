//! Write a PNG, with no dependencies.
//!
//! # Why this exists rather than the `png` crate
//!
//! Nothing in this workspace has an external dependency, and that rule is
//! load-bearing: it is why a release binary starts on a machine with no runtime
//! installed. Reaching for `png` would also pull `flate2`, which pulls a C zlib
//! or a Rust reimplementation of it — two crates and a build-time choice, to
//! emit a file format that is a signature, three chunks and a checksum.
//!
//! # Why the pixel data is stored rather than compressed
//!
//! A PNG's `IDAT` holds a zlib stream, and zlib's deflate has three block types:
//! stored, fixed Huffman, and dynamic Huffman. **Stored is a legal deflate
//! block** — every decoder in existence reads it, because it is the fallback
//! deflate itself uses on incompressible input. It costs five bytes per 65,535
//! and no compression.
//!
//! That is the right first answer here. A 1024×1024 RGB image is 3.1 MB raw; a
//! real deflate might reach 2.6 MB on photographic content, which is not worth
//! a Huffman coder and its bugs on the day the pipeline first produces pixels.
//! When it becomes worth it, [`deflate_stored`] is the only function that has to
//! change, and the round-trip test above it will still pass.
//!
//! # What is deliberately not here
//!
//! No interlacing, no palettes, no 16-bit channels, no ancillary chunks. A
//! generated image is 8-bit RGB, written once, and read by an image viewer.

/// Bytes per pixel for the only colour type this writes: 8-bit RGB.
const RGB: usize = 3;

/// The eight bytes every PNG starts with.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Encode `w` × `h` 8-bit RGB pixels as a PNG.
///
/// `rgb` is row-major, three bytes per pixel, top row first — the layout a
/// decoder hands back and the one a VAE produces after the channel dimension is
/// moved last. Returns `None` when the length does not match the dimensions,
/// because a truncated image is a bug worth surfacing rather than padding.
pub fn encode_rgb(w: u32, h: u32, rgb: &[u8]) -> Option<Vec<u8>> {
    let expect = (w as usize).checked_mul(h as usize)?.checked_mul(RGB)?;
    if rgb.len() != expect || w == 0 || h == 0 {
        return None;
    }

    // **A filter byte per scanline, and it is not optional.** PNG's rows each
    // begin with a filter type; 0 means "this row is stored as-is". Forgetting
    // it shifts every row by one byte and produces an image that decodes
    // without error and looks like diagonal static.
    let mut raw = Vec::with_capacity(h as usize * (1 + w as usize * RGB));
    for row in rgb.chunks_exact(w as usize * RGB) {
        raw.push(0);
        raw.extend_from_slice(row);
    }

    let mut out = Vec::with_capacity(raw.len() + 1024);
    out.extend_from_slice(&SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(2); // colour type 2 = truecolour RGB
    ihdr.push(0); // deflate, the only compression PNG defines
    ihdr.push(0); // adaptive filtering, the only filter method
    ihdr.push(0); // no interlacing
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib(&raw));
    chunk(&mut out, b"IEND", &[]);
    Some(out)
}

/// Append one PNG chunk: length, type, data, CRC over type and data.
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    // **The CRC covers the type as well as the data**, and not the length. Every
    // one of those three is easy to get wrong and each produces a file that one
    // decoder accepts and another rejects.
    let mut crc = Crc::new();
    crc.update(kind);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// Wrap deflate output in the zlib container `IDAT` requires.
fn zlib(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 16);
    // 0x78 0x01: deflate, 32 KiB window, no preset dictionary, fastest level.
    // The pair must be a multiple of 31 read big-endian, which 0x7801 is.
    out.push(0x78);
    out.push(0x01);
    out.extend_from_slice(&deflate_stored(raw));
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

/// Deflate `raw` as stored blocks: legal, and no compression.
///
/// Each block is a three-bit header padded to a byte, then LEN and its
/// one's complement as little-endian 16-bit, then the bytes. The final block
/// sets the low bit.
fn deflate_stored(raw: &[u8]) -> Vec<u8> {
    const MAX: usize = 65_535;
    let mut out = Vec::with_capacity(raw.len() + raw.len() / MAX * 5 + 5);
    // An empty input still needs one final, empty block, or the stream is
    // truncated rather than empty.
    if raw.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
        return out;
    }
    let mut chunks = raw.chunks(MAX).peekable();
    while let Some(part) = chunks.next() {
        out.push(u8::from(chunks.peek().is_none()));
        let len = part.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(part);
    }
    out
}

/// Adler-32, the checksum zlib puts after the deflate stream.
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    // 5552 is the most bytes that can accumulate before `b` can overflow a u32,
    // so the modulo happens per block rather than per byte.
    for block in data.chunks(5552) {
        for &byte in block {
            a += u32::from(byte);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// CRC-32 as PNG defines it: the reflected polynomial, pre- and post-inverted.
struct Crc(u32);

impl Crc {
    fn new() -> Self {
        Self(0xFFFF_FFFF)
    }

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let mut c = (self.0 ^ u32::from(byte)) & 0xFF;
            for _ in 0..8 {
                // 0xEDB8_8320 is 0x04C1_1DB7 bit-reversed, which is the form
                // that matches shifting right.
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            self.0 = c ^ (self.0 >> 8);
        }
    }

    fn finish(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two checksums, against values that can be checked by hand.
    ///
    /// A wrong CRC produces a file every decoder rejects, and a wrong Adler-32
    /// produces one that some decoders accept — so the second is the more
    /// dangerous of the two and the reason both are pinned.
    #[test]
    fn the_checksums_are_the_documented_ones() {
        // The canonical CRC-32 check value: "123456789" is 0xCBF43926.
        let mut crc = Crc::new();
        crc.update(b"123456789");
        assert_eq!(crc.finish(), 0xCBF4_3926);

        // Adler-32 of "Wikipedia" is 0x11E60398, from the specification's own
        // worked example.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);

        // Empty input: Adler starts at 1 and stays there.
        assert_eq!(adler32(b""), 1);
    }

    /// Stored deflate blocks are shaped as the format requires.
    #[test]
    fn stored_blocks_carry_len_and_its_complement() {
        let out = deflate_stored(b"abc");
        // final block, then 3 and !3 little-endian, then the bytes
        assert_eq!(out[0], 1, "the only block must be marked final");
        assert_eq!(u16::from_le_bytes([out[1], out[2]]), 3);
        assert_eq!(u16::from_le_bytes([out[3], out[4]]), !3u16);
        assert_eq!(&out[5..], b"abc");

        // Over one block: the first is not final, the last is.
        let big = vec![7u8; 70_000];
        let out = deflate_stored(&big);
        assert_eq!(out[0], 0, "a non-final block must not set the low bit");
        assert_eq!(u16::from_le_bytes([out[1], out[2]]), 65_535);
        let second = 5 + 65_535;
        assert_eq!(out[second], 1, "the last block must be final");

        // Empty is one empty final block, not nothing at all.
        assert_eq!(deflate_stored(b""), vec![1, 0, 0, 0xFF, 0xFF]);
    }

    /// A PNG this writes can be read back, structurally and by pixel.
    ///
    /// Read by a parser written here rather than by a crate, which is the only
    /// way to check it without a dependency — and it catches the failures that
    /// matter: a missing filter byte, a chunk length that disagrees with its
    /// data, a CRC over the wrong bytes.
    #[test]
    fn a_written_png_reads_back_with_the_same_pixels() {
        // Three pixels wide, two tall, with values that make a row shift
        // obvious: a missing filter byte moves every row by one.
        let w = 3u32;
        let h = 2u32;
        let pixels: Vec<u8> = vec![
            255, 0, 0, 0, 255, 0, 0, 0, 255, // red, green, blue
            10, 20, 30, 40, 50, 60, 70, 80, 90,
        ];
        let png = encode_rgb(w, h, &pixels).expect("encode");

        assert_eq!(&png[..8], &SIGNATURE, "signature");

        // Walk the chunks, checking every CRC as we go.
        let mut i = 8;
        let mut seen: Vec<String> = Vec::new();
        let mut idat: Vec<u8> = Vec::new();
        let mut dims = (0u32, 0u32);
        while i + 8 <= png.len() {
            let len = u32::from_be_bytes([png[i], png[i + 1], png[i + 2], png[i + 3]]) as usize;
            let kind = &png[i + 4..i + 8];
            let data = &png[i + 8..i + 8 + len];
            let want = u32::from_be_bytes([
                png[i + 8 + len],
                png[i + 9 + len],
                png[i + 10 + len],
                png[i + 11 + len],
            ]);
            let mut crc = Crc::new();
            crc.update(kind);
            crc.update(data);
            assert_eq!(
                crc.finish(),
                want,
                "CRC mismatch on chunk {:?}",
                String::from_utf8_lossy(kind)
            );
            let name = String::from_utf8_lossy(kind).to_string();
            if name == "IHDR" {
                dims = (
                    u32::from_be_bytes([data[0], data[1], data[2], data[3]]),
                    u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
                );
                assert_eq!(data[8], 8, "bit depth");
                assert_eq!(data[9], 2, "colour type RGB");
            }
            if name == "IDAT" {
                idat.extend_from_slice(data);
            }
            seen.push(name);
            i += 12 + len;
        }
        assert_eq!(seen, vec!["IHDR", "IDAT", "IEND"], "chunk order");
        assert_eq!(dims, (w, h));
        assert_eq!(i, png.len(), "trailing bytes after IEND");

        // Undo the zlib wrapper and the stored blocks, then the filter bytes.
        assert_eq!(&idat[..2], &[0x78, 0x01], "zlib header");
        assert_eq!(
            (u32::from(idat[0]) << 8 | u32::from(idat[1])) % 31,
            0,
            "the zlib header pair must be a multiple of 31"
        );
        let deflated = &idat[2..idat.len() - 4];
        let mut raw = Vec::new();
        let mut p = 0;
        loop {
            let final_block = deflated[p] & 1 == 1;
            let len = u16::from_le_bytes([deflated[p + 1], deflated[p + 2]]) as usize;
            let nlen = u16::from_le_bytes([deflated[p + 3], deflated[p + 4]]);
            assert_eq!(nlen, !(len as u16), "LEN and NLEN must be complements");
            raw.extend_from_slice(&deflated[p + 5..p + 5 + len]);
            p += 5 + len;
            if final_block {
                break;
            }
        }
        let stored_adler = u32::from_be_bytes([
            idat[idat.len() - 4],
            idat[idat.len() - 3],
            idat[idat.len() - 2],
            idat[idat.len() - 1],
        ]);
        assert_eq!(stored_adler, adler32(&raw), "Adler-32 over the raw bytes");

        // Strip the per-row filter byte and compare to what went in.
        let stride = 1 + w as usize * RGB;
        assert_eq!(raw.len(), stride * h as usize, "one filter byte per row");
        let mut back = Vec::new();
        for row in raw.chunks_exact(stride) {
            assert_eq!(row[0], 0, "filter type 0 = none");
            back.extend_from_slice(&row[1..]);
        }
        assert_eq!(back, pixels, "pixels must survive the round trip");
    }

    /// Dimensions that disagree with the data are refused, not padded.
    #[test]
    fn a_length_that_does_not_match_is_refused() {
        assert!(encode_rgb(2, 2, &[0; 11]).is_none(), "one byte short");
        assert!(encode_rgb(2, 2, &[0; 13]).is_none(), "one byte long");
        assert!(encode_rgb(0, 5, &[]).is_none(), "zero width");
        assert!(encode_rgb(5, 0, &[]).is_none(), "zero height");
        assert!(encode_rgb(2, 2, &[0; 12]).is_some(), "exactly right");
    }

    /// A full-size image is written without a panic and at a sane size.
    ///
    /// 1024×1024 is what the reference command line generates, and stored
    /// deflate means the file is the raw size plus a few hundred bytes — which
    /// this asserts, so switching to real compression later is visible here
    /// rather than silent.
    #[test]
    fn a_1024_square_image_is_the_expected_size() {
        let n = 1024usize * 1024 * RGB;
        let png = encode_rgb(1024, 1024, &vec![128; n]).expect("encode");
        let raw_with_filters = 1024 * (1 + 1024 * RGB);
        assert!(
            png.len() > raw_with_filters,
            "stored deflate cannot be smaller than its input"
        );
        assert!(
            png.len() < raw_with_filters + 1024,
            "overhead should be five bytes per 65535 plus three chunks, got {} over",
            png.len() - raw_with_filters
        );
    }
}
