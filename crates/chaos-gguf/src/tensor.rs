//! `ggml` tensor types and their block sizes.
//!
//! Quantized types are stored in *blocks*: a fixed number of weights share one
//! set of scales, so a tensor's byte size is not `elements * bits / 8` but
//! `elements / block_elems * block_bytes`. Getting this wrong under-reports
//! every quantized tensor, which is exactly the number Chaos exists to get
//! right — so the table is exhaustive rather than approximate.

/// A `ggml` tensor element type, as stored in a GGUF tensor index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlType(pub u32);

impl GgmlType {
    /// `(name, elements_per_block, bytes_per_block)`.
    ///
    /// `None` for types this build does not know — callers must refuse to
    /// guess a size rather than silently mis-report it.
    fn spec(self) -> Option<(&'static str, u64, u64)> {
        // (type id, name, block elements, block bytes)
        let table: &[(u32, &str, u64, u64)] = &[
            (0, "F32", 1, 4),
            (1, "F16", 1, 2),
            (2, "Q4_0", 32, 18),
            (3, "Q4_1", 32, 20),
            (6, "Q5_0", 32, 22),
            (7, "Q5_1", 32, 24),
            (8, "Q8_0", 32, 34),
            (9, "Q8_1", 32, 36),
            (10, "Q2_K", 256, 84),
            (11, "Q3_K", 256, 110),
            (12, "Q4_K", 256, 144),
            (13, "Q5_K", 256, 176),
            (14, "Q6_K", 256, 210),
            (15, "Q8_K", 256, 292),
            (16, "IQ2_XXS", 256, 66),
            (17, "IQ2_XS", 256, 74),
            (18, "IQ3_XXS", 256, 98),
            (19, "IQ1_S", 256, 50),
            (20, "IQ4_NL", 32, 18),
            (21, "IQ3_S", 256, 110),
            (22, "IQ2_S", 256, 82),
            (23, "IQ4_XS", 256, 136),
            (24, "I8", 1, 1),
            (25, "I16", 1, 2),
            (26, "I32", 1, 4),
            (27, "I64", 1, 8),
            (28, "F64", 1, 8),
            (29, "IQ1_M", 256, 56),
            (30, "BF16", 1, 2),
            (34, "TQ1_0", 256, 54),
            (35, "TQ2_0", 256, 66),
            (39, "MXFP4", 32, 17),
        ];
        table
            .iter()
            .find(|(id, ..)| *id == self.0)
            .map(|&(_, name, be, bb)| (name, be, bb))
    }

    pub fn name(self) -> Option<&'static str> {
        self.spec().map(|(n, ..)| n)
    }

    pub fn block_elems(self) -> Option<u64> {
        self.spec().map(|(_, be, _)| be)
    }

    pub fn block_bytes(self) -> Option<u64> {
        self.spec().map(|(.., bb)| bb)
    }

    /// Bytes on disk for `elements` values of this type.
    ///
    /// `None` when the type is unknown, or when `elements` is not a whole
    /// number of blocks — the latter means the file disagrees with the format,
    /// which is worth surfacing rather than rounding away.
    pub fn size_of(self, elements: u64) -> Option<u64> {
        let (_, block_elems, block_bytes) = self.spec()?;
        if block_elems == 0 || elements % block_elems != 0 {
            return None;
        }
        Some(elements / block_elems * block_bytes)
    }

    /// Effective bits per weight, useful for comparing quantizations.
    pub fn bits_per_weight(self) -> Option<f64> {
        let (_, block_elems, block_bytes) = self.spec()?;
        Some(block_bytes as f64 * 8.0 / block_elems as f64)
    }

    pub fn is_quantized(self) -> bool {
        self.block_elems().is_some_and(|b| b > 1)
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.name() {
            Some(n) => f.write_str(n),
            None => write!(f, "type#{}", self.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_block_math_matches_the_format() {
        // 256 weights per block, 144 bytes -> 4.5 bits/weight, not 4.
        let t = GgmlType(12);
        assert_eq!(t.name(), Some("Q4_K"));
        assert_eq!(t.size_of(256), Some(144));
        assert_eq!(t.size_of(2560), Some(1440));
        assert!((t.bits_per_weight().unwrap() - 4.5).abs() < 1e-9);
    }

    #[test]
    fn f32_is_one_element_per_block() {
        assert_eq!(GgmlType(0).size_of(1000), Some(4000));
        assert_eq!(GgmlType(0).bits_per_weight(), Some(32.0));
        assert!(!GgmlType(0).is_quantized());
    }

    #[test]
    fn one_bit_quants_really_are_sub_two_bits() {
        // IQ1_S: the whole reason a 2.8T model can be considered at all.
        let bpw = GgmlType(19).bits_per_weight().unwrap();
        assert!(bpw > 1.5 && bpw < 1.6, "IQ1_S was {bpw}");
    }

    #[test]
    fn partial_block_is_refused_not_rounded() {
        // 100 is not a multiple of 256; silently rounding would mis-size the
        // tensor, so this must fail loudly.
        assert_eq!(GgmlType(12).size_of(100), None);
    }

    #[test]
    fn unknown_type_refuses_to_guess() {
        let unknown = GgmlType(9999);
        assert_eq!(unknown.name(), None);
        assert_eq!(unknown.size_of(256), None);
        assert_eq!(unknown.to_string(), "type#9999");
    }
}
