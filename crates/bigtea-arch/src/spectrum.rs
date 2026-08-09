//! How much of an expert bank's weight actually needs to be read.
//!
//! # The question
//!
//! Streaming an expert costs its full bytes. But an expert's matrix `W` is not
//! arbitrary: every row is a vector in `R^ne0`, and if the rows of *every*
//! expert in a layer happen to lie near a shared `r`-dimensional subspace, then
//! `W_i ≈ C_i B^T` with `B` (`ne0 × r`) **shared across all 256 experts** and
//! only `C_i` (`ne1 × r`) per expert.
//!
//! That is worth `ne0 / r` on bytes read, and the arithmetic runs the right way
//! too: `W_i x = C_i (B^T x)`, where `B^T x` is computed **once per layer** and
//! reused by every selected expert. The factored form is *cheaper* to apply
//! than the dense one, so this is not the usual compute-for-bytes trade.
//!
//! # What this module measures
//!
//! Only one number decides whether any of that is real: **how fast the singular
//! spectrum of the stacked expert bank decays.** This module answers it without
//! running the model, by accumulating
//!
//! ```text
//! G = Σ_i W_i^T W_i        (ne0 × ne0, symmetric, streamable one expert at a time)
//! ```
//!
//! and reporting, for each rank `r`, the fraction of `‖W‖_F²` that the best
//! rank-`r` subspace captures. `trace(G)` is the total.
//!
//! # Why the answer is trustworthy in the direction that matters
//!
//! The top subspace comes from randomised subspace iteration, not an exact
//! eigensolver. That means every captured-energy figure is a **lower bound** on
//! the true optimum: a positive result cannot be an artefact of the method, only
//! an underestimate. A negative result is the one that would need more care, and
//! a negative result kills the idea anyway.
//!
//! Energy is a screen, not a quality claim. Capturing 95% of Frobenius energy
//! does not prove the model still answers correctly — that needs a forward pass
//! against the oracle. It is the cheapest way to find out whether such a test is
//! worth building.

/// A symmetric `n × n` Gram matrix, accumulated over expert slices.
///
/// Row-major. Held in `f32` because the inputs are 4-bit weights dequantised to
/// `f32` — carrying `f64` here would double 67 MB for precision the data does
/// not have. Sums that run over the whole matrix use `f64` accumulators.
pub struct Gram {
    n: usize,
    g: Vec<f32>,
}

impl Gram {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            g: vec![0.0; n * n],
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Add another `n × n` Gram block, as produced for one expert.
    pub fn accumulate(&mut self, block: &[f32]) {
        assert_eq!(block.len(), self.g.len(), "Gram block shape");
        for (a, b) in self.g.iter_mut().zip(block) {
            *a += *b;
        }
    }

    /// `Σ_j ‖row_j‖²` over every row of every expert — the total energy.
    pub fn trace(&self) -> f64 {
        (0..self.n).map(|i| self.g[i * self.n + i] as f64).sum()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.g
    }

    /// `G @ Y` for `Y` of shape `n × k`, row-major.
    ///
    /// Threaded over rows of the output. This is the inner loop of subspace
    /// iteration and, at `n = 4096, k = 1024`, is 17 GFLOP per call — enough
    /// that a single thread is felt and not enough to justify a kernel.
    pub fn mul_block(&self, y: &[f32], k: usize) -> Vec<f32> {
        assert_eq!(y.len(), self.n * k, "Y shape");
        let n = self.n;
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(n);
        let mut out = vec![0.0f32; n * k];

        // Hand each thread whole output rows. `chunks_mut` splits the output so
        // the borrow checker sees disjoint slices; no locking is needed because
        // `G` and `Y` are read-only.
        let rows_per = n.div_ceil(threads);
        std::thread::scope(|s| {
            for (t, chunk) in out.chunks_mut(rows_per * k).enumerate() {
                let g = &self.g;
                s.spawn(move || {
                    let base = t * rows_per;
                    for (local, orow) in chunk.chunks_mut(k).enumerate() {
                        let grow = &g[(base + local) * n..(base + local) * n + n];
                        // i,k,j order: `orow` and `yrow` are both contiguous, so
                        // the inner loop is a scaled add the compiler vectorises.
                        for (p, &gv) in grow.iter().enumerate() {
                            if gv == 0.0 {
                                continue;
                            }
                            let yrow = &y[p * k..p * k + k];
                            for (o, &yv) in orow.iter_mut().zip(yrow) {
                                *o += gv * yv;
                            }
                        }
                    }
                });
            }
        });
        out
    }
}

/// Orthonormalise the `k` columns of an `n × k` row-major matrix, in place.
///
/// Modified Gram-Schmidt, twice. Classical Gram-Schmidt loses orthogonality on
/// exactly the input this produces — subspace iteration drives columns towards
/// each other — and a second pass is the standard, cheap repair.
pub fn orthonormalise(y: &mut [f32], n: usize, k: usize) {
    assert_eq!(y.len(), n * k);
    for _pass in 0..2 {
        for j in 0..k {
            // Project out every earlier column.
            for i in 0..j {
                let mut dot = 0.0f64;
                for r in 0..n {
                    dot += (y[r * k + i] as f64) * (y[r * k + j] as f64);
                }
                let dot = dot as f32;
                if dot != 0.0 {
                    for r in 0..n {
                        y[r * k + j] -= dot * y[r * k + i];
                    }
                }
            }
            let mut norm = 0.0f64;
            for r in 0..n {
                let v = y[r * k + j] as f64;
                norm += v * v;
            }
            let norm = norm.sqrt();
            // A dependent column means the subspace is smaller than `k`; leave
            // it at zero rather than dividing by ~0 and producing noise that
            // would be counted as captured energy.
            let inv = if norm > 1e-12 {
                (1.0 / norm) as f32
            } else {
                0.0
            };
            for r in 0..n {
                y[r * k + j] *= inv;
            }
        }
    }
}

/// Eigenvalues of a small symmetric `k × k` matrix, descending.
///
/// Cyclic one-sided Jacobi. Used only on the `k × k` Rayleigh quotient — never
/// on the full Gram — so `k` is in the hundreds and `O(k³)` per sweep is free.
pub fn jacobi_eigenvalues(a: &mut [f32], k: usize) -> Vec<f64> {
    assert_eq!(a.len(), k * k);
    let mut m: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let off = |m: &[f64]| -> f64 {
        let mut s = 0.0;
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    s += m[i * k + j] * m[i * k + j];
                }
            }
        }
        s
    };
    let scale = (0..k).map(|i| m[i * k + i].abs()).sum::<f64>().max(1e-30);
    for _sweep in 0..60 {
        if off(&m).sqrt() <= 1e-9 * scale {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = m[p * k + q];
                if apq.abs() < 1e-30 {
                    continue;
                }
                let app = m[p * k + p];
                let aqq = m[q * k + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..k {
                    let arp = m[r * k + p];
                    let arq = m[r * k + q];
                    m[r * k + p] = c * arp - s * arq;
                    m[r * k + q] = s * arp + c * arq;
                }
                for r in 0..k {
                    let apr = m[p * k + r];
                    let aqr = m[q * k + r];
                    m[p * k + r] = c * apr - s * aqr;
                    m[q * k + r] = s * apr + c * aqr;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..k).map(|i| m[i * k + i]).collect();
    ev.sort_by(|a, b| b.partial_cmp(a).expect("finite eigenvalue"));
    ev
}

/// What fraction of the bank's energy the best rank-`r` subspace holds.
pub struct EnergyCurve {
    /// Total energy, `trace(G)`.
    pub total: f64,
    /// Eigenvalues of the recovered top subspace, descending — `k` of them.
    pub eigenvalues: Vec<f64>,
}

impl EnergyCurve {
    /// Fraction of energy captured at rank `r`. Ranks past the recovered
    /// subspace are not knowable from this run and return `None`.
    pub fn captured(&self, r: usize) -> Option<f64> {
        if r > self.eigenvalues.len() {
            return None;
        }
        Some(self.eigenvalues[..r].iter().sum::<f64>() / self.total)
    }

    /// The smallest rank capturing at least `want` of the energy.
    pub fn rank_for(&self, want: f64) -> Option<usize> {
        let mut acc = 0.0;
        for (i, e) in self.eigenvalues.iter().enumerate() {
            acc += e;
            if acc / self.total >= want {
                return Some(i + 1);
            }
        }
        None
    }
}

/// Recover the top-`k` eigenspace of `g` and report the energy curve.
///
/// Randomised subspace iteration: sketch, then `power_iters` multiplications by
/// `G` to sharpen towards the dominant subspace, re-orthonormalising each time
/// so the columns do not all collapse onto the leading eigenvector.
///
/// `seed` fixes the sketch, so a run is reproducible and two models can be
/// compared without the random draw being part of the difference.
pub fn energy_curve(g: &Gram, k: usize, power_iters: usize, seed: u64) -> EnergyCurve {
    let n = g.n();
    let k = k.min(n);
    let mut y = gaussian(n * k, seed);
    orthonormalise(&mut y, n, k);
    for _ in 0..power_iters {
        y = g.mul_block(&y, k);
        orthonormalise(&mut y, n, k);
    }

    // Rayleigh quotient R = Qᵀ G Q, then its eigenvalues are the top-k
    // eigenvalues of G restricted to the recovered subspace.
    let gq = g.mul_block(&y, k);
    let mut r = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0f64;
            for p in 0..n {
                s += (y[p * k + i] as f64) * (gq[p * k + j] as f64);
            }
            r[i * k + j] = s as f32;
        }
    }
    // Symmetrise: R is symmetric in exact arithmetic and the Jacobi solver
    // assumes it. Rounding alone can break the assumption.
    for i in 0..k {
        for j in 0..i {
            let m = 0.5 * (r[i * k + j] + r[j * k + i]);
            r[i * k + j] = m;
            r[j * k + i] = m;
        }
    }
    let eigenvalues = jacobi_eigenvalues(&mut r, k);
    EnergyCurve {
        total: g.trace(),
        eigenvalues,
    }
}

/// Deterministic standard normals, via a SplitMix64 stream and Box-Muller.
///
/// Bigtea has no external crates and this is not worth becoming the first one.
/// The sketch only needs to be generic with respect to the eigenbasis; the
/// exact distribution is not load-bearing.
pub fn gaussian(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Open interval: log(0) is not recoverable.
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let (u1, u2) = (next(), next());
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        if out.len() < count {
            out.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `G = MᵀM` for a row-major `rows × n` matrix, the slow honest way.
    fn gram_of(m: &[f32], rows: usize, n: usize) -> Gram {
        let mut g = Gram::zeros(n);
        let mut block = vec![0.0f32; n * n];
        for r in 0..rows {
            let row = &m[r * n..r * n + n];
            for i in 0..n {
                for j in 0..n {
                    block[i * n + j] += row[i] * row[j];
                }
            }
        }
        g.accumulate(&block);
        g
    }

    #[test]
    fn a_bank_that_really_is_low_rank_is_reported_as_low_rank() {
        // 200 rows in R^64, but every row is a combination of 8 basis vectors.
        // This is exactly the structure the factorisation hopes for, so the
        // curve must saturate at 8 and not before.
        let (n, rank, rows) = (64usize, 8usize, 200usize);
        let basis = gaussian(rank * n, 1);
        let coef = gaussian(rows * rank, 2);
        let mut m = vec![0.0f32; rows * n];
        for r in 0..rows {
            for b in 0..rank {
                let c = coef[r * rank + b];
                for i in 0..n {
                    m[r * n + i] += c * basis[b * n + i];
                }
            }
        }
        let g = gram_of(&m, rows, n);
        let curve = energy_curve(&g, 32, 4, 7);

        let at8 = curve.captured(rank).expect("rank 8 is inside k=32");
        assert!(at8 > 0.999, "rank-8 subspace should hold everything: {at8}");
        let at4 = curve.captured(4).expect("inside k");
        assert!(at4 < 0.95, "half the basis cannot hold it all: {at4}");
        assert_eq!(curve.rank_for(0.99), Some(rank));
    }

    #[test]
    fn a_random_bank_is_reported_as_full_rank() {
        // The control the project keeps having to relearn to include: without
        // it, "rank 512 of 4096 holds 40%" reads as a finding when it is what
        // any matrix at all would give.
        let (n, rows) = (64usize, 400usize);
        let m = gaussian(rows * n, 3);
        let g = gram_of(&m, rows, n);
        let curve = energy_curve(&g, 32, 4, 7);

        let at8 = curve.captured(8).expect("inside k");
        // 8/64 = 12.5% by construction; the top of a Marchenko-Pastur spectrum
        // is above its share but nowhere near saturation.
        assert!(
            (0.12..0.30).contains(&at8),
            "random data should be near its uniform share, got {at8}"
        );
        assert_eq!(curve.rank_for(0.99), None, "no small rank captures 99%");
    }

    #[test]
    fn captured_energy_never_exceeds_the_total() {
        let (n, rows) = (48usize, 120usize);
        let m = gaussian(rows * n, 11);
        let g = gram_of(&m, rows, n);
        let curve = energy_curve(&g, 48, 3, 5);
        let full = curve.captured(48).expect("k == n");
        assert!(
            (0.98..=1.02).contains(&full),
            "the whole space must hold ~all the energy, got {full}"
        );
        // Monotone: adding a dimension cannot lose energy.
        let mut prev = 0.0;
        for r in 1..=48 {
            let c = curve.captured(r).expect("inside k");
            assert!(c >= prev - 1e-6, "energy curve dipped at r={r}");
            prev = c;
        }
    }

    #[test]
    fn orthonormalise_produces_an_orthonormal_basis() {
        let (n, k) = (40usize, 6usize);
        let mut y = gaussian(n * k, 21);
        orthonormalise(&mut y, n, k);
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0f64;
                for r in 0..n {
                    dot += (y[r * k + i] as f64) * (y[r * k + j] as f64);
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-4, "Q^T Q [{i},{j}] = {dot}");
            }
        }
    }

    #[test]
    fn jacobi_recovers_known_eigenvalues() {
        // Diagonal matrix conjugated by a rotation: eigenvalues are preserved,
        // so the answer is known exactly and independently of the solver.
        let k = 3;
        let (c, s) = (0.6f32, 0.8f32);
        let d = [5.0f32, 2.0, 1.0];
        let mut a = vec![0.0f32; k * k];
        // Rotate in the (0,1) plane.
        a[0] = c * c * d[0] + s * s * d[1];
        a[1] = c * s * (d[0] - d[1]);
        a[3] = a[1];
        a[4] = s * s * d[0] + c * c * d[1];
        a[8] = d[2];
        let ev = jacobi_eigenvalues(&mut a, k);
        for (got, want) in ev.iter().zip([5.0, 2.0, 1.0]) {
            assert!((got - want).abs() < 1e-4, "{ev:?}");
        }
    }

    #[test]
    fn mul_block_matches_a_single_threaded_multiply() {
        let (n, k) = (37usize, 5usize);
        let mut g = Gram::zeros(n);
        let block = gaussian(n * n, 31);
        // Symmetrise, since `Gram` is only ever fed symmetric blocks.
        let mut sym = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                sym[i * n + j] = 0.5 * (block[i * n + j] + block[j * n + i]);
            }
        }
        g.accumulate(&sym);
        let y = gaussian(n * k, 41);
        let got = g.mul_block(&y, k);
        for i in 0..n {
            for j in 0..k {
                let mut want = 0.0f32;
                for p in 0..n {
                    want += sym[i * n + p] * y[p * k + j];
                }
                assert!((got[i * k + j] - want).abs() < 1e-3, "[{i},{j}]");
            }
        }
    }
}
