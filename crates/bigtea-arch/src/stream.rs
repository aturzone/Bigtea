//! Streaming MoE: keep the dense weights resident, fetch experts on demand.
//!
//! This is the point of the whole project. Qwen3-30B-A3B is 17.28 GiB, but
//! only **0.93 GiB is read every token** — the other 16.35 GiB is routed
//! experts, of which a single token touches 8 of 128, about 1.02 GiB. A
//! machine with 15.7 GiB cannot hold the model, and does not need to.
//!
//! # Why this cannot be one graph
//!
//! `ggml` graphs are declarative: the whole computation is described, then
//! executed. But *which* experts a token needs is decided by a router that
//! runs partway through — so the weights required cannot be known when the
//! graph is built. The forward pass is therefore executed layer by layer:
//!
//! 1. attention and the router for layer *n*
//! 2. read the selected experts off disk
//! 3. the expert feed-forward for layer *n*
//! 4. carry the activations into layer *n+1*
//!
//! The cost is losing cross-layer graph fusion. The benefit is running a model
//! an order of magnitude larger than memory, which is not a trade — it is the
//! difference between running and not running.

use std::collections::HashMap;
use std::sync::Arc;

use bigtea_ggml::{arena_for, Context, RopeParams, Tensor, WeightSet};
use bigtea_model::Model;

use crate::kv::KvCache;
use crate::qwen3::{Qwen3Config, Qwen3Model};
use crate::{ArchError, Result};

const ROPE_TYPE_NEOX: i32 = 2;
const GIB: f64 = (1u64 << 30) as f64;

/// How much work streaming actually did — the numbers that justify it.
#[derive(Debug, Default, Clone)]
pub struct StreamStats {
    pub expert_reads: u64,
    pub expert_bytes: u64,
    pub resident_bytes: u64,
    pub read_seconds: f64,
    /// Slice requests served from memory. With reads, gives the hit rate —
    /// the number that says whether the expert cache is earning its RAM.
    pub cache_hits: u64,
    pub cache_evictions: u64,
    /// Wall time inside the expert feed-forward, excluding the disk reads
    /// already counted in `read_seconds`.
    pub expert_seconds: f64,
    /// Wall time in attention, including building the KV tensors.
    pub attn_seconds: f64,
    /// Wall time spent handing cached bytes to the binder — a copy per slice
    /// per use, which is invisible until it is measured.
    pub copy_seconds: f64,
    /// Wall time copying the KV history into fresh tensors for attention.
    /// Grows with context length times layers, so it is a prime suspect for
    /// why generation slows down as the prompt gets longer.
    pub kv_build_seconds: f64,
    /// Everything else computed per token: embeddings, Q/K/V, the router and
    /// the vocabulary projection.
    pub other_seconds: f64,
}

impl std::fmt::Display for StreamStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "resident {:.2} GiB, streamed {:.2} GiB over {} expert reads in {:.1}s, \
             {} cache hits ({:.0}%), {} evictions",
            self.resident_bytes as f64 / GIB,
            self.expert_bytes as f64 / GIB,
            self.expert_reads,
            self.read_seconds,
            self.cache_hits,
            100.0 * self.cache_hits as f64 / (self.cache_hits + self.expert_reads).max(1) as f64,
            self.cache_evictions
        )?;
        write!(
            f,
            "\n           time: {:.1}s disk, {:.1}s expert compute, {:.1}s attention, \
             {:.1}s slice copies, {:.1}s kv build, {:.1}s other compute",
            self.read_seconds,
            self.expert_seconds,
            self.attn_seconds,
            self.copy_seconds,
            self.kv_build_seconds,
            self.other_seconds
        )
    }
}

/// A forward pass that streams experts instead of holding them.
/// A cached expert slice.
struct CacheEntry {
    bytes: Arc<[u8]>,
}

/// A `Model` pointer that may cross to worker threads.
///
/// # Safety
///
/// `read_tensor_range` takes `&self` and mutates nothing, and `seek_read`
/// carries its own offset, so concurrent reads through one handle are
/// positional and cannot race — the property `ResidentSet::load_parallel`
/// already relies on. The pointer stays valid because the pool lives inside
/// `StreamingRunner`, which borrows the model for `'m` and joins every worker
/// when it drops.
#[derive(Clone, Copy)]
struct ModelPtr(*const Model);
unsafe impl Send for ModelPtr {}
unsafe impl Sync for ModelPtr {}

impl ModelPtr {
    /// Read through the pointer.
    ///
    /// A method rather than a field access at the call site on purpose: under
    /// edition 2021's disjoint closure capture, touching `.0` inside a spawned
    /// closure captures the bare `*const Model` — which is not `Send` — instead
    /// of this wrapper, and the spawn will not compile.
    ///
    /// # Safety
    ///
    /// See [`ModelPtr`]: the model must still be alive, which the pool
    /// guarantees by joining its workers on drop.
    unsafe fn read(&self, name: &str, off: u64, len: u64) -> std::result::Result<Vec<u8>, String> {
        unsafe { (*self.0).read_tensor_range(name, off, len) }.map_err(|e| e.to_string())
    }
}

/// One read request: where to read, and where the answer goes.
type ReadJob = (
    String,
    u64,
    u64,
    usize,
    std::sync::mpsc::Sender<(usize, std::result::Result<Vec<u8>, String>)>,
);

/// Long-lived reader threads.
///
/// Expert reads were previously issued with `std::thread::scope`, which
/// creates and joins threads on every call — 48 layers x 8 experts' worth of
/// misses meant roughly 432 thread spawns per generated token. Measured
/// effect: 13.22 GiB read in 14.4s, about 0.92 GB/s, against the 2.79 GB/s
/// this NVMe delivers in parallel. Per layer that was 9.4ms to fetch ~9 MiB,
/// which is what ten serialized thread creations cost — the spawning *was*
/// the disk time.
struct ReadPool {
    jobs: Option<std::sync::mpsc::Sender<ReadJob>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl ReadPool {
    fn new(model: ModelPtr, threads: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<ReadJob>();
        let rx = Arc::new(std::sync::Mutex::new(rx));
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let rx = Arc::clone(&rx);
            workers.push(std::thread::spawn(move || loop {
                // Hold the lock only long enough to take a job, never across
                // the read itself.
                let job = { rx.lock().expect("read queue").recv() };
                let Ok((name, off, len, idx, done)) = job else {
                    return; // sender dropped: the runner is going away
                };
                // SAFETY: see `ModelPtr` — the model outlives this pool and
                // the read is positional and immutable.
                let got = unsafe { model.read(&name, off, len) };
                let _ = done.send((idx, got));
            }));
        }
        ReadPool {
            jobs: Some(tx),
            workers,
        }
    }

    fn submit(&self, job: ReadJob) {
        if let Some(tx) = &self.jobs {
            let _ = tx.send(job);
        }
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        // Dropping the sender is what tells the workers to stop; they must be
        // joined before the borrowed model can go out of scope.
        self.jobs = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Expert slices keyed by `(tensor name, expert index)`.
///
/// `Arc` rather than `Vec` because the same slice is bound again on every
/// token that routes to it; handing back an owned copy meant memcpying about
/// a gigabyte per token for bytes that never change.
type ExpertSlices = HashMap<(String, u32), Arc<[u8]>>;

pub struct StreamingRunner<'m> {
    model: &'m Model,
    arch: Qwen3Model,
    /// Cache of expert slices already read this session, keyed by
    /// (tensor name, expert index). Bounded, because an unbounded cache would
    /// silently become the thing we set out to avoid.
    cache: HashMap<(String, u32), CacheEntry>,
    cache_budget: usize,
    cache_bytes: usize,
    /// How often each slice has been wanted, whether or not it is cached.
    ///
    /// Kept outside the cache on purpose: an entry's history must survive its
    /// eviction, otherwise a slice that keeps being evicted and re-read looks
    /// permanently new and can never earn its place back.
    freq: HashMap<(String, u32), u32>,
    /// Cached keys, for sampling candidates to evict by index. May hold keys
    /// already gone from `cache`; those are dropped when sampling finds them.
    keys: Vec<(String, u32)>,
    rng: u64,
    /// One arena, reused for every expert graph, instead of a fresh multi-
    /// megabyte allocation per layer per token.
    scratch: Vec<u8>,
    /// Reader threads, created once. Declared after `model` so it drops first.
    pool: ReadPool,
    /// Threads for expert matmuls. Held rather than queried per call because
    /// the expert loop runs it thousands of times per token.
    threads: usize,
    pub stats: StreamStats,
}

impl<'m> StreamingRunner<'m> {
    pub fn new(model: &'m Model, config: Qwen3Config, cache_budget: usize) -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        // Generation runs single-column matmuls, where the work per thread can
        // be smaller than the barrier that synchronises them. Overridable so
        // the trade-off can be measured rather than assumed.
        let _threads = std::env::var("BIGTEA_EXPERT_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(cores);
        StreamingRunner {
            // The reader pool always gets every core: its threads block on
            // disk rather than compete for CPU.
            pool: ReadPool::new(ModelPtr(model as *const Model), cores),
            model,
            arch: Qwen3Model::new(config),
            cache: HashMap::new(),
            cache_budget,
            cache_bytes: 0,
            freq: HashMap::new(),
            keys: Vec::new(),
            rng: 0x9E3779B97F4A7C15,
            scratch: Vec::new(),
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            stats: StreamStats::default(),
        }
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.arch.config
    }

    /// Tensors that stay in RAM: everything except routed experts.
    pub fn resident_tensor_names(&self) -> Vec<String> {
        self.arch
            .required_tensors()
            .into_iter()
            .filter(|n| !n.contains("_exps"))
            .collect()
    }

    /// Read one expert's slice out of a stacked expert tensor.
    ///
    /// The stack is `[ne0, ne1, n_expert]`, so expert `idx` is a contiguous
    /// run at `idx * slice_bytes`. Reading only that run is what keeps a
    /// token's cost at 1 GiB instead of 16.
    /// One slice, through the same path as a batch — so cache accounting,
    /// eviction and statistics have a single implementation rather than two
    /// that can drift apart.
    fn expert_slice(&mut self, name: &str, idx: u32) -> Result<Arc<[u8]>> {
        let key = (name.to_string(), idx);
        let mut got = self.read_slices_parallel(std::slice::from_ref(&key))?;
        got.remove(&key)
            .ok_or_else(|| ArchError::MissingTensor(format!("{name}[{idx}]")))
    }

    /// Which experts this token routes to, and their weights.
    ///
    /// Runs the router on the CPU rather than in the graph, because the result
    /// determines what to read next — the graph cannot branch on it.
    fn route(&self, probs: &[f32], n_expert_used: usize) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(n_expert_used);

        // Renormalise over the selected experts only, which is what Qwen3
        // does; leaving them unnormalised scales every expert output down by
        // the mass that went to the experts we did not run.
        let total: f32 = scored.iter().map(|(_, p)| *p).sum();
        if total > 0.0 {
            for (_, p) in &mut scored {
                *p /= total;
            }
        }
        scored
    }

    /// Apply one layer's expert feed-forward to a whole block of tokens.
    ///
    /// The per-token version reads an expert's three tensors every time a token
    /// selects it. Across a 256-token block that is 256 * 8 * 3 = 6,144 reads
    /// per layer, against 384 distinct slices that actually exist — the same
    /// bytes fetched from disk sixteen times over. Measured on a 565-token
    /// prompt: 609,665 reads, 537 GiB streamed, 470s of prefill.
    ///
    /// So invert the loop. Group the block's tokens by the expert they routed
    /// to, read each distinct expert once, and run it against all of its tokens
    /// as a single matrix. Experts are visited in ascending index order, which
    /// also makes the reads walk the file forwards instead of seeking.
    ///
    /// `normed` holds the block's post-FFN-norm activations, `n_embd` floats
    /// per token; `probs` holds `n_expert` router probabilities per token.
    fn expert_ffn_block(
        &mut self,
        normed: &[f32],
        il: u32,
        probs: &[f32],
        n_tokens: usize,
    ) -> Result<Vec<f32>> {
        use std::collections::BTreeMap;

        let c = self.arch.config.clone();
        let n_embd = c.n_embd as usize;
        let n_expert = c.n_expert as usize;

        // expert -> the tokens that chose it, with their routing weights.
        // BTreeMap keeps the read order ascending by expert index.
        let mut by_expert: BTreeMap<u32, Vec<(usize, f32)>> = BTreeMap::new();
        for t in 0..n_tokens {
            let picks = self.route(
                &probs[t * n_expert..(t + 1) * n_expert],
                c.n_expert_used as usize,
            );
            for (expert, weight) in picks {
                by_expert.entry(expert).or_default().push((t, weight));
            }
        }

        let gate_name = format!("blk.{il}.ffn_gate_exps.weight");
        let up_name = format!("blk.{il}.ffn_up_exps.weight");
        let down_name = format!("blk.{il}.ffn_down_exps.weight");
        let gate_ty = self.tensor_type(&gate_name)?;
        let up_ty = self.tensor_type(&up_name)?;
        let down_ty = self.tensor_type(&down_name)?;
        let n_embd_i = c.n_embd as i64;
        let n_ff = c.n_ff_expert as i64;

        let mut accum = vec![0f32; n_embd * n_tokens];

        // Generating a token is one position, so every expert matmul here is a
        // single column: a few microseconds of arithmetic wrapped in a graph
        // allocation and a 12-thread barrier. Per token that was 48 layers x 8
        // experts x (gate, up, down) = 1,152 compute calls, and the overhead of
        // each dwarfed the work inside it.
        //
        // With one position the experts' outputs are all n_embd x 1, so they
        // can be scaled by their routing weights and summed inside a single
        // graph — one allocation and one barrier per layer instead of 24.
        if n_tokens == 1 {
            return self.expert_ffn_single(normed, &by_expert, &gate_name, &up_name, &down_name);
        }

        // Read ahead in groups. A single synchronous read cannot saturate this
        // NVMe — measured 1.26 GB/s against 2.79 GB/s across 16 threads — and
        // the expert loop is almost entirely waiting on disk. Groups bound the
        // memory this costs: 16 experts is ~43 MiB in flight, against ~340 MiB
        // for a whole layer's experts.
        const READ_GROUP: usize = 16;
        let order: Vec<u32> = by_expert.keys().copied().collect();

        for group in order.chunks(READ_GROUP) {
            let mut wanted: Vec<(String, u32)> = Vec::with_capacity(group.len() * 3);
            for &e in group {
                wanted.push((gate_name.clone(), e));
                wanted.push((up_name.clone(), e));
                wanted.push((down_name.clone(), e));
            }
            let fetched = self.read_slices_parallel(&wanted)?;

            // One arena for the whole group. Prefilling a 4395-token prompt
            // otherwise allocates and first-touches a fresh multi-megabyte
            // arena 55,296 times — 128 experts x 48 layers x 9 blocks.
            let mut buf = std::mem::take(&mut self.scratch);
            let threads = self.threads;
            let mut group_secs = 0f64;
            let group_result = (|| -> Result<()> {
                for &expert in group {
                    let members = &by_expert[&expert];
                    let take = |n: &String| -> Result<Arc<[u8]>> {
                        fetched
                            .get(&(n.clone(), expert))
                            .cloned()
                            .ok_or_else(|| ArchError::MissingTensor(format!("{n}[{expert}]")))
                    };
                    let gate_bytes = take(&gate_name)?;
                    let up_bytes = take(&up_name)?;
                    let down_bytes = take(&down_name)?;

                    let m = members.len() as i64;
                    let need = arena_for(
                        &[
                            (n_embd_i, m), // this expert's tokens, gathered
                            (n_ff, m),     // gate
                            (n_ff, m),     // up
                            (n_ff, m),     // silu(gate) * up
                            (n_embd_i, m), // down projection
                        ],
                        24,
                    ) + gate_bytes.len()
                        + up_bytes.len()
                        + down_bytes.len();
                    if buf.len() < need {
                        buf.resize(need, 0);
                    }
                    // SAFETY: `buf` is a local outliving `ctx`; each expert's context
                    // is dropped and its output copied out before the next is built.
                    let ctx = unsafe { Context::in_buffer(&mut buf, false)? };
                    let mut ws = WeightSet::new();
                    ws.bind(
                        &ctx,
                        "gate",
                        gate_ty,
                        &[n_embd_i as u64, n_ff as u64],
                        gate_bytes,
                    )?;
                    ws.bind(&ctx, "up", up_ty, &[n_embd_i as u64, n_ff as u64], up_bytes)?;
                    ws.bind(
                        &ctx,
                        "down",
                        down_ty,
                        &[n_ff as u64, n_embd_i as u64],
                        down_bytes,
                    )?;

                    // Gather this expert's tokens into one contiguous matrix so the
                    // three matmuls run once for the group rather than once per token.
                    let mut gathered = vec![0f32; n_embd * members.len()];
                    for (slot, (t, _)) in members.iter().enumerate() {
                        gathered[slot * n_embd..(slot + 1) * n_embd]
                            .copy_from_slice(&normed[t * n_embd..(t + 1) * n_embd]);
                    }
                    let xt = ctx.new_f32_2d(n_embd_i, m)?;
                    xt.set_f32(&gathered)?;

                    let g = ctx.mul_mat(ws.get("gate").expect("bound"), &xt)?;
                    let u = ctx.mul_mat(ws.get("up").expect("bound"), &xt)?;
                    let act = ctx.mul(&ctx.silu(&g)?, &u)?;
                    let out = ctx.mul_mat(ws.get("down").expect("bound"), &act)?;
                    // Not 0: `compute` floors the count at 1, so passing 0 ran every
                    // expert matmul on a single thread — the bulk of the model's
                    // arithmetic, on one core of twelve.
                    let t_exp = std::time::Instant::now();
                    ctx.compute(&out, threads)?;
                    group_secs += t_exp.elapsed().as_secs_f64();

                    let produced = out.to_vec_f32();
                    for (slot, (t, weight)) in members.iter().enumerate() {
                        let src = &produced[slot * n_embd..(slot + 1) * n_embd];
                        let dst = &mut accum[t * n_embd..(t + 1) * n_embd];
                        for (d, v) in dst.iter_mut().zip(src) {
                            *d += v * weight;
                        }
                    }
                }
                Ok(())
            })();
            self.scratch = buf;
            group_result?;
            self.stats.expert_seconds += group_secs;
        }
        Ok(accum)
    }

    /// One layer's experts for a single position, in one graph.
    ///
    /// Every expert contributes `n_embd x 1`, so the routing weights can be
    /// applied with `scale` and the results added together as part of the same
    /// graph. That leaves one arena and one `compute` per layer instead of one
    /// per expert matmul.
    fn expert_ffn_single(
        &mut self,
        normed: &[f32],
        by_expert: &std::collections::BTreeMap<u32, Vec<(usize, f32)>>,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
    ) -> Result<Vec<f32>> {
        let c = self.arch.config.clone();
        let n_embd = c.n_embd as i64;
        let n_ff = c.n_ff_expert as i64;
        let gate_ty = self.tensor_type(gate_name)?;
        let up_ty = self.tensor_type(up_name)?;
        let down_ty = self.tensor_type(down_name)?;

        // Fetch every expert this position needs in one parallel batch.
        let mut wanted: Vec<(String, u32)> = Vec::with_capacity(by_expert.len() * 3);
        for &e in by_expert.keys() {
            wanted.push((gate_name.to_string(), e));
            wanted.push((up_name.to_string(), e));
            wanted.push((down_name.to_string(), e));
        }
        let fetched = self.read_slices_parallel(&wanted)?;

        let n_exp = by_expert.len() as i64;
        // Binding a weight allocates its full size in the arena before the data
        // pointer is replaced, so every expert's three quantized tensors have to
        // be paid for here. Eight experts is ~21 MiB, which overran the 16 MiB
        // graph reserve when only the intermediates were counted — and ggml
        // aborts rather than reporting it.
        let weight_bytes: usize = fetched.values().map(|v| v.len()).sum();
        let need = arena_for(
            &[
                (n_embd, 1),
                (n_ff, n_exp * 3),   // gate, up and their product, per expert
                (n_embd, n_exp * 3), // down output, scaled, and the sums
            ],
            16 * by_expert.len() + 32,
        ) + weight_bytes;

        // Borrow the scratch arena out of `self` so the context can hold it
        // mutably while statistics are still being updated.
        let mut buf = std::mem::take(&mut self.scratch);
        if buf.len() < need {
            buf.resize(need, 0);
        }
        let threads = self.threads;
        let result = (|| -> Result<(Vec<f32>, f64)> {
            // SAFETY: `buf` is a local that outlives `ctx`, and no other
            // context is live on it — the previous one was dropped and its
            // results copied out before this call.
            let ctx = unsafe { Context::in_buffer(&mut buf, false)? };
            let mut ws = WeightSet::new();
            let xt = ctx.new_f32_2d(n_embd, 1)?;
            xt.set_f32(&normed[..n_embd as usize])?;

            let mut total: Option<Tensor> = None;
            for (&expert, members) in by_expert {
                let weight = members[0].1;
                let take = |n: &str| -> Result<Arc<[u8]>> {
                    fetched
                        .get(&(n.to_string(), expert))
                        .cloned()
                        .ok_or_else(|| ArchError::MissingTensor(format!("{n}[{expert}]")))
                };
                // Names must be unique per expert: one WeightSet holds them all.
                let (gk, uk, dk) = (
                    format!("g{expert}"),
                    format!("u{expert}"),
                    format!("d{expert}"),
                );
                ws.bind(
                    &ctx,
                    &gk,
                    gate_ty,
                    &[n_embd as u64, n_ff as u64],
                    take(gate_name)?,
                )?;
                ws.bind(
                    &ctx,
                    &uk,
                    up_ty,
                    &[n_embd as u64, n_ff as u64],
                    take(up_name)?,
                )?;
                ws.bind(
                    &ctx,
                    &dk,
                    down_ty,
                    &[n_ff as u64, n_embd as u64],
                    take(down_name)?,
                )?;

                let g = ctx.mul_mat(ws.get(&gk).expect("bound"), &xt)?;
                let u = ctx.mul_mat(ws.get(&uk).expect("bound"), &xt)?;
                let act = ctx.mul(&ctx.silu(&g)?, &u)?;
                let out = ctx.mul_mat(ws.get(&dk).expect("bound"), &act)?;
                let scaled = ctx.scale(&out, weight)?;
                total = Some(match total {
                    None => scaled,
                    Some(t) => ctx.add(&t, &scaled)?,
                });
            }

            let Some(total) = total else {
                return Ok((vec![0f32; n_embd as usize], 0.0));
            };
            let t = std::time::Instant::now();
            ctx.compute(&total, threads)?;
            let secs = t.elapsed().as_secs_f64();
            Ok((total.to_vec_f32(), secs))
        })();

        self.scratch = buf;
        let (out, secs) = result?;
        self.stats.expert_seconds += secs;
        Ok(out)
    }

    /// Put a freshly read slice in the cache if it deserves the space.
    ///
    /// The access pattern here is a *cyclic scan*: every block walks layers 0
    /// to 47 and reads most experts in each. When such a cycle is larger than
    /// the cache — 16.35 GiB of experts against a 6.26 GiB budget — recency is
    /// precisely the wrong signal. Layer 0's slices are always the oldest thing
    /// present when layer 47 needs room, so they are evicted just before the
    /// next block asks for them again. Measured: a 6.26 GiB LRU-ish cache
    /// returned a **17% hit rate with 20,975 evictions**, worse than the ~38%
    /// that pinning an arbitrary fixed third would have given for free.
    ///
    /// So admission is by frequency and nothing else. A newcomer must be wanted
    /// strictly more often than the entry it would displace, which stops the
    /// churn and lets the cache settle on genuinely hot experts. Routing is
    /// skewed enough for that to beat a fixed subset.
    fn admit(&mut self, key: (String, u32), bytes: Arc<[u8]>) {
        const SAMPLE: usize = 8;
        if bytes.len() > self.cache_budget {
            return;
        }
        let mine = self.freq.get(&key).copied().unwrap_or(1);

        while self.cache_bytes + bytes.len() > self.cache_budget {
            let Some(victim) = self.weakest(SAMPLE) else {
                return;
            };
            let theirs = self.freq.get(&victim).copied().unwrap_or(1);
            if mine <= theirs {
                return; // the incumbent is wanted at least as often; leave it
            }
            if let Some(entry) = self.cache.remove(&victim) {
                self.cache_bytes -= entry.bytes.len();
                self.stats.cache_evictions += 1;
            } else {
                return;
            }
        }
        self.cache_bytes += bytes.len();
        self.keys.push(key.clone());
        self.cache.insert(key, CacheEntry { bytes });
    }

    /// The least valuable of a small random sample: fewest uses, then oldest.
    ///
    /// Sampling reads through `keys` by index. Iterating the HashMap with
    /// `step_by` would have looked equivalent and been O(n) — `step_by` still
    /// calls `next` for every element it skips — which at one eviction per miss
    /// is a full scan of the cache thousands of times per token.
    fn weakest(&mut self, sample: usize) -> Option<(String, u32)> {
        let mut best: Option<((String, u32), u32)> = None;
        let mut tries = 0;
        while tries < sample * 2 && !self.keys.is_empty() {
            tries += 1;
            let i = (self.next_rand() as usize) % self.keys.len();
            let key = self.keys[i].clone();
            if !self.cache.contains_key(&key) {
                self.keys.swap_remove(i); // stale: evicted earlier
                continue;
            }
            let uses = self.freq.get(&key).copied().unwrap_or(1);
            if best.as_ref().is_none_or(|(_, b)| uses < *b) {
                best = Some((key, uses));
            }
        }
        best.map(|(k, _)| k)
    }

    /// xorshift64: enough randomness to spread the sample, no dependency.
    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    /// Read several expert slices at once, returning cache hits and fresh
    /// reads together.
    ///
    /// The expert loop spends most of its time blocked on disk, and one
    /// synchronous read leaves most of the device idle: 1.26 GB/s measured
    /// single-threaded against 2.79 GB/s across 16 threads on this NVMe.
    /// `seek_read` carries its own offset, so concurrent reads through one
    /// handle are positional and do not race — the same property
    /// `ResidentSet::load_parallel` already relies on.
    fn read_slices_parallel(&mut self, wanted: &[(String, u32)]) -> Result<ExpertSlices> {
        let mut out: ExpertSlices = HashMap::with_capacity(wanted.len());
        let mut misses: Vec<(String, u32, u64, u64)> = Vec::new();

        for key in wanted {
            if out.contains_key(key) {
                continue; // the same slice asked for twice in one group
            }
            *self.freq.entry(key.clone()).or_insert(0) += 1;
            if let Some(hit) = self.cache.get(key) {
                self.stats.cache_hits += 1;
                let t = std::time::Instant::now();
                let copy = hit.bytes.clone();
                self.stats.copy_seconds += t.elapsed().as_secs_f64();
                out.insert(key.clone(), copy);
                continue;
            }
            let (name, idx) = key;
            let loc = self
                .model
                .location(name)
                .ok_or_else(|| ArchError::MissingTensor(name.to_string()))?;
            let n_expert = *loc.dims.last().unwrap_or(&1);
            if n_expert == 0 || *idx as u64 >= n_expert {
                return Err(ArchError::MissingTensor(format!(
                    "{name}: expert {idx} of {n_expert}"
                )));
            }
            let slice_bytes = loc.size / n_expert;
            misses.push((name.clone(), *idx, *idx as u64 * slice_bytes, slice_bytes));
        }

        if !misses.is_empty() {
            let start = std::time::Instant::now();

            // Hand every miss to the standing pool at once, then wait. The
            // reads overlap without any thread being created here.
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            for (i, (name, _idx, off, len)) in misses.iter().enumerate() {
                self.pool
                    .submit((name.clone(), *off, *len, i, done_tx.clone()));
            }
            drop(done_tx);

            let mut fetched: Vec<Option<Vec<u8>>> = (0..misses.len()).map(|_| None).collect();
            let mut failure: Option<String> = None;
            for _ in 0..misses.len() {
                match done_rx.recv() {
                    Ok((i, Ok(bytes))) => fetched[i] = Some(bytes),
                    Ok((_, Err(e))) => failure = Some(e),
                    Err(_) => {
                        failure = Some("read pool stopped before finishing".into());
                        break;
                    }
                }
            }
            self.stats.read_seconds += start.elapsed().as_secs_f64();

            if let Some(e) = failure {
                return Err(ArchError::MissingTensor(format!("expert read failed: {e}")));
            }
            for (i, (name, idx, _, _)) in misses.iter().enumerate() {
                let bytes: Arc<[u8]> = Arc::from(
                    fetched[i]
                        .take()
                        .ok_or_else(|| ArchError::MissingTensor(format!("{name}[{idx}]")))?,
                );
                let key = (name.clone(), *idx);
                self.stats.expert_reads += 1;
                self.stats.expert_bytes += bytes.len() as u64;
                self.admit(key.clone(), bytes.clone());
                out.insert(key, bytes);
            }
        }
        Ok(out)
    }

    /// Apply one layer's expert feed-forward for a single token.
    ///
    /// `x` is that token's activations after the FFN norm.
    #[allow(dead_code)]
    fn expert_ffn(&mut self, x: &[f32], il: u32, probs: &[f32]) -> Result<Vec<f32>> {
        let c = self.arch.config.clone();
        let picks = self.route(probs, c.n_expert_used as usize);

        let gate_name = format!("blk.{il}.ffn_gate_exps.weight");
        let up_name = format!("blk.{il}.ffn_up_exps.weight");
        let down_name = format!("blk.{il}.ffn_down_exps.weight");

        let mut accum = vec![0f32; c.n_embd as usize];

        for (expert, weight) in picks {
            let gate_bytes = self.expert_slice(&gate_name, expert)?;
            let up_bytes = self.expert_slice(&up_name, expert)?;
            let down_bytes = self.expert_slice(&down_name, expert)?;

            let gate_ty = self.tensor_type(&gate_name)?;
            let up_ty = self.tensor_type(&up_name)?;
            let down_ty = self.tensor_type(&down_name)?;

            // A small arena: one expert's worth of work, not the model's.
            let ctx = Context::new(256 << 20)?;
            let mut ws = WeightSet::new();
            let n_embd = c.n_embd as i64;
            let n_ff = c.n_ff_expert as i64;

            ws.bind(
                &ctx,
                "gate",
                gate_ty,
                &[n_embd as u64, n_ff as u64],
                gate_bytes,
            )?;
            ws.bind(&ctx, "up", up_ty, &[n_embd as u64, n_ff as u64], up_bytes)?;
            ws.bind(
                &ctx,
                "down",
                down_ty,
                &[n_ff as u64, n_embd as u64],
                down_bytes,
            )?;

            let xt = ctx.new_f32_2d(n_embd, 1)?;
            xt.set_f32(x)?;

            let g = ctx.mul_mat(ws.get("gate").expect("bound"), &xt)?;
            let u = ctx.mul_mat(ws.get("up").expect("bound"), &xt)?;
            let act = ctx.mul(&ctx.silu(&g)?, &u)?;
            let out = ctx.mul_mat(ws.get("down").expect("bound"), &act)?;
            ctx.compute(&out, self.threads)?;

            for (dst, v) in accum.iter_mut().zip(out.to_vec_f32()) {
                *dst += v * weight;
            }
        }
        Ok(accum)
    }

    fn tensor_type(&self, name: &str) -> Result<bigtea_gguf::GgmlType> {
        self.model
            .location(name)
            .map(|l| l.ty)
            .ok_or_else(|| ArchError::MissingTensor(name.to_string()))
    }

    /// RoPE parameters this architecture uses.
    pub fn rope(&self) -> RopeParams {
        RopeParams {
            freq_base: self.arch.config.rope_freq_base,
            ..RopeParams::default()
        }
    }

    pub fn rope_type() -> i32 {
        ROPE_TYPE_NEOX
    }

    /// Bind every resident tensor into `ctx`, reporting what it cost.
    pub fn load_resident<'a>(
        &mut self,
        ctx: &'a Context,
        weights: &mut WeightSet<'a>,
    ) -> Result<u64> {
        let mut total = 0u64;
        for name in self.resident_tensor_names() {
            let loc = self
                .model
                .location(&name)
                .ok_or_else(|| ArchError::MissingTensor(name.clone()))?
                .clone();
            let data = self.model.read_tensor(&name)?;
            total += data.len() as u64;
            weights.bind(ctx, &name, loc.ty, &loc.dims, data)?;
        }
        // The output projection may be tied to the embeddings.
        if self.model.location("output.weight").is_some() {
            let loc = self
                .model
                .location("output.weight")
                .expect("checked")
                .clone();
            let data = self.model.read_tensor("output.weight")?;
            total += data.len() as u64;
            weights.bind(ctx, "output.weight", loc.ty, &loc.dims, data)?;
        }
        self.stats.resident_bytes = total;
        Ok(total)
    }

    /// Expose the expert FFN for the runner's layer loop.
    pub fn run_expert_ffn(&mut self, x: &[f32], il: u32, router_probs: &[f32]) -> Result<Vec<f32>> {
        self.expert_ffn(x, il, router_probs)
    }

    /// One post-norm applied to a whole activation buffer.
    ///
    /// Gemma normalises the attention output and the FFN output *before* each
    /// rejoins the residual stream. Both arrive here as plain `Vec<f32>`
    /// because the streaming path materialises between phases.
    fn post_norm<'a>(
        &self,
        weights: &WeightSet<'a>,
        il: u32,
        values: &[f32],
        n_new: i64,
        suffix: &str,
    ) -> Result<Vec<f32>> {
        let n_embd = self.arch.config.n_embd as i64;
        let name = format!("blk.{il}.{suffix}.weight");
        let w = weights
            .get(&name)
            .copied()
            .ok_or(ArchError::MissingTensor(name))?;
        let ctx = Context::new(arena_for(&[(n_embd, n_new)], 12))?;
        let t = ctx.new_f32_2d(n_embd, n_new)?;
        t.set_f32(values)?;
        let out = self.arch.norm_scaled(&ctx, &t, &w)?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        ctx.compute(&out, threads)?;
        Ok(out.to_vec_f32())
    }

    /// Incremental forward pass: computes only the new positions, attending
    /// over cached history.
    ///
    /// This is what makes generation linear. The uncached path recomputes
    /// every previous position for every token, so a response costs O(n²) —
    /// measured at 31,032 expert reads for 5 tokens, against ~1,152 needed
    /// for one position. Here each token's experts are read once.
    ///
    /// `pos_start` is the absolute position of the first token in `tokens`;
    /// RoPE and the causal mask both depend on it, and an off-by-one degrades
    /// output subtly rather than visibly.
    pub fn forward_cached<'a>(
        &mut self,
        weights: &WeightSet<'a>,
        cache: &mut KvCache,
        tokens: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>> {
        let c = self.arch.config.clone();
        let n_new = tokens.len() as i64;
        let n_embd = c.n_embd as i64;
        let head_dim = c.head_dim as i64;
        let n_kv = c.n_head_kv as i64;
        let kv_width = (n_kv * head_dim) as usize;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let positions: Vec<i32> = (0..n_new).map(|i| (pos_start as i64 + i) as i32).collect();

        // The causal mask depends only on positions, so it is the same for all
        // 48 layers. Build it once. Masked entries must be -inf, not 0 — a zero
        // there lets a token attend to its own future and the model produces
        // fluent repetition rather than an error.
        let n_total_final = (cache.len() + tokens.len()) as i64;
        // F16 because ggml's fused attention asserts that type. The only two
        // values are 0 and -inf, so the bit patterns go in directly — no
        // conversion, and -inf stays exactly -inf.
        const F16_ZERO: [u8; 2] = [0x00, 0x00];
        const F16_NEG_INF: [u8; 2] = [0x00, 0xFC];
        let mask: Vec<u8> = {
            let mut m = vec![0u8; (n_total_final * n_new) as usize * 2];
            for query in 0..n_new {
                let absolute = pos_start as i64 + query;
                let row = (query * n_total_final) as usize * 2;
                for key in (absolute + 1)..n_total_final {
                    let at = row + key as usize * 2;
                    m[at..at + 2].copy_from_slice(&F16_NEG_INF);
                }
            }
            let _ = F16_ZERO; // the default fill is already the zero pattern
            m
        };

        let mut x: Vec<f32> = {
            // Sized from the prompt, not a constant: `get_rows` materialises
            // n_embd * n_new floats, and ggml aborts the process when an arena
            // runs out rather than returning an error we could catch.
            let ctx = Context::new(arena_for(&[(n_embd, n_new)], 8))?;
            let tok = ctx.new_i32_1d(n_new)?;
            tok.set_i32(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>())?;
            let emb = weights
                .get("token_embd.weight")
                .ok_or_else(|| ArchError::MissingTensor("token_embd.weight".into()))?;
            let rows = ctx.get_rows(emb, &tok)?;
            // Gemma multiplies the embeddings by sqrt(n_embd) on the way in.
            // Skipping it does not fail -- every activation downstream is just
            // the wrong magnitude, and the model answers fluently and wrongly.
            let rows = if c.scale_embeddings {
                ctx.scale(&rows, (c.n_embd as f32).sqrt())?
            } else {
                rows
            };
            ctx.compute(&rows, threads)?;
            rows.to_vec_f32()
        };

        for il in 0..c.n_layer {
            let get = |w: &WeightSet<'a>, n: String| -> Result<Tensor<'a>> {
                w.get(&n).copied().ok_or(ArchError::MissingTensor(n))
            };

            // Phase 1: Q, K and V for the new positions only.
            let (q_v, k_v, v_v, residual) = {
                // Was a fixed 256 MiB, which aborted at a 4096-token block:
                // ggml asked for 318,787,536 bytes and got told 268,435,456.
                // Every arena in this function has to scale with the block.
                let ctx = Context::new(arena_for(
                    &[
                        (n_embd, n_new),                     // input activations
                        (n_embd, n_new),                     // normalised
                        (n_embd, n_new),                     // rms intermediate
                        (c.n_head as i64 * head_dim, n_new), // q
                        (n_kv * head_dim, n_new),            // k
                        (n_kv * head_dim, n_new),            // v
                        (c.n_head as i64 * head_dim, n_new), // q normalised
                        (n_kv * head_dim, n_new),            // k normalised
                        (c.n_head as i64 * head_dim, n_new), // q roped
                        (n_kv * head_dim, n_new),            // k roped
                    ],
                    32,
                ))?;
                let xt = ctx.new_f32_2d(n_embd, n_new)?;
                xt.set_f32(&x)?;
                let pos = ctx.new_i32_1d(n_new)?;
                pos.set_i32(&positions)?;

                let normed = self.arch.norm_scaled(
                    &ctx,
                    &xt,
                    &get(weights, format!("blk.{il}.attn_norm.weight"))?,
                )?;
                let (qw, kw, vw) = self.arch.qkv_weights(&ctx, weights, il)?;
                let q = ctx.mul_mat(&qw, &normed)?;
                let k = ctx.mul_mat(&kw, &normed)?;
                let v = ctx.mul_mat(&vw, &normed)?;

                let q = ctx.reshape_3d(&q, head_dim, c.n_head as i64, n_new)?;
                let k = ctx.reshape_3d(&k, head_dim, n_kv, n_new)?;
                // Qwen3 normalises each head before RoPE; llama, mistral,
                // qwen2, gemma and phi have no such tensors at all, so asking
                // for them refuses those containers outright.
                let (q, k) = if c.qk_norm {
                    (
                        self.arch.norm_scaled(
                            &ctx,
                            &q,
                            &get(weights, format!("blk.{il}.attn_q_norm.weight"))?,
                        )?,
                        self.arch.norm_scaled(
                            &ctx,
                            &k,
                            &get(weights, format!("blk.{il}.attn_k_norm.weight"))?,
                        )?,
                    )
                } else {
                    (q, k)
                };

                let rp = self.rope();
                // NORM for llama/mistral, NeoX for qwen/phi/gemma. Both run
                // without error on either layout and the wrong one is fluent
                // nonsense, so it comes from the config rather than a constant.
                let rope_type = c.rope_type;
                let q = ctx.rope_ext(&q, &pos, None, head_dim as i32, rope_type, 0, rp)?;
                let k = ctx.rope_ext(&k, &pos, None, head_dim as i32, rope_type, 0, rp)?;

                // One compute materialises all three; they share a graph.
                let t = std::time::Instant::now();
                ctx.compute(&q, threads)?;
                ctx.compute(&k, threads)?;
                ctx.compute(&v, threads)?;
                self.stats.other_seconds += t.elapsed().as_secs_f64();
                (q.to_vec_f32(), k.to_vec_f32(), v.to_vec_f32(), x.clone())
            };

            // K and V for these positions never change again, so store them.
            for t in 0..tokens.len() {
                let lo = t * kv_width;
                cache.push(
                    il as usize,
                    &k_v[lo..lo + kv_width],
                    &v_v[lo..lo + kv_width],
                )?;
            }

            // Phase 2: attend over the whole history, not just the new part.
            let n_total = (cache.len() + tokens.len()) as i64;
            // The scores and their softmax dominate: n_total * n_new * n_head
            // floats each. At 4395 tokens with a 512-token block that pair is
            // ~576 MiB, so the arena runs past a gigabyte — and it was being
            // allocated and first-touched afresh for every layer of every
            // block, 432 times over a single prompt. Reuse one buffer.
            // The fused kernel never builds the scores or their softmax, which
            // were 288 MiB each at 4395 tokens and dominated this arena. What
            // remains is Q, K, V, the mask and the output — about 100 MiB where
            // the explicit path needed 1.3 GiB.
            let n_head = c.n_head as i64;
            let need = arena_for(
                &[
                    (head_dim * n_head, n_new), // q, contiguous
                    (head_dim * n_kv, n_total), // k, contiguous
                    (head_dim * n_kv, n_total), // v, contiguous (not transposed)
                    (n_total, n_new),           // causal mask (F16, so over-counted)
                    (head_dim * n_new, n_head), // attention output
                    (head_dim * n_new, n_head), // ...made contiguous
                    (n_embd, n_new),            // output projection
                ],
                24,
            );
            let mut buf = std::mem::take(&mut self.scratch);
            if buf.len() < need {
                buf.resize(need, 0);
            }
            let arch = &self.arch;
            let out_w = get(weights, format!("blk.{il}.attn_output.weight"))?;
            let attn_result = (|| -> Result<(Vec<f32>, f64, f64)> {
                // SAFETY: `buf` is a local outliving `ctx`, and no other context
                // is live on it — the Q/K/V context above was dropped and its
                // results copied out before this point.
                let ctx = unsafe { Context::in_buffer(&mut buf, false)? };
                let q = ctx.new_f32_3d(head_dim, n_head, n_new)?;
                q.set_f32(&q_v)?;

                // F16, matching how the cache stores them and what the fused
                // kernel wants — no conversion on this path at all.
                let tkv = std::time::Instant::now();
                let k_all = ctx.new_f16_3d(head_dim, n_kv, n_total)?;
                k_all.set_bytes(cache.keys(il as usize))?;
                let v_all = ctx.new_f16_3d(head_dim, n_kv, n_total)?;
                v_all.set_bytes(cache.values(il as usize))?;
                let kv_secs = tkv.elapsed().as_secs_f64();

                let out = arch.attention_flash(&ctx, &q, &k_all, &v_all, n_new, n_total, &mask)?;
                let out = ctx.mul_mat(&out_w, &out)?;
                let t = std::time::Instant::now();
                ctx.compute(&out, threads)?;
                Ok((out.to_vec_f32(), kv_secs, t.elapsed().as_secs_f64()))
            })();
            self.scratch = buf;
            let (attn_out, kv_secs, attn_secs) = attn_result?;
            self.stats.kv_build_seconds += kv_secs;
            self.stats.attn_seconds += attn_secs;

            // Residual, then the feed-forward.
            // Gemma normalises the attention output *before* it rejoins the
            // residual stream, on top of the pre-norm every other architecture
            // here uses.
            let attn_out = if c.post_norms {
                self.post_norm(weights, il, &attn_out, n_new, "post_attention_norm")?
            } else {
                attn_out
            };
            let mut ffn_input = residual;
            for (dst, v) in ffn_input.iter_mut().zip(attn_out) {
                *dst += v;
            }

            let (normed_v, probs_v) = {
                let ctx = Context::new(arena_for(
                    &[
                        (n_embd, n_new),            // ffn input
                        (n_embd, n_new),            // normalised
                        (n_embd, n_new),            // rms intermediate
                        (c.n_expert as i64, n_new), // router logits
                        (c.n_expert as i64, n_new), // router probabilities
                    ],
                    24,
                ))?;
                let xt = ctx.new_f32_2d(n_embd, n_new)?;
                xt.set_f32(&ffn_input)?;
                let normed = self.arch.norm_scaled(
                    &ctx,
                    &xt,
                    &get(weights, format!("blk.{il}.ffn_norm.weight"))?,
                )?;
                if !c.is_moe() {
                    // Dense: no router at all. The FFN is one gate/up/down
                    // triple on resident weights, so it runs here rather than
                    // through the expert machinery.
                    let ffn = self.arch.dense_ffn(&ctx, weights, &normed, il)?;
                    let ffn = if c.post_norms {
                        let w = get(weights, format!("blk.{il}.post_ffw_norm.weight"))?;
                        self.arch.norm_scaled(&ctx, &ffn, &w)?
                    } else {
                        ffn
                    };
                    let out = ctx.add(&ffn, &xt)?;
                    let t = std::time::Instant::now();
                    ctx.compute(&out, threads)?;
                    self.stats.other_seconds += t.elapsed().as_secs_f64();
                    x = out.to_vec_f32();
                    continue;
                }
                let logits = ctx.mul_mat(
                    &get(weights, format!("blk.{il}.ffn_gate_inp.weight"))?,
                    &normed,
                )?;
                let probs = ctx.soft_max_ext(&logits, None, 1.0, 0.0)?;
                let t = std::time::Instant::now();
                ctx.compute(&probs, threads)?;
                self.stats.other_seconds += t.elapsed().as_secs_f64();
                (normed.to_vec_f32(), probs.to_vec_f32())
            };

            let mut next = ffn_input;
            let expert_out = self.expert_ffn_block(&normed_v, il, &probs_v, tokens.len())?;
            let expert_out = if c.post_norms {
                self.post_norm(weights, il, &expert_out, n_new, "post_ffw_norm")?
            } else {
                expert_out
            };
            for (dst, v) in next.iter_mut().zip(expert_out) {
                *dst += v;
            }
            x = next;
        }
        cache.advance_by(tokens.len());

        // Only the last position's logits are ever sampled, and the vocabulary
        // projection is the widest matmul in the model (151,936 rows here).
        // Running it for every prompt token costs the whole prefill that much
        // again for results nothing reads.
        let t_out = std::time::Instant::now();
        let last = x.len() - n_embd as usize;
        let ctx = Context::new(arena_for(
            &[(n_embd, 1), (n_embd, 1), (c.vocab_size as i64, 1)],
            16,
        ))?;
        let xt = ctx.new_f32_2d(n_embd, 1)?;
        xt.set_f32(&x[last..])?;
        let normed = self.arch.norm_scaled(
            &ctx,
            &xt,
            weights
                .get("output_norm.weight")
                .ok_or_else(|| ArchError::MissingTensor("output_norm.weight".into()))?,
        )?;
        let out_name = if weights.get("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let out = ctx.mul_mat(
            weights
                .get(out_name)
                .ok_or_else(|| ArchError::MissingTensor(out_name.into()))?,
            &normed,
        )?;
        // Gemma bounds the final logits smoothly rather than clipping them.
        // Without it every sampling decision is made on the wrong scale.
        let out = ctx.softcap(&out, c.final_logit_softcap)?;
        ctx.compute(&out, threads)?;
        let logits = out.to_vec_f32();
        self.stats.other_seconds += t_out.elapsed().as_secs_f64();
        Ok(logits)
    }

    /// Full forward pass, streaming experts, returning logits for every token.
    ///
    /// Runs one layer at a time so the router's choice can drive what is read
    /// next. Activations cross layer boundaries as plain `Vec<f32>` — small
    /// (`n_embd * n_tokens`) compared to the weights, and it lets each layer's
    /// arena be reclaimed immediately.
    pub fn forward<'a>(
        &mut self,
        weights: &WeightSet<'a>,
        tokens: &[u32],
        positions: &[i32],
    ) -> Result<Vec<f32>> {
        let c = self.arch.config.clone();
        let n_tokens = tokens.len() as i64;
        let n_embd = c.n_embd as i64;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Embedding lookup, once.
        let mut x: Vec<f32> = {
            let ctx = Context::new(512 << 20)?;
            let tok = ctx.new_i32_1d(n_tokens)?;
            tok.set_i32(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>())?;
            let emb = weights
                .get("token_embd.weight")
                .ok_or_else(|| ArchError::MissingTensor("token_embd.weight".into()))?;
            let rows = ctx.get_rows(emb, &tok)?;
            ctx.compute(&rows, threads)?;
            rows.to_vec_f32()
        };

        for il in 0..c.n_layer {
            // Attention and the router run as one graph; computing the router
            // output also materialises everything upstream, so the residual
            // and the normed activations can be read from the same pass.
            let ctx = Context::new(1 << 30)?;
            let get = |n: &str| -> Result<&Tensor<'a>> {
                weights
                    .get(n)
                    .ok_or_else(|| ArchError::MissingTensor(n.into()))
            };

            let xt = ctx.new_f32_2d(n_embd, n_tokens)?;
            xt.set_f32(&x)?;
            let pos = ctx.new_i32_1d(n_tokens)?;
            pos.set_i32(positions)?;

            let attn_out = self.arch.attention_block(
                &ctx,
                weights,
                &xt,
                &pos,
                n_tokens,
                il,
                self.rope(),
                ROPE_TYPE_NEOX,
            )?;
            let ffn_input = ctx.add(&attn_out, &xt)?;
            let normed = self.arch.norm_scaled(
                &ctx,
                &ffn_input,
                get(&format!("blk.{il}.ffn_norm.weight"))?,
            )?;

            let logits = ctx.mul_mat(get(&format!("blk.{il}.ffn_gate_inp.weight"))?, &normed)?;
            let probs = ctx.soft_max_ext(&logits, None, 1.0, 0.0)?;
            ctx.compute(&probs, threads)?;

            let residual = ffn_input.to_vec_f32();
            let normed_v = normed.to_vec_f32();
            let probs_v = probs.to_vec_f32();
            drop(ctx);

            // Experts, per token: the router's choice differs for each.
            let n_expert = c.n_expert as usize;
            let mut next = residual;
            for t in 0..tokens.len() {
                let lo = t * c.n_embd as usize;
                let hi = lo + c.n_embd as usize;
                let token_probs = &probs_v[t * n_expert..(t + 1) * n_expert];
                let expert_out = self.expert_ffn(&normed_v[lo..hi], il, token_probs)?;
                for (dst, v) in next[lo..hi].iter_mut().zip(expert_out) {
                    *dst += v;
                }
            }
            x = next;
        }

        // Final norm and output projection.
        let ctx = Context::new(1 << 30)?;
        let xt = ctx.new_f32_2d(n_embd, n_tokens)?;
        xt.set_f32(&x)?;
        let normed = self.arch.norm_scaled(
            &ctx,
            &xt,
            weights
                .get("output_norm.weight")
                .ok_or_else(|| ArchError::MissingTensor("output_norm.weight".into()))?,
        )?;
        let out_name = if weights.get("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let out = ctx.mul_mat(
            weights
                .get(out_name)
                .ok_or_else(|| ArchError::MissingTensor(out_name.into()))?,
            &normed,
        )?;
        ctx.compute(&out, threads)?;
        Ok(out.to_vec_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Qwen3Config {
        Qwen3Config {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 4,
            n_ff: 16,
            vocab_size: 32,
            rms_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            n_expert: 4,
            n_expert_used: 2,
            n_ff_expert: 16,
            qk_norm: true,
            rope_type: 2,
            rope_type_is_known: true,
            fused_qkv: false,
            fused_gate_up: false,
            post_norms: false,
            scale_embeddings: false,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            sliding_window: 0,
        }
    }

    /// Routing is pure arithmetic, so it is testable without a model.
    fn route_only(probs: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        let total: f32 = scored.iter().map(|(_, p)| *p).sum();
        if total > 0.0 {
            for (_, p) in &mut scored {
                *p /= total;
            }
        }
        scored
    }

    #[test]
    fn routing_picks_the_highest_scoring_experts() {
        let picks = route_only(&[0.1, 0.5, 0.2, 0.9], 2);
        let ids: Vec<u32> = picks.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![3, 1], "should pick experts 3 and 1");
    }

    #[test]
    fn routing_weights_are_renormalised_over_the_selection() {
        // Without this the output is scaled down by the probability mass of
        // the experts that were never run.
        let picks = route_only(&[0.1, 0.5, 0.2, 0.9], 2);
        let total: f32 = picks.iter().map(|(_, w)| *w).sum();
        assert!((total - 1.0).abs() < 1e-6, "weights summed to {total}");
    }

    #[test]
    fn resident_set_excludes_experts() {
        // The entire premise: experts must not be resident.
        let arch = Qwen3Model::new(cfg());
        let resident: Vec<String> = arch
            .required_tensors()
            .into_iter()
            .filter(|n| !n.contains("_exps"))
            .collect();
        assert!(!resident.is_empty());
        assert!(
            resident.iter().all(|n| !n.contains("_exps")),
            "an expert tensor leaked into the resident set"
        );
        assert!(resident.iter().any(|n| n.contains("attn_q")));
    }

    #[test]
    fn zero_probabilities_do_not_divide_by_zero() {
        let picks = route_only(&[0.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(picks.len(), 2);
        assert!(picks.iter().all(|(_, w)| w.is_finite()));
    }
}
