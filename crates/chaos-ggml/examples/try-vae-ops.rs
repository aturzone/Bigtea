//! Try `conv_2d` and `group_norm` in isolation before they go near the suite.
//!
//! **ggml aborts rather than returning an error**, and an abort takes the whole
//! test binary with it — not one test. So a new op is exercised here first,
//! where a crash costs one `cargo run` and tells you exactly which call did it.
//!
//! Numbers chosen to be checkable by hand:
//!
//! - a 1×1 kernel of 2, over `[1, 2, 3, 4]`, must give `[2, 4, 6, 8]`;
//! - group norm over one group of `[1, 2, 3, 4]` has mean 2.5 and population
//!   variance 1.25, so the output is `(x - 2.5) / sqrt(1.25)` — about
//!   `[-1.342, -0.447, 0.447, 1.342]`;
//! - nearest-neighbour 2x on the 2×2 `[[1,2],[3,4]]` doubles every pixel into
//!   a 2×2 block;
//! - `pad` puts its zeros on the **right and bottom only**, which is what
//!   makes it diffusers' `F.pad(x, (0, 1, 0, 1))` rather than a symmetric one.

fn main() {
    let ctx = match chaos_ggml::Context::new(256 << 20) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no context: {e:?}");
            std::process::exit(1);
        }
    };

    // -- conv_2d --------------------------------------------------------------
    // data [w, h, channels, batch] = [2, 2, 1, 1]
    let data = ctx.new_f32_1d(4).expect("data");
    data.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set data");
    let data = ctx.reshape_4d(&data, [2, 2, 1, 1]).expect("reshape data");

    // kernel [kw, kh, in, out] = [1, 1, 1, 1], F16, value 2
    let kernel = ctx.new_f16_4d(1, 1, 1, 1).expect("kernel");
    // F16 2.0 is 0x4000. Written as raw bytes because there is no f16 in Rust.
    kernel
        .set_bytes(&0x4000u16.to_le_bytes())
        .expect("set kernel");

    println!("calling conv_2d ...");
    match ctx.conv_2d(&kernel, &data, (1, 1), (0, 0), (1, 1)) {
        Ok(out) => {
            ctx.compute(&out, 1).expect("compute conv");
            let got = out.to_vec_f32();
            println!("  conv_2d -> {got:?}");
            let want = [2.0f32, 4.0, 6.0, 8.0];
            let ok = got.len() == 4 && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-3);
            println!("  {}", if ok { "MATCHES [2,4,6,8]" } else { "DIFFERS" });
        }
        Err(e) => println!("  conv_2d refused: {e:?}"),
    }

    // -- group_norm -----------------------------------------------------------
    let x = ctx.new_f32_1d(4).expect("x");
    x.set_f32(&[1.0, 2.0, 3.0, 4.0]).expect("set x");
    // [w, h, channels, batch]: one channel, so one group covers everything.
    let x = ctx.reshape_4d(&x, [4, 1, 1, 1]).expect("reshape x");
    println!("calling group_norm ...");
    match ctx.group_norm(&x, 1, 1e-5) {
        Ok(out) => {
            ctx.compute(&out, 1).expect("compute gn");
            let got = out.to_vec_f32();
            println!("  group_norm -> {got:?}");
            let want = [-1.3416f32, -0.4472, 0.4472, 1.3416];
            let ok = got.len() == 4 && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 2e-3);
            println!(
                "  {}",
                if ok {
                    "MATCHES (x - 2.5) / sqrt(1.25)"
                } else {
                    "DIFFERS"
                }
            );
        }
        Err(e) => println!("  group_norm refused: {e:?}"),
    }
    // -- conv_2d_direct, with an F32 kernel ----------------------------------
    // If this works, the VAE's 138 F32 weights need no conversion at all.
    let k32 = ctx.new_f32_1d(1).expect("k32");
    k32.set_f32(&[2.0]).expect("set k32");
    let k32 = ctx.reshape_4d(&k32, [1, 1, 1, 1]).expect("reshape k32");
    println!("calling conv_2d_direct with an F32 kernel ...");
    match ctx.conv_2d_direct(&k32, &data, (1, 1), (0, 0), (1, 1)) {
        Ok(out) => {
            ctx.compute(&out, 1).expect("compute direct");
            let got = out.to_vec_f32();
            println!("  conv_2d_direct -> {got:?}");
            let want = [2.0f32, 4.0, 6.0, 8.0];
            let ok = got.len() == 4 && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-3);
            println!(
                "  {}",
                if ok {
                    "MATCHES -- F32 kernels work, no conversion needed"
                } else {
                    "DIFFERS"
                }
            );
        }
        Err(e) => println!("  conv_2d_direct refused: {e:?}"),
    }

    // -- upscale_nearest ------------------------------------------------------
    // [[1, 2], [3, 4]] at 2x is each pixel smeared over a 2x2 block.
    println!("calling upscale_nearest ...");
    match ctx.upscale_nearest(&data, 2) {
        Ok(out) => {
            ctx.compute(&out, 1).expect("compute upscale");
            let got = out.to_vec_f32();
            println!("  upscale_nearest -> {got:?}");
            let want = [
                1.0f32, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
            ];
            let ok = got.len() == 16 && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6);
            println!(
                "  {}",
                if ok {
                    "MATCHES a 2x2 nearest smear"
                } else {
                    "DIFFERS"
                }
            );
        }
        Err(e) => println!("  upscale_nearest refused: {e:?}"),
    }

    // -- pad ------------------------------------------------------------------
    // Right and bottom only: rows become (1,2,0) and (3,4,0), then a zero row.
    println!("calling pad ...");
    match ctx.pad(&data, [1, 1, 0, 0]) {
        Ok(out) => {
            ctx.compute(&out, 1).expect("compute pad");
            let got = out.to_vec_f32();
            println!("  pad -> {got:?}");
            let want = [1.0f32, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
            let ok = got.len() == 9 && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6);
            println!(
                "  {}",
                if ok {
                    "MATCHES -- zeros on the right and bottom, not centred"
                } else {
                    "DIFFERS -- the asymmetry assumption is wrong"
                }
            );
        }
        Err(e) => println!("  pad refused: {e:?}"),
    }

    println!("survived all calls");
}
