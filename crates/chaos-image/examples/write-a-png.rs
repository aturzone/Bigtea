fn main() {
    // A gradient with a hard edge: a row shift or a channel swap is obvious.
    let (w, h) = (256u32, 128u32);
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let left = x < w / 2;
            px.push(if left { x as u8 } else { 255 - x as u8 });
            px.push((y * 2) as u8);
            px.push(if left { 40 } else { 200 });
        }
    }
    let png = chaos_image::png::encode_rgb(w, h, &px).expect("encode");
    std::fs::write(std::env::args().nth(1).unwrap(), &png).unwrap();
    eprintln!("wrote {} bytes", png.len());
}
