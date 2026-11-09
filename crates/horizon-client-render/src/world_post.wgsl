// Horizon's world post pass: the world, rendered at the resolution scale,
// copied (bilinear) or anti-aliased with FXAA onto the screen before the
// hand and HUD. Only used when a non-vanilla setting asks for it.
@group(0) @binding(0) var world: texture_2d<f32>;
@group(0) @binding(1) var world_sampler: sampler;

struct PostOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_post(@builtin(vertex_index) i: u32) -> PostOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: PostOut;
    out.clip = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2(p.x, 1.0 - p.y);
    return out;
}

@fragment fn fs_copy(in: PostOut) -> @location(0) vec4<f32> {
    // The alpha carries the emissive mark through to HDR output (hdr.wgsl).
    return textureSampleLevel(world, world_sampler, in.uv, 0.0);
}

// The world drawn below the screen's size: Catmull-Rom in five bilinear taps (the four corners
// of its 4x4 weigh too little to matter), clamped to what the taps saw so edges don't ring.
@fragment fn fs_upscale(in: PostOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(world));
    let pos = in.uv * size;
    let centre = floor(pos - 0.5) + 0.5;
    let f = pos - centre;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (centre - 1.0) / size;
    let t12 = (centre + w2 / w12) / size;
    let t3 = (centre + 2.0) / size;
    let middle = textureSampleLevel(world, world_sampler, t12, 0.0);
    let up = textureSampleLevel(world, world_sampler, vec2(t12.x, t0.y), 0.0).rgb;
    let left = textureSampleLevel(world, world_sampler, vec2(t0.x, t12.y), 0.0).rgb;
    let right = textureSampleLevel(world, world_sampler, vec2(t3.x, t12.y), 0.0).rgb;
    let down = textureSampleLevel(world, world_sampler, vec2(t12.x, t3.y), 0.0).rgb;
    let wc = w12.x * w12.y;
    let wu = w12.x * w0.y;
    let wl = w0.x * w12.y;
    let wr = w3.x * w12.y;
    let wd = w12.x * w3.y;
    let sum = middle.rgb * wc + up * wu + left * wl + right * wr + down * wd;
    let low = min(min(min(min(middle.rgb, up), left), right), down);
    let high = max(max(max(max(middle.rgb, up), left), right), down);
    // The alpha carries the emissive mark through to HDR output (hdr.wgsl).
    return vec4(clamp(sum / (wc + wu + wl + wr + wd), low, high), middle.a);
}

// FSR 1.0 EASU (AMD FidelityFX Super Resolution 1, MIT licence), ported to WGSL in f32. It reads
// the world's 12-texel neighbourhood with `textureLoad` (no gathers, so the GL backend takes it
// too), finds the local edge direction from luma and stretches a Lanczos-like kernel along it.
// `EASU_SHARPEN` stands in for RCAS: an unsharp step against the plain bilinear value, kept
// inside the 4 nearest texels' range so it cannot ring. 0.0 turns it off.
const EASU_SHARPEN: f32 = 0.3;

fn easu_load(p: vec2<i32>, last: vec2<i32>) -> vec4<f32> {
    return textureLoad(world, clamp(p, vec2(0), last), 0);
}

fn easu_luma(c: vec3<f32>) -> f32 {
    // Luma times two, in the cheapest form FSR uses.
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

// One bilinear quad's share of the edge direction (xy) and of how edge-like it is (z).
fn easu_set(w: f32, a: f32, b: f32, c: f32, d: f32, e: f32) -> vec3<f32> {
    let len_x = 1.0 / max(max(abs(d - c), abs(c - b)), 1.0e-6);
    let dir_x = d - b;
    let sx = clamp(abs(dir_x) * len_x, 0.0, 1.0);
    let len_y = 1.0 / max(max(abs(e - c), abs(c - a)), 1.0e-6);
    let dir_y = e - a;
    let sy = clamp(abs(dir_y) * len_y, 0.0, 1.0);
    return vec3(dir_x * w, dir_y * w, (sx * sx + sy * sy) * w);
}

// One texel's weight (w) added to the running colour sum (rgb).
fn easu_tap(off: vec2<f32>, dir: vec2<f32>, len: vec2<f32>, lob: f32, clp: f32, c: vec3<f32>) -> vec4<f32> {
    let v = vec2(off.x * dir.x + off.y * dir.y, off.x * -dir.y + off.y * dir.x) * len;
    let d2 = min(dot(v, v), clp);
    // Lanczos2 without sin() or sqrt().
    var wb = 0.4 * d2 - 1.0;
    var wa = lob * d2 - 1.0;
    wb = wb * wb;
    wa = wa * wa;
    wb = 1.5625 * wb - 0.5625;
    let w = wb * wa;
    return vec4(c * w, w);
}

@fragment fn fs_easu(in: PostOut) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(world));
    let last = dims - 1;
    var pp = in.uv * vec2<f32>(dims) - 0.5;
    let fp = floor(pp);
    pp = pp - fp;
    let o = vec2<i32>(fp);
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    let b = easu_load(o + vec2(0, -1), last);
    let c = easu_load(o + vec2(1, -1), last);
    let e = easu_load(o + vec2(-1, 0), last);
    let f = easu_load(o, last);
    let g = easu_load(o + vec2(1, 0), last);
    let h = easu_load(o + vec2(2, 0), last);
    let i = easu_load(o + vec2(-1, 1), last);
    let j = easu_load(o + vec2(0, 1), last);
    let k = easu_load(o + vec2(1, 1), last);
    let l = easu_load(o + vec2(2, 1), last);
    let n = easu_load(o + vec2(0, 2), last);
    let q = easu_load(o + vec2(1, 2), last);

    let bl = easu_luma(b.rgb);
    let cl = easu_luma(c.rgb);
    let el = easu_luma(e.rgb);
    let fl = easu_luma(f.rgb);
    let gl = easu_luma(g.rgb);
    let hl = easu_luma(h.rgb);
    let il = easu_luma(i.rgb);
    let jl = easu_luma(j.rgb);
    let kl = easu_luma(k.rgb);
    let ll = easu_luma(l.rgb);
    let nl = easu_luma(n.rgb);
    let ql = easu_luma(q.rgb);

    let s = easu_set((1.0 - pp.x) * (1.0 - pp.y), bl, el, fl, gl, jl)
        + easu_set(pp.x * (1.0 - pp.y), cl, fl, gl, hl, kl)
        + easu_set((1.0 - pp.x) * pp.y, fl, il, jl, kl, nl)
        + easu_set(pp.x * pp.y, gl, jl, kl, ll, ql);
    var dir = s.xy;
    var len = s.z;
    var dir_r = dir.x * dir.x + dir.y * dir.y;
    if dir_r < 1.0 / 32768.0 {
        dir_r = 1.0;
        dir.x = 1.0;
    } else {
        dir_r = inverseSqrt(dir_r);
    }
    dir = dir * dir_r;
    // From {0 to 2} to {0 to 1}, shaped with a square.
    len = len * 0.5;
    len = len * len;
    // Stretch the kernel from 1 (straight edges) to sqrt(2) (diagonals).
    let stretch = (dir.x * dir.x + dir.y * dir.y) / max(abs(dir.x), abs(dir.y));
    let len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    // The window shifts from +/-sqrt(2) to just beyond 2 with the amount of edge.
    let lob = 0.5 + ((0.25 - 0.04) - 0.5) * len;
    let clp = 1.0 / lob;

    // Deringing: the output stays inside the 4 nearest texels' range.
    let min4 = min(min(j.rgb, k.rgb), min(f.rgb, g.rgb));
    let max4 = max(max(j.rgb, k.rgb), max(f.rgb, g.rgb));

    var acc = vec4(0.0);
    acc += easu_tap(vec2(0.0, -1.0) - pp, dir, len2, lob, clp, b.rgb);
    acc += easu_tap(vec2(1.0, -1.0) - pp, dir, len2, lob, clp, c.rgb);
    acc += easu_tap(vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, i.rgb);
    acc += easu_tap(vec2(0.0, 1.0) - pp, dir, len2, lob, clp, j.rgb);
    acc += easu_tap(vec2(0.0, 0.0) - pp, dir, len2, lob, clp, f.rgb);
    acc += easu_tap(vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, e.rgb);
    acc += easu_tap(vec2(1.0, 1.0) - pp, dir, len2, lob, clp, k.rgb);
    acc += easu_tap(vec2(2.0, 1.0) - pp, dir, len2, lob, clp, l.rgb);
    acc += easu_tap(vec2(2.0, 0.0) - pp, dir, len2, lob, clp, h.rgb);
    acc += easu_tap(vec2(1.0, 0.0) - pp, dir, len2, lob, clp, g.rgb);
    acc += easu_tap(vec2(1.0, 2.0) - pp, dir, len2, lob, clp, q.rgb);
    acc += easu_tap(vec2(0.0, 2.0) - pp, dir, len2, lob, clp, n.rgb);
    let pix = min(max4, max(min4, acc.rgb / acc.a));

    let plain = mix(mix(f.rgb, g.rgb, pp.x), mix(j.rgb, k.rgb, pp.x), pp.y);
    let sharp = clamp(pix + EASU_SHARPEN * (pix - plain), min4, max4);
    // The alpha carries the emissive mark through to HDR output (hdr.wgsl).
    let alpha = mix(mix(f.a, g.a, pp.x), mix(j.a, k.a, pp.x), pp.y);
    return vec4(sharp, alpha);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

// FXAA 3.11 (console quality): blend along the local edge direction.
@fragment fn fs_fxaa(in: PostOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(world));
    let centre = textureSampleLevel(world, world_sampler, in.uv, 0.0);
    let rgb_m = centre.rgb;
    let nw = luma(textureSampleLevel(world, world_sampler, in.uv + vec2(-0.5, -0.5) * texel, 0.0).rgb);
    let ne = luma(textureSampleLevel(world, world_sampler, in.uv + vec2(0.5, -0.5) * texel, 0.0).rgb);
    let sw = luma(textureSampleLevel(world, world_sampler, in.uv + vec2(-0.5, 0.5) * texel, 0.0).rgb);
    let se = luma(textureSampleLevel(world, world_sampler, in.uv + vec2(0.5, 0.5) * texel, 0.0).rgb);
    let m = luma(rgb_m);
    let lo = min(m, min(min(nw, ne), min(sw, se)));
    let hi = max(m, max(max(nw, ne), max(sw, se)));
    if hi - lo < max(0.0312, hi * 0.125) {
        return centre;
    }
    var dir = vec2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
    let reduce = max((nw + ne + sw + se) * 0.25 * 0.125, 1.0 / 128.0);
    let scale = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * scale, vec2(-8.0), vec2(8.0)) * texel;
    let a = 0.5 * (textureSampleLevel(world, world_sampler, in.uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb
        + textureSampleLevel(world, world_sampler, in.uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
    let b = a * 0.5 + 0.25 * (textureSampleLevel(world, world_sampler, in.uv - dir * 0.5, 0.0).rgb
        + textureSampleLevel(world, world_sampler, in.uv + dir * 0.5, 0.0).rgb);
    let lb = luma(b);
    if lb < lo || lb > hi {
        return vec4(a, centre.a);
    }
    return vec4(b, centre.a);
}

// TAA (`world_post.rs`): blend this frame into the history, reprojected from the
// depth buffer through the camera's motion and clamped to this frame's 3x3
// neighbourhood so that what moved or appeared does not ghost.
struct Taa {
    reproject: mat4x4<f32>,
    // xy: this frame's jitter in clip space; z: 1 when the history is unusable.
    jitter: vec4<f32>,
};
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var depth: texture_depth_2d;
@group(0) @binding(4) var<uniform> taa: Taa;

// Catmull-Rom over the history in 5 bilinear taps: sharper than one under motion.
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(history));
    let pos = uv * size;
    let centre = floor(pos - 0.5) + 0.5;
    let f = pos - centre;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (centre - 1.0) / size;
    let t3 = (centre + 2.0) / size;
    let t12 = (centre + w2 / w12) / size;
    let c = textureSampleLevel(history, world_sampler, vec2(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y)
        + textureSampleLevel(history, world_sampler, vec2(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y)
        + textureSampleLevel(history, world_sampler, t12, 0.0).rgb * (w12.x * w12.y)
        + textureSampleLevel(history, world_sampler, vec2(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y)
        + textureSampleLevel(history, world_sampler, vec2(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    let weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(c / weight, vec3(0.0));
}

// It runs at the history's size, which upscaling makes larger than the world's.
// The history is unjittered: each output pixel takes the world pixel whose
// jittered sample lands nearest its centre, weighted by how near it lands.
@fragment fn fs_taa(in: PostOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(world));
    // The jitter moved the scene by `jitter` in clip space: where this pixel's
    // point was drawn, in world pixels.
    let drawn = (in.uv + taa.jitter.xy * vec2(0.5, -0.5)) * size;
    let last = vec2<i32>(size) - 1;
    let p = clamp(vec2<i32>(floor(drawn)), vec2(0), last);
    let centre = textureLoad(world, p, 0);
    var lo = centre.rgb;
    var hi = centre.rgb;
    // Reversed depth: the nearest of the 3x3 is the largest, and its motion keeps
    // the edges of what is in front from trailing.
    var near = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(p + vec2(x, y), vec2(0), last);
            let c = textureLoad(world, q, 0).rgb;
            lo = min(lo, c);
            hi = max(hi, c);
            near = max(near, textureLoad(depth, q, 0));
        }
    }
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let previous = taa.reproject * vec4(ndc, near, 1.0);
    let uv = vec2(0.5, -0.5) * previous.xy / previous.w + 0.5;
    if taa.jitter.z > 0.0 || any(uv != clamp(uv, vec2(0.0), vec2(1.0))) {
        return vec4(centre.rgb, centre.a);
    }
    let past = clamp(sample_history(uv), lo, hi);
    // How far the sample landed from this pixel's centre, in output pixels: a
    // Gaussian of it scales the new frame's share, so that frames whose sample
    // fell far away (most of them when upscaling) barely move the pixel.
    let off = (drawn - (vec2<f32>(p) + 0.5)) * vec2<f32>(textureDimensions(history)) / size;
    let share = 0.1 * exp(-2.0 * dot(off, off));
    // The alpha carries the emissive mark through to HDR output (hdr.wgsl).
    return vec4(mix(past, centre.rgb, share), centre.a);
}
