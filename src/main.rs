//! LUMEN DRIFT — a soft, dreamy third-person 3D orb-collecting game.
//!
//! How the "not sharp / not pixelated" look is made:
//!   1. The 3D scene is rendered into a 4x MSAA render target at 1.5x resolution
//!      (supersampling), then smoothly downsampled to the window.
//!   2. A post-process shader does edge-aware softening, a gentle bloom and a vignette.
//!   3. All geometry is smooth-shaded (CPU lighting per vertex), glows fade out at their
//!      silhouettes, shadows are feathered blobs, and distance fog melts the world away.
//!   4. HUD text uses a smooth TrueType font with linear filtering.

use macroquad::miniquad::{ShaderSource, UniformDesc, UniformType};
use macroquad::prelude::*;
use macroquad::rand::{gen_range, srand};
use std::f32::consts::{PI, TAU};

// ------------------------------------------------------------------------------------
// Tuning
// ------------------------------------------------------------------------------------
const ARENA_R: f32 = 26.0;
const ORBS_TO_WIN: u32 = 20;
const MAX_HP: i32 = 3;
const SUPERSAMPLE: f32 = 1.0;
const FOG: Vec3 = Vec3::new(0.82, 0.80, 0.93);
const PLUM: Color = Color::new(0.27, 0.18, 0.45, 1.0);
const BEST_FILE: &str = "lumen_drift_best.txt";

// ------------------------------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------------------------------
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn col(v: Vec3, a: f32) -> Color {
    Color::new(
        v.x.clamp(0.0, 1.0),
        v.y.clamp(0.0, 1.0),
        v.z.clamp(0.0, 1.0),
        a.clamp(0.0, 1.0),
    )
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut d = (b - a) % TAU;
    if d > PI {
        d -= TAU;
    } else if d < -PI {
        d += TAU;
    }
    a + d * t
}

fn horiz(v: Vec3) -> Vec3 {
    vec3(v.x, 0.0, v.z)
}

// ------------------------------------------------------------------------------------
// Soft shading + mesh builders
// ------------------------------------------------------------------------------------
struct Env {
    cam: Vec3,
    light: Vec3,
}

#[derive(Clone, Copy)]
enum Mat {
    /// Soft half-lambert lighting + pastel rim light, fogged.
    Lit,
    /// Self-lit, slightly brighter in the middle, fogged.
    Emissive,
    /// Transparent halo that fades out toward its silhouette (blurry edge!). Fogged.
    Glow(f32),
    /// Same as Glow but never fogged (sun, sky).
    SkyGlow(f32),
}

impl Env {
    fn fog_t(&self, p: Vec3) -> f32 {
        smoothstep(14.0, 70.0, p.distance(self.cam))
    }

    fn shade(&self, m: Mat, base: Vec3, alpha: f32, n: Vec3, p: Vec3) -> Color {
        let view = (self.cam - p).normalize_or_zero();
        match m {
            Mat::Lit => {
                let hemi = n.y * 0.5 + 0.5;
                let amb = Vec3::new(0.52, 0.47, 0.60).lerp(Vec3::new(0.95, 0.95, 1.0), hemi);
                let ndl = n.dot(self.light) * 0.5 + 0.5;
                let diff = ndl * ndl;
                let rim = (1.0 - n.dot(view).max(0.0)).powi(3);
                let c = base * (amb * 0.80 + Vec3::new(1.0, 0.94, 0.84) * diff * 0.50)
                    + Vec3::new(1.0, 0.85, 0.95) * rim * 0.20;
                col(c.lerp(FOG, self.fog_t(p) * 0.95), alpha)
            }
            Mat::Emissive => {
                let f = n.dot(view).abs();
                let c = base + (Vec3::ONE - base) * (f * f * 0.45);
                col(c.lerp(FOG, self.fog_t(p) * 0.6), alpha)
            }
            Mat::Glow(power) => {
                let f = n.dot(view).abs().powf(power);
                col(base, alpha * f * (1.0 - self.fog_t(p)))
            }
            Mat::SkyGlow(power) => {
                let f = n.dot(view).abs().powf(power);
                col(base, alpha * f)
            }
        }
    }
}

fn grid_indices(rings: usize, slices: usize) -> Vec<u16> {
    let mut idx = Vec::with_capacity(rings * slices * 6);
    for i in 0..rings {
        for j in 0..slices {
            let a = (i * (slices + 1) + j) as u16;
            let b = a + (slices + 1) as u16;
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    idx
}

#[allow(clippy::too_many_arguments)]
fn ellipsoid(
    c: Vec3,
    r: Vec3,
    yaw: f32,
    base: Vec3,
    alpha: f32,
    mat: Mat,
    env: &Env,
    rings: usize,
    slices: usize,
) -> Mesh {
    let q = Quat::from_rotation_y(yaw);
    let mut vertices = Vec::with_capacity((rings + 1) * (slices + 1));
    for i in 0..=rings {
        let phi = PI * i as f32 / rings as f32;
        let (sp, cp) = phi.sin_cos();
        for j in 0..=slices {
            let th = TAU * j as f32 / slices as f32;
            let (st, ct) = th.sin_cos();
            let u = vec3(sp * ct, cp, sp * st);
            let n = (q * vec3(u.x / r.x, u.y / r.y, u.z / r.z)).normalize();
            let p = c + q * (u * r);
            let uv = vec2(j as f32 / slices as f32, i as f32 / rings as f32);
            vertices.push(Vertex::new2(p, uv, env.shade(mat, base, alpha, n, p)));
        }
    }
    Mesh {
        vertices,
        indices: grid_indices(rings, slices),
        texture: None,
    }
}

fn blob(c: Vec3, r: f32, base: Vec3, mat: Mat, alpha: f32, env: &Env, quality: usize) -> Mesh {
    ellipsoid(c, Vec3::splat(r), 0.0, base, alpha, mat, env, quality, quality + quality / 2)
}

/// Vertical tapered tube with radial (smooth) normals.
fn tube(base_pos: Vec3, h: f32, r0: f32, r1: f32, base: Vec3, env: &Env, segs: usize, rows: usize) -> Mesh {
    let mut vertices = Vec::with_capacity((rows + 1) * (segs + 1));
    for i in 0..=rows {
        let t = i as f32 / rows as f32;
        let r = r0 + (r1 - r0) * t;
        for j in 0..=segs {
            let th = TAU * j as f32 / segs as f32;
            let (s, c) = th.sin_cos();
            let n = vec3(c, 0.0, s);
            let p = base_pos + vec3(c * r, h * t, s * r);
            let uv = vec2(j as f32 / segs as f32, t);
            vertices.push(Vertex::new2(p, uv, env.shade(Mat::Lit, base, 1.0, n, p)));
        }
    }
    Mesh {
        vertices,
        indices: grid_indices(rows, segs),
        texture: None,
    }
}

/// Glowing torus (used for shockwave rings).
fn torus(center: Vec3, major: f32, minor: f32, base: Vec3, alpha: f32, env: &Env) -> Mesh {
    let (ms, ns) = (44usize, 8usize);
    let mut vertices = Vec::with_capacity((ms + 1) * (ns + 1));
    for i in 0..=ms {
        let a = TAU * i as f32 / ms as f32;
        let (sa, ca) = a.sin_cos();
        for j in 0..=ns {
            let b = TAU * j as f32 / ns as f32;
            let (sb, cb) = b.sin_cos();
            let n = vec3(cb * ca, sb, cb * sa);
            let p = center + vec3(ca * major, 0.0, sa * major) + n * minor;
            let uv = vec2(i as f32 / ms as f32, j as f32 / ns as f32);
            vertices.push(Vertex::new2(p, uv, env.shade(Mat::Glow(1.4), base, alpha, n, p)));
        }
    }
    Mesh {
        vertices,
        indices: grid_indices(ms, ns),
        texture: None,
    }
}

/// Feathered blob shadow / light pool lying on the ground — alpha fades to 0 at the rim.
fn shadow(x: f32, z: f32, radius: f32, alpha: f32, tint: Vec3) -> Mesh {
    let n = 22usize;
    let y = 0.08;
    let mut vertices = vec![Vertex::new2(vec3(x, y, z), vec2(0.5, 0.5), col(tint, alpha))];
    for ring in 0..2 {
        let (rf, af) = if ring == 0 { (0.55, 0.72) } else { (1.0, 0.0) };
        for i in 0..=n {
            let a = TAU * i as f32 / n as f32;
            vertices.push(Vertex::new2(
                vec3(x + a.cos() * radius * rf, y, z + a.sin() * radius * rf),
                vec2(0.0, 0.0),
                col(tint, alpha * af),
            ));
        }
    }
    let mut indices = Vec::new();
    let w = (n + 1) as u16;
    for i in 0..n as u16 {
        indices.extend_from_slice(&[0, 1 + i, 2 + i]);
        let a = 1 + i;
        let b = 1 + w + i;
        indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
    }
    Mesh {
        vertices,
        indices,
        texture: None,
    }
}

fn sky_mesh(env: &Env) -> Mesh {
    let (rings, slices) = (12usize, 20usize);
    let radius = 400.0;
    let top = Vec3::new(0.66, 0.72, 0.96);
    let mid = Vec3::new(0.98, 0.84, 0.92);
    let mut vertices = Vec::new();
    for i in 0..=rings {
        let phi = PI * i as f32 / rings as f32;
        let (sp, cp) = phi.sin_cos();
        for j in 0..=slices {
            let th = TAU * j as f32 / slices as f32;
            let (st, ct) = th.sin_cos();
            let d = vec3(sp * ct, cp, sp * st);
            let t = d.y.max(0.0);
            let c = FOG.lerp(mid, smoothstep(0.0, 0.18, t)).lerp(top, smoothstep(0.12, 0.8, t));
            vertices.push(Vertex::new2(env.cam + d * radius, vec2(0.0, 0.0), col(c, 1.0)));
        }
    }
    Mesh {
        vertices,
        indices: grid_indices(rings, slices),
        texture: None,
    }
}

fn ground_color(x: f32, z: f32) -> Color {
    let r = (x * x + z * z).sqrt();
    let mint = Vec3::new(0.62, 0.91, 0.80);
    let lav = Vec3::new(0.76, 0.72, 0.98);
    let peach = Vec3::new(1.0, 0.82, 0.78);
    let t1 = 0.5 + 0.5 * (0.19 * x + 0.8 * (0.13 * z).sin()).sin();
    let t2 = 0.5 + 0.5 * (0.17 * z - 0.6 * (0.11 * x).cos()).sin();
    let mut c = mint.lerp(lav, t1 * 0.75).lerp(peach, t2 * t2 * 0.28);
    let ring = 0.5 + 0.5 * (r * 0.9).cos();
    c *= 0.96 + 0.04 * ring;
    c += Vec3::splat(0.06 * (-r * r / 60.0).exp());
    c = c.lerp(Vec3::new(1.0, 0.88, 0.95), smoothstep(ARENA_R - 3.0, ARENA_R, r) * 0.7);
    c = c.lerp(FOG, smoothstep(ARENA_R, ARENA_R + 14.0, r));
    col(c * 0.95, 1.0)
}

fn build_ground() -> Vec<Mesh> {
    let half = 44i32;
    let n = (half * 2) as usize;
    let row_vertices = (n + 1) * 2;
    const ROWS_PER_MESH: usize = 8;

    let mut meshes = Vec::new();

    for start_row in (0..n).step_by(ROWS_PER_MESH) {
        let end_row = (start_row + ROWS_PER_MESH).min(n);
        let rows = end_row - start_row;
        let mut vertices = Vec::with_capacity(rows * row_vertices);
        let mut indices = Vec::with_capacity(rows * n * 6);

        for row in start_row..end_row {
            let z0 = -half as f32 + row as f32;
            let z1 = z0 + 1.0;

            for i in 0..=n {
                let x = -half as f32 + i as f32;
                vertices.push(Vertex::new2(
                    vec3(x, 0.0, z0),
                    vec2(0.0, 0.0),
                    ground_color(x, z0),
                ));
                vertices.push(Vertex::new2(
                    vec3(x, 0.0, z1),
                    vec2(0.0, 0.0),
                    ground_color(x, z1),
                ));
            }

            let row_start = ((row - start_row) * row_vertices) as u16;
            for i in 0..n {
                let a = row_start + (i * 2) as u16;
                indices.extend_from_slice(&[a, a + 1, a + 2, a + 2, a + 1, a + 3]);
            }
        }

        meshes.push(Mesh {
            vertices,
            indices,
            texture: None,
        });
    }

    meshes
}

// ------------------------------------------------------------------------------------
// Game data
// ------------------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq)]
enum State {
    Title,
    Playing,
    Paused,
    Won,
    Lost,
}

struct Player {
    pos: Vec3,
    vel: Vec3,
    face: f32,
    on_ground: bool,
    jumps: u8,
    stamina: f32,
    hp: i32,
    inv: f32,
    burst_cd: f32,
    land_squash: f32,
    anim: f32,
    dust: f32,
}

impl Player {
    fn new() -> Self {
        Player {
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            face: PI,
            on_ground: true,
            jumps: 2,
            stamina: 1.0,
            hp: MAX_HP,
            inv: 0.0,
            burst_cd: 0.0,
            land_squash: 0.0,
            anim: 0.0,
            dust: 0.0,
        }
    }
}

struct Orb {
    pos: Vec3,
    phase: f32,
    heart: bool,
}

struct Enemy {
    pos: Vec3,
    vel: Vec3,
    phase: f32,
    stun: f32,
    age: f32,
}

struct Particle {
    pos: Vec3,
    vel: Vec3,
    life: f32,
    max: f32,
    col: Vec3,
    size: f32,
    grav: f32,
}

struct Prop {
    pos: Vec3,
    r: f32,
    kind: u8,
    scale: f32,
    hue: Vec3,
}

struct Shockwave {
    t: f32,
    center: Vec3,
}

#[allow(clippy::too_many_arguments)]
fn emit(
    ps: &mut Vec<Particle>,
    pos: Vec3,
    n: usize,
    color: Vec3,
    speed: f32,
    up: f32,
    size: f32,
    life: f32,
    grav: f32,
) {
    for _ in 0..n {
        if ps.len() > 380 {
            break;
        }
        let a = gen_range(0.0, TAU);
        let s = gen_range(0.3, 1.0) * speed;
        let l = life * gen_range(0.6, 1.0);
        ps.push(Particle {
            pos,
            vel: vec3(a.cos() * s, up * gen_range(0.4, 1.0), a.sin() * s),
            life: l,
            max: l,
            col: color,
            size,
            grav,
        });
    }
}

fn make_props() -> Vec<Prop> {
    let pal = [
        vec3(1.0, 0.62, 0.75),
        vec3(0.55, 0.85, 0.90),
        vec3(1.0, 0.85, 0.50),
        vec3(0.75, 0.65, 1.0),
    ];
    let mut props: Vec<Prop> = Vec::new();
    let mut tries = 0;
    while props.len() < 14 && tries < 800 {
        tries += 1;
        let ang = gen_range(0.0, TAU);
        let rad = gen_range(6.0, ARENA_R - 4.0);
        let pos = vec3(ang.cos() * rad, 0.0, ang.sin() * rad);
        if props.iter().any(|p| p.pos.distance(pos) < 5.5) {
            continue;
        }
        let kind = (props.len() % 3) as u8;
        let scale = gen_range(0.8, 1.35);
        let hue = pal[gen_range(0, 4) as usize];
        let r = match kind {
            0 => 0.55 * scale,
            1 => 1.05 * scale,
            _ => 0.6 * scale,
        };
        props.push(Prop { pos, r, kind, scale, hue });
    }
    props
}

fn push_out_props(pos: &mut Vec3, radius: f32, props: &[Prop]) {
    for p in props {
        let d = vec3(pos.x - p.pos.x, 0.0, pos.z - p.pos.z);
        let len = d.length();
        let min = p.r + radius;
        if len < min {
            if len > 1e-4 {
                pos.x = p.pos.x + d.x / len * min;
                pos.z = p.pos.z + d.z / len * min;
            } else {
                pos.x += min;
            }
        }
    }
}

fn clamp_arena(pos: &mut Vec3, margin: f32) {
    let h = vec2(pos.x, pos.z);
    let max = ARENA_R - margin;
    if h.length() > max {
        let h = h.normalize() * max;
        pos.x = h.x;
        pos.z = h.y;
    }
}

fn random_spot(props: &[Prop], avoid: Vec3, min_avoid: f32) -> Vec3 {
    let mut best = vec3(8.0, 0.0, 8.0);
    for _ in 0..80 {
        let a = gen_range(0.0, TAU);
        let r = gen_range(3.0, ARENA_R - 3.5);
        let p = vec3(a.cos() * r, 0.0, a.sin() * r);
        best = p;
        let blocked = props.iter().any(|q| q.pos.distance(p) < q.r + 1.2);
        if !blocked && horiz(p).distance(horiz(avoid)) > min_avoid {
            return p;
        }
    }
    best
}

fn set_grab(on: bool) {
    set_cursor_grab(on);
    show_mouse(!on);
}

fn load_best() -> Option<f32> {
    std::fs::read_to_string(BEST_FILE).ok()?.trim().parse().ok()
}

fn save_best(t: f32) {
    let _ = std::fs::write(BEST_FILE, format!("{t:.2}"));
}

struct Game {
    state: State,
    player: Player,
    orbs: Vec<Orb>,
    enemies: Vec<Enemy>,
    particles: Vec<Particle>,
    props: Vec<Prop>,
    waves: Vec<Shockwave>,
    motes: Vec<Vec3>,
    collected: u32,
    time: f32,
    t: f32,
    spawn_timer: f32,
    heart_timer: f32,
    shake: f32,
    flash: f32,
    cam_yaw: f32,
    cam_pitch: f32,
    cam_dist: f32,
    cam_target: Vec3,
    last_mouse: Option<Vec2>,
    best: Option<f32>,
    new_best: bool,
    want_quit: bool,
}

impl Game {
    fn new() -> Self {
        let motes = (0..40)
            .map(|_| {
                let a = gen_range(0.0, TAU);
                let r = gen_range(2.0, ARENA_R);
                vec3(a.cos() * r, gen_range(0.8, 5.5), a.sin() * r)
            })
            .collect();
        Game {
            state: State::Title,
            player: Player::new(),
            orbs: vec![],
            enemies: vec![],
            particles: vec![],
            props: make_props(),
            waves: vec![],
            motes,
            collected: 0,
            time: 0.0,
            t: 0.0,
            spawn_timer: 0.0,
            heart_timer: 8.0,
            shake: 0.0,
            flash: 0.0,
            cam_yaw: 0.6,
            cam_pitch: 0.38,
            cam_dist: 22.0,
            cam_target: vec3(0.0, 1.5, 0.0),
            last_mouse: None,
            best: load_best(),
            new_best: false,
            want_quit: false,
        }
    }

    fn start(&mut self) {
        self.player = Player::new();
        self.orbs.clear();
        self.enemies.clear();
        self.particles.clear();
        self.waves.clear();
        self.props = make_props();
        self.collected = 0;
        self.time = 0.0;
        self.spawn_timer = 0.0;
        self.heart_timer = 8.0;
        self.shake = 0.0;
        self.flash = 0.0;
        self.new_best = false;
        self.cam_yaw = 0.0;
        self.cam_pitch = 0.45;
        self.cam_dist = 8.5;
        self.cam_target = self.player.pos + vec3(0.0, 1.3, 0.0);
        self.last_mouse = None;
        self.top_up_orbs();
        self.state = State::Playing;
        set_grab(true);
    }

    fn top_up_orbs(&mut self) {
        loop {
            let normal = self.orbs.iter().filter(|o| !o.heart).count() as u32;
            if normal >= 4 || self.collected + normal >= ORBS_TO_WIN {
                break;
            }
            let pos = random_spot(&self.props, self.player.pos, 7.0);
            self.orbs.push(Orb {
                pos,
                phase: gen_range(0.0, TAU),
                heart: false,
            });
        }
    }

    fn update(&mut self, dt: f32) {
        self.t += dt;
        match self.state {
            State::Title => {
                self.cam_yaw += dt * 0.12;
                self.cam_pitch = 0.38;
                self.cam_dist = 24.0;
                self.cam_target = vec3(0.0, 1.5, 0.0);
                if is_key_pressed(KeyCode::Enter) || is_key_pressed(KeyCode::Space) {
                    self.start();
                }
                if is_key_pressed(KeyCode::Escape) {
                    self.want_quit = true;
                }
            }
            State::Playing => self.update_playing(dt),
            State::Paused => {
                if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::P) {
                    self.state = State::Playing;
                    self.last_mouse = None;
                    set_grab(true);
                }
                if is_key_pressed(KeyCode::Q) {
                    self.want_quit = true;
                }
            }
            State::Won | State::Lost => {
                self.cam_yaw += dt * 0.25;
                let want = self.player.pos + vec3(0.0, 1.3, 0.0);
                self.cam_target = self.cam_target.lerp(want, 1.0 - (-6.0 * dt).exp());
                self.cam_dist += (13.0 - self.cam_dist) * (1.0 - (-2.0 * dt).exp());
                self.shake = (self.shake - dt * 2.0).max(0.0);
                if is_key_pressed(KeyCode::Enter) || is_key_pressed(KeyCode::R) {
                    self.start();
                }
                if is_key_pressed(KeyCode::Escape) {
                    self.want_quit = true;
                }
            }
        }
        if self.state != State::Paused {
            for p in &mut self.particles {
                p.life -= dt;
                p.vel.y -= p.grav * dt;
                p.vel *= 1.0 - 0.9 * dt;
                p.pos += p.vel * dt;
            }
            self.particles.retain(|p| p.life > 0.0);
        }
    }

    fn update_playing(&mut self, dt: f32) {
        self.time += dt;

        if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::P) {
            self.state = State::Paused;
            set_grab(false);
            return;
        }

        // ---- camera orbit input ----------------------------------------------------
        let (mx, my) = mouse_position();
        let mp = vec2(mx, my);
        let delta = self.last_mouse.map(|l| mp - l).unwrap_or(Vec2::ZERO);
        self.last_mouse = Some(mp);
        self.cam_yaw -= delta.x * 0.0035;
        self.cam_pitch += delta.y * 0.0030;
        let turn = 1.9 * dt;
        if is_key_down(KeyCode::Left) || is_key_down(KeyCode::Q) {
            self.cam_yaw += turn;
        }
        if is_key_down(KeyCode::Right) || is_key_down(KeyCode::E) {
            self.cam_yaw -= turn;
        }
        if is_key_down(KeyCode::Up) {
            self.cam_pitch -= turn * 0.6;
        }
        if is_key_down(KeyCode::Down) {
            self.cam_pitch += turn * 0.6;
        }
        let (_, wheel) = mouse_wheel();
        if wheel.abs() > 0.01 {
            self.cam_dist = (self.cam_dist - wheel.signum() * 0.9).clamp(4.5, 16.0);
        }
        self.cam_pitch = self.cam_pitch.clamp(0.08, 1.35);

        // ---- player movement ---------------------------------------------------------
        let f = vec3(-self.cam_yaw.sin(), 0.0, -self.cam_yaw.cos());
        let r = vec3(self.cam_yaw.cos(), 0.0, -self.cam_yaw.sin());
        let mut ix = 0.0;
        let mut iy = 0.0;
        if is_key_down(KeyCode::W) {
            iy += 1.0;
        }
        if is_key_down(KeyCode::S) {
            iy -= 1.0;
        }
        if is_key_down(KeyCode::D) {
            ix += 1.0;
        }
        if is_key_down(KeyCode::A) {
            ix -= 1.0;
        }
        let wish = (f * iy + r * ix).normalize_or_zero();
        let sprinting = (is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift))
            && self.player.stamina > 0.03
            && wish.length() > 0.1;

        {
            let p = &mut self.player;
            if sprinting {
                p.stamina = (p.stamina - 0.38 * dt).max(0.0);
            } else {
                p.stamina = (p.stamina + 0.30 * dt).min(1.0);
            }
            let speed = if sprinting { 10.5 } else { 6.8 };
            let target = wish * speed;
            let rate = if p.on_ground { 14.0 } else { 5.0 };
            let k = 1.0 - (-rate * dt).exp();
            p.vel.x += (target.x - p.vel.x) * k;
            p.vel.z += (target.z - p.vel.z) * k;
            p.vel.y -= 28.0 * dt;

            if is_key_pressed(KeyCode::Space) && p.jumps > 0 {
                p.vel.y = if p.jumps == 2 { 9.5 } else { 8.6 };
                p.on_ground = false;
                let second = p.jumps == 1;
                p.jumps -= 1;
                let c = if second {
                    vec3(0.85, 0.9, 1.0)
                } else {
                    vec3(1.0, 0.95, 0.85)
                };
                emit(&mut self.particles, p.pos + vec3(0.0, 0.1, 0.0), 10, c, 2.2, 0.8, 0.14, 0.5, 2.0);
            }

            p.pos += p.vel * dt;
            if p.pos.y <= 0.0 {
                if p.vel.y < -6.0 {
                    p.land_squash = 0.28;
                    emit(
                        &mut self.particles,
                        p.pos + vec3(0.0, 0.1, 0.0),
                        12,
                        vec3(1.0, 0.95, 0.9),
                        3.0,
                        0.7,
                        0.15,
                        0.5,
                        2.0,
                    );
                }
                p.pos.y = 0.0;
                p.vel.y = 0.0;
                p.on_ground = true;
                p.jumps = 2;
            } else {
                p.on_ground = false;
            }
            p.land_squash = (p.land_squash - dt * 1.6).max(0.0);
            p.inv = (p.inv - dt).max(0.0);
            p.burst_cd = (p.burst_cd - dt).max(0.0);

            let hs = horiz(p.vel).length();
            if hs > 0.8 {
                p.face = lerp_angle(p.face, p.vel.x.atan2(p.vel.z), 1.0 - (-12.0 * dt).exp());
            }
            p.anim += hs * dt * 1.5;

            if sprinting && p.on_ground {
                p.dust -= dt;
                if p.dust <= 0.0 {
                    p.dust = 0.07;
                    emit(
                        &mut self.particles,
                        p.pos + vec3(0.0, 0.1, 0.0),
                        2,
                        vec3(1.0, 0.93, 0.9),
                        1.2,
                        0.6,
                        0.12,
                        0.45,
                        1.0,
                    );
                }
            }

            clamp_arena(&mut p.pos, 0.9);
            push_out_props(&mut p.pos, 0.5, &self.props);
        }

        // ---- shockwave ---------------------------------------------------------------
        if (is_key_pressed(KeyCode::F) || is_mouse_button_pressed(MouseButton::Left))
            && self.player.burst_cd <= 0.0
        {
            self.player.burst_cd = 3.0;
            let c = self.player.pos;
            self.waves.push(Shockwave { t: 0.0, center: c });
            emit(
                &mut self.particles,
                c + vec3(0.0, 0.5, 0.0),
                36,
                vec3(0.78, 0.7, 1.0),
                9.0,
                1.0,
                0.16,
                0.8,
                1.0,
            );
            for e in &mut self.enemies {
                let d = horiz(e.pos - c);
                if d.length() < 9.5 {
                    e.stun = 2.4;
                    let dir = d.normalize_or_zero();
                    e.vel.x = dir.x * 16.0;
                    e.vel.z = dir.z * 16.0;
                }
            }
            self.shake = self.shake.max(0.25);
        }
        for w in &mut self.waves {
            w.t += dt;
        }
        self.waves.retain(|w| w.t < 0.75);

        // ---- orbs --------------------------------------------------------------------
        let ppos = self.player.pos;
        let mut picked_normal = 0;
        let mut healed = false;
        {
            let parts = &mut self.particles;
            self.orbs.retain_mut(|o| {
                let to = horiz(ppos - o.pos);
                let d = to.length();
                if d < 3.8 && d > 0.01 {
                    let pull = (3.8 - d) * 2.6;
                    o.pos += to / d * pull * dt;
                }
                let oy = 1.0;
                if d < 1.45 && (ppos.y + 0.6 - oy).abs() < 1.7 {
                    if o.heart {
                        healed = true;
                        emit(parts, o.pos + vec3(0.0, 1.0, 0.0), 26, vec3(1.0, 0.6, 0.75), 4.5, 3.0, 0.17, 0.9, 2.0);
                    } else {
                        picked_normal += 1;
                        emit(parts, o.pos + vec3(0.0, 1.0, 0.0), 28, vec3(1.0, 0.88, 0.5), 5.0, 3.5, 0.17, 0.9, 2.0);
                    }
                    false
                } else {
                    true
                }
            });
        }
        if healed {
            self.player.hp = (self.player.hp + 1).min(MAX_HP);
        }
        if picked_normal > 0 {
            self.collected += picked_normal;
            self.player.stamina = 1.0;
            if self.collected >= ORBS_TO_WIN {
                self.finish(true);
                return;
            }
            self.top_up_orbs();
        }
        self.heart_timer -= dt;
        if self.heart_timer <= 0.0 {
            self.heart_timer = 12.0;
            if self.player.hp < MAX_HP && !self.orbs.iter().any(|o| o.heart) {
                let pos = random_spot(&self.props, self.player.pos, 6.0);
                self.orbs.push(Orb {
                    pos,
                    phase: gen_range(0.0, TAU),
                    heart: true,
                });
            }
        }

        // ---- enemies -----------------------------------------------------------------
        let time = self.time;
        let collected = self.collected;
        let mut damaged_from: Option<Vec3> = None;
        for e in &mut self.enemies {
            e.age += dt;
            e.stun = (e.stun - dt).max(0.0);
            let to = horiz(ppos - e.pos);
            let dist = to.length();
            let dir = to.normalize_or_zero();
            let side = vec3(-dir.z, 0.0, dir.x);
            let grow = (e.age / 1.5).min(1.0);
            let speed = if e.stun > 0.0 {
                0.0
            } else {
                (2.8 + 0.18 * collected as f32).min(5.2) * grow
            };
            let desired = dir * speed + side * (time * 2.0 + e.phase).sin() * 0.9 * grow;
            let k = 1.0 - (-3.0 * dt).exp();
            e.vel.x += (desired.x - e.vel.x) * k;
            e.vel.z += (desired.z - e.vel.z) * k;
            e.pos.x += e.vel.x * dt;
            e.pos.z += e.vel.z * dt;
            e.pos.y = 0.9 + (time * 2.0 + e.phase).sin() * 0.15;
            push_out_props(&mut e.pos, 0.7, &self.props);
            clamp_arena(&mut e.pos, 0.8);

            if dist < 1.15 && ppos.y < e.pos.y + 0.7 && e.stun <= 0.0 && e.age > 1.0 {
                damaged_from = Some(e.pos);
            }
        }
        // separation
        let n = self.enemies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = horiz(self.enemies[i].pos - self.enemies[j].pos);
                let len = d.length();
                if len < 1.7 && len > 1e-4 {
                    let push = d / len * (1.7 - len) * 0.5;
                    self.enemies[i].pos += push;
                    self.enemies[j].pos -= push;
                }
            }
        }
        if let Some(from) = damaged_from {
            if self.player.inv <= 0.0 {
                self.player.hp -= 1;
                self.player.inv = 1.8;
                let away = horiz(self.player.pos - from).normalize_or_zero();
                self.player.vel = vec3(away.x * 9.0, 6.5, away.z * 9.0);
                self.player.on_ground = false;
                self.shake = 1.0;
                self.flash = 1.0;
                emit(
                    &mut self.particles,
                    self.player.pos + vec3(0.0, 0.8, 0.0),
                    24,
                    vec3(0.6, 0.45, 0.9),
                    5.0,
                    3.0,
                    0.16,
                    0.8,
                    3.0,
                );
                if self.player.hp <= 0 {
                    self.finish(false);
                    return;
                }
            }
        }

        // ---- enemy spawning -----------------------------------------------------------
        let target = if self.time < 3.0 {
            0
        } else {
            (2 + self.collected / 3).min(7) as usize
        };
        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 && self.enemies.len() < target {
            self.spawn_timer = 2.5;
            for _ in 0..30 {
                let a = gen_range(0.0, TAU);
                let pos = vec3(a.cos() * (ARENA_R - 1.5), 0.9, a.sin() * (ARENA_R - 1.5));
                if horiz(pos).distance(horiz(self.player.pos)) > 14.0 {
                    emit(&mut self.particles, pos, 14, vec3(0.6, 0.5, 0.85), 3.0, 1.5, 0.18, 0.9, -0.5);
                    self.enemies.push(Enemy {
                        pos,
                        vel: Vec3::ZERO,
                        phase: gen_range(0.0, TAU),
                        stun: 0.0,
                        age: 0.0,
                    });
                    break;
                }
            }
        }

        // ---- camera follow / effects -------------------------------------------------
        let want = self.player.pos + vec3(0.0, 1.3, 0.0);
        self.cam_target = self.cam_target.lerp(want, 1.0 - (-12.0 * dt).exp());
        self.shake = (self.shake - dt * 2.2).max(0.0);
        self.flash = (self.flash - dt * 2.5).max(0.0);
    }

    fn finish(&mut self, won: bool) {
        self.state = if won { State::Won } else { State::Lost };
        set_grab(false);
        let c = self.player.pos + vec3(0.0, 0.8, 0.0);
        if won {
            emit(&mut self.particles, c, 60, vec3(1.0, 0.88, 0.5), 8.0, 7.0, 0.2, 1.6, 3.0);
            emit(&mut self.particles, c, 40, vec3(1.0, 0.7, 0.85), 7.0, 6.0, 0.18, 1.6, 3.0);
            if self.best.map(|b| self.time < b).unwrap_or(true) {
                self.best = Some(self.time);
                self.new_best = true;
                save_best(self.time);
            }
        } else {
            emit(&mut self.particles, c, 50, vec3(0.6, 0.45, 0.9), 6.0, 4.0, 0.2, 1.4, 2.0);
            self.shake = 1.0;
        }
    }

    fn camera_pos_target(&self) -> (Vec3, Vec3) {
        let dir = vec3(
            self.cam_yaw.sin() * self.cam_pitch.cos(),
            self.cam_pitch.sin(),
            self.cam_yaw.cos() * self.cam_pitch.cos(),
        );
        let s = self.shake * 0.18;
        let shake = vec3(
            (self.t * 53.0).sin() * s,
            (self.t * 71.0).sin() * s,
            (self.t * 47.0).cos() * s,
        );
        (
            self.cam_target + dir * self.cam_dist + shake,
            self.cam_target + shake * 0.5,
        )
    }
}

// ------------------------------------------------------------------------------------
// World rendering
// ------------------------------------------------------------------------------------
fn queue(trans: &mut Vec<(f32, Mesh)>, env: &Env, at: Vec3, m: Mesh) {
    trans.push((at.distance_squared(env.cam), m));
}

fn draw_prop(p: &Prop, env: &Env) {
    let s = p.scale;
    match p.kind {
        0 => {
            draw_mesh(&tube(p.pos, 2.6 * s, 0.28 * s, 0.2 * s, vec3(0.88, 0.74, 0.80), env, 10, 3));
            draw_mesh(&ellipsoid(
                p.pos + vec3(0.0, 3.4 * s, 0.0),
                vec3(1.5, 1.35, 1.5) * s,
                0.0,
                p.hue,
                1.0,
                Mat::Lit,
                env,
                14,
                20,
            ));
            draw_mesh(&ellipsoid(
                p.pos + vec3(0.9 * s, 2.7 * s, 0.5 * s),
                vec3(0.8, 0.7, 0.8) * s,
                0.0,
                p.hue * 0.93,
                1.0,
                Mat::Lit,
                env,
                10,
                14,
            ));
        }
        1 => {
            let rock = vec3(0.80, 0.78, 0.92);
            draw_mesh(&ellipsoid(
                p.pos + vec3(0.0, 0.6 * s, 0.0),
                vec3(1.15, 0.8, 1.05) * s,
                0.5,
                rock,
                1.0,
                Mat::Lit,
                env,
                12,
                18,
            ));
            draw_mesh(&ellipsoid(
                p.pos + vec3(1.05 * s, 0.27 * s, 0.4 * s),
                vec3(0.55, 0.4, 0.5) * s,
                0.0,
                rock * 0.96,
                1.0,
                Mat::Lit,
                env,
                8,
                12,
            ));
        }
        _ => {
            draw_mesh(&tube(p.pos, 1.1 * s, 0.3 * s, 0.22 * s, vec3(1.0, 0.96, 0.9), env, 10, 3));
            draw_mesh(&ellipsoid(
                p.pos + vec3(0.0, 1.25 * s, 0.0),
                vec3(1.15, 0.7, 1.15) * s,
                0.0,
                p.hue,
                1.0,
                Mat::Lit,
                env,
                12,
                18,
            ));
            for k in 0..3 {
                let a = k as f32 * 2.1 + 0.4;
                draw_mesh(&blob(
                    p.pos + vec3(a.cos() * 0.5 * s, 1.75 * s, a.sin() * 0.5 * s),
                    0.14 * s,
                    vec3(1.0, 0.97, 0.92),
                    Mat::Lit,
                    1.0,
                    env,
                    5,
                ));
            }
        }
    }
}

fn render_world(g: &Game, env: &Env, ground: &[Mesh]) {
    let t = g.t;

    // sky + sun
    draw_mesh(&sky_mesh(env));
    let sun = env.cam + env.light * 200.0;
    draw_mesh(&blob(sun, 60.0, vec3(1.0, 0.93, 0.80), Mat::SkyGlow(2.2), 0.45, env, 8));
    draw_mesh(&blob(sun, 24.0, vec3(1.0, 0.96, 0.85), Mat::SkyGlow(1.4), 0.9, env, 8));

    // ground
    for mesh in ground {
        draw_mesh(mesh);
    }

    // feathered shadows & light pools (on the ground, drawn before objects)
    for p in &g.props {
        let big = if p.kind == 0 { 2.3 } else { 1.7 };
        draw_mesh(&shadow(p.pos.x + 0.25, p.pos.z + 0.2, big * p.scale, 0.30, vec3(0.28, 0.2, 0.45)));
    }
    {
        let pl = &g.player;
        let h = pl.pos.y;
        let r = (0.85 - h * 0.08).max(0.45);
        draw_mesh(&shadow(pl.pos.x, pl.pos.z, r, 0.42, vec3(0.28, 0.2, 0.45)));
    }
    for e in &g.enemies {
        draw_mesh(&shadow(e.pos.x, e.pos.z, 1.2, 0.35, vec3(0.25, 0.15, 0.4)));
    }
    for o in &g.orbs {
        let tint = if o.heart { vec3(1.0, 0.6, 0.75) } else { vec3(1.0, 0.85, 0.45) };
        draw_mesh(&shadow(o.pos.x, o.pos.z, 1.7, 0.35, tint));
    }

    let mut trans: Vec<(f32, Mesh)> = Vec::new();

    // boundary lanterns
    let lantern_n = 24;
    for i in 0..lantern_n {
        let a = TAU * i as f32 / lantern_n as f32;
        let base = vec3(a.cos() * (ARENA_R + 0.4), 0.0, a.sin() * (ARENA_R + 0.4));
        draw_mesh(&tube(base, 1.7, 0.08, 0.06, vec3(0.95, 0.88, 0.95), env, 6, 2));
        let top = base + vec3(0.0, 1.85, 0.0);
        let pulse = 0.85 + 0.15 * (t * 1.6 + i as f32 * 0.7).sin();
        let c = if i % 2 == 0 { vec3(1.0, 0.75, 0.88) } else { vec3(0.72, 0.9, 1.0) };
        draw_mesh(&blob(top, 0.17 * pulse, c, Mat::Emissive, 1.0, env, 5));
        queue(&mut trans, env, top, blob(top, 0.75 * pulse, c, Mat::Glow(2.0), 0.55, env, 6));
    }

    // props
    for p in &g.props {
        draw_prop(p, env);
    }

    // player (a little coral blob buddy with a glowing antenna)
    {
        let p = &g.player;
        let flicker = if p.inv > 0.0 && ((p.inv * 14.0) as i32) % 2 == 0 { 0.55 } else { 0.0 };
        let body = vec3(1.0, 0.72, 0.70).lerp(Vec3::ONE, flicker);
        let sq = (p.vel.y * 0.022).clamp(-0.22, 0.28) - p.land_squash;
        let sy = 1.0 + sq;
        let sxz = 1.0 - sq * 0.55;
        let fd = vec3(p.face.sin(), 0.0, p.face.cos());
        let right = vec3(-fd.z, 0.0, fd.x);
        let bob = if p.on_ground { (p.anim * 2.0).sin().abs() * 0.05 } else { 0.0 };
        let bc = p.pos + vec3(0.0, 0.2 + 0.58 * sy + bob, 0.0);
        draw_mesh(&ellipsoid(bc, vec3(0.52 * sxz, 0.6 * sy, 0.52 * sxz), p.face, body, 1.0, Mat::Lit, env, 14, 20));
        // belly
        draw_mesh(&ellipsoid(
            bc + fd * 0.2 * sxz + vec3(0.0, -0.12 * sy, 0.0),
            vec3(0.36 * sxz, 0.38 * sy, 0.36 * sxz),
            0.0,
            vec3(1.0, 0.92, 0.88).lerp(Vec3::ONE, flicker),
            1.0,
            Mat::Lit,
            env,
            10,
            14,
        ));
        // eyes + cheeks
        for side in [-1.0f32, 1.0] {
            let e = bc + fd * 0.43 * sxz + right * side * 0.17 + vec3(0.0, 0.16 * sy, 0.0);
            draw_mesh(&ellipsoid(e, vec3(0.075, 0.105, 0.075), 0.0, vec3(0.2, 0.12, 0.3), 1.0, Mat::Lit, env, 6, 9));
            draw_mesh(&blob(
                e + fd * 0.03 + vec3(0.0, 0.04, 0.0) + right * side * -0.01,
                0.025,
                Vec3::ONE,
                Mat::Emissive,
                1.0,
                env,
                4,
            ));
            let ch = bc + fd * 0.38 * sxz + right * side * 0.3 + vec3(0.0, -0.02 * sy, 0.0);
            draw_mesh(&blob(ch, 0.075, vec3(1.0, 0.55, 0.65), Mat::Lit, 1.0, env, 5));
        }
        // feet
        let hspeed = (horiz(p.vel).length() / 7.0).min(1.5);
        for (k, side) in [-1.0f32, 1.0].iter().enumerate() {
            let ph = p.anim * 2.0 + k as f32 * PI;
            let swing = ph.sin() * 0.22 * hspeed;
            let lift = if p.on_ground {
                ph.cos().max(0.0) * 0.1 * hspeed.min(1.0)
            } else {
                0.12
            };
            let fp = p.pos + right * *side * 0.24 + fd * swing + vec3(0.0, 0.12 + lift, 0.0);
            draw_mesh(&ellipsoid(fp, vec3(0.18, 0.12, 0.24), p.face, vec3(0.95, 0.6, 0.66), 1.0, Mat::Lit, env, 6, 10));
        }
        // antenna
        let top = bc + vec3(0.0, 0.58 * sy, 0.0);
        draw_mesh(&tube(top - vec3(0.0, 0.05, 0.0), 0.38, 0.035, 0.025, vec3(0.95, 0.8, 0.85), env, 6, 2));
        let tip = top + vec3(0.0, 0.46 + sq * 0.2, 0.0);
        draw_mesh(&blob(tip, 0.1, vec3(1.0, 0.92, 0.55), Mat::Emissive, 1.0, env, 6));
        queue(&mut trans, env, tip, blob(tip, 0.45, vec3(1.0, 0.9, 0.5), Mat::Glow(2.0), 0.6, env, 6));
    }

    // enemies ("gloomlings")
    for e in &g.enemies {
        let grow = smoothstep(0.0, 0.8, e.age);
        let stunned = e.stun > 0.0;
        let flick = stunned && ((g.t * 12.0) as i32) % 2 == 0;
        let body = if stunned {
            vec3(0.72, 0.72, 0.95).lerp(vec3(0.5, 0.45, 0.8), if flick { 1.0 } else { 0.0 })
        } else {
            vec3(0.42, 0.33, 0.62)
        };
        let wob = 1.0 + 0.07 * (g.t * 3.0 + e.phase).sin();
        let r = vec3(0.72 * wob, 0.64 / wob, 0.72 * wob) * grow.max(0.05);
        let to = horiz(g.player.pos - e.pos).normalize_or_zero();
        let right = vec3(-to.z, 0.0, to.x);
        draw_mesh(&ellipsoid(e.pos + vec3(0.0, 0.35, 0.0), r, 0.0, body, 1.0, Mat::Lit, env, 12, 18));
        let back = horiz(e.vel).normalize_or_zero();
        for (k, sz) in [0.34f32, 0.2].iter().enumerate() {
            let tp = e.pos + vec3(0.0, 0.2, 0.0) - back * (0.85 + 0.55 * k as f32) * grow;
            draw_mesh(&blob(tp, sz * grow.max(0.05), body * 0.95, Mat::Lit, 1.0, env, 6));
        }
        let eye_col = if stunned { vec3(0.8, 0.85, 1.0) } else { vec3(1.0, 0.9, 0.5) };
        for side in [-1.0f32, 1.0] {
            let ep = e.pos + vec3(0.0, 0.5, 0.0) + to * 0.55 * grow + right * side * 0.22 * grow;
            draw_mesh(&blob(ep, 0.1 * grow.max(0.05), eye_col, Mat::Emissive, 1.0, env, 5));
        }
        queue(
            &mut trans,
            env,
            e.pos,
            blob(e.pos + vec3(0.0, 0.35, 0.0), 1.5 * grow.max(0.05), vec3(0.55, 0.4, 0.9), Mat::Glow(2.2), 0.45, env, 8),
        );
    }

    // orb cores + halos
    for o in &g.orbs {
        let c = o.pos + vec3(0.0, 1.0 + (t * 2.0 + o.phase).sin() * 0.18, 0.0);
        let color = if o.heart { vec3(1.0, 0.55, 0.72) } else { vec3(1.0, 0.86, 0.45) };
        let pulse = 1.0 + 0.08 * (t * 4.0 + o.phase).sin();
        draw_mesh(&blob(c, 0.38 * pulse, color, Mat::Emissive, 1.0, env, 9));
        queue(&mut trans, env, c, blob(c, 0.9 * pulse, color, Mat::Glow(1.8), 0.55, env, 8));
        queue(&mut trans, env, c, blob(c, 1.7 * pulse, color, Mat::Glow(2.6), 0.28, env, 8));
    }

    // ambient motes
    for (i, b) in g.motes.iter().enumerate() {
        let fi = i as f32;
        let p = *b
            + vec3(
                (t * 0.3 + fi).sin() * 1.5,
                (t * 0.5 + fi * 1.7).sin() * 0.8,
                (t * 0.27 + fi * 2.1).cos() * 1.5,
            );
        let c = match i % 3 {
            0 => vec3(1.0, 0.95, 0.75),
            1 => vec3(1.0, 0.8, 0.9),
            _ => vec3(0.8, 0.92, 1.0),
        };
        queue(&mut trans, env, p, blob(p, 0.11, c, Mat::Glow(1.3), 0.75, env, 4));
    }

    // shockwaves
    for w in &g.waves {
        let k = w.t / 0.6;
        let rad = k.min(1.0) * 9.0;
        let fade = (1.0 - k).clamp(0.0, 1.0);
        let c = w.center + vec3(0.0, 0.35, 0.0);
        queue(&mut trans, env, c, torus(c, rad, 0.28, vec3(0.82, 0.74, 1.0), 0.8 * fade, env));
        queue(&mut trans, env, c, torus(c, rad, 0.7, vec3(0.9, 0.8, 1.0), 0.3 * fade, env));
    }

    // particles
    for p in &g.particles {
        let k = (p.life / p.max).clamp(0.0, 1.0);
        queue(
            &mut trans,
            env,
            p.pos,
            blob(p.pos, p.size * (0.4 + 0.6 * k), p.col, Mat::Glow(1.2), 0.9 * k.powf(0.8), env, 4),
        );
    }

    // transparent stuff, far to near
    trans.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, m) in &trans {
        draw_mesh(m);
    }
}

// ------------------------------------------------------------------------------------
// HUD
// ------------------------------------------------------------------------------------
fn text(font: Option<&Font>, s: &str, x: f32, y: f32, size: f32, c: Color) {
    let size = size.round().max(8.0) as u16;
    draw_text_ex(
        s,
        x + 1.5,
        y + 2.0,
        TextParams {
            font,
            font_size: size,
            color: Color::new(1.0, 1.0, 1.0, 0.5 * c.a),
            ..Default::default()
        },
    );
    draw_text_ex(
        s,
        x,
        y,
        TextParams {
            font,
            font_size: size,
            color: c,
            ..Default::default()
        },
    );
}

fn text_w(font: Option<&Font>, s: &str, size: f32) -> f32 {
    measure_text(s, font, size.round().max(8.0) as u16, 1.0).width
}

fn text_c(font: Option<&Font>, s: &str, cx: f32, y: f32, size: f32, c: Color) {
    text(font, s, cx - text_w(font, s, size) * 0.5, y, size, c);
}

fn rrect(x: f32, y: f32, w: f32, h: f32, r: f32, c: Color) {
    let r = r.min(w * 0.5).min(h * 0.5);
    let mut pts: Vec<Vec2> = Vec::new();
    let corners = [
        (x + w - r, y + r, -PI / 2.0),
        (x + w - r, y + h - r, 0.0),
        (x + r, y + h - r, PI / 2.0),
        (x + r, y + r, PI),
    ];
    for (cx, cy, start) in corners {
        for i in 0..=8 {
            let a = start + (PI / 2.0) * i as f32 / 8.0;
            pts.push(vec2(cx + a.cos() * r, cy + a.sin() * r));
        }
    }
    let center = vec2(x + w * 0.5, y + h * 0.5);
    for i in 0..pts.len() {
        draw_triangle(center, pts[i], pts[(i + 1) % pts.len()], c);
    }
}

fn bar(x: f32, y: f32, w: f32, h: f32, frac: f32, fill: Color) {
    rrect(x, y, w, h, h * 0.5, Color::new(0.35, 0.25, 0.55, 0.25));
    let fw = (w * frac.clamp(0.0, 1.0)).max(if frac > 0.0 { h } else { 0.0 });
    if fw > 0.0 {
        rrect(x, y, fw, h, h * 0.5, fill);
    }
}

fn heart(x: f32, y: f32, s: f32, c: Color) {
    draw_circle(x - s * 0.25, y - s * 0.08, s * 0.29, c);
    draw_circle(x + s * 0.25, y - s * 0.08, s * 0.29, c);
    draw_triangle(vec2(x - s * 0.52, y), vec2(x + s * 0.52, y), vec2(x, y + s * 0.62), c);
}

fn fmt_time(t: f32) -> String {
    let m = (t / 60.0) as u32;
    let s = t - m as f32 * 60.0;
    format!("{:02}:{:04.1}", m, s)
}

fn draw_hud(g: &Game, font: Option<&Font>) {
    let sw = screen_width();
    let sh = screen_height();
    let ui = (sh / 720.0).clamp(0.75, 2.0);
    let panel = Color::new(1.0, 1.0, 1.0, 0.40);

    let show_game_hud = matches!(g.state, State::Playing | State::Paused | State::Won | State::Lost);
    if show_game_hud {
        // orbs panel
        let (px, py, pw, ph) = (18.0 * ui, 18.0 * ui, 270.0 * ui, 84.0 * ui);
        rrect(px, py, pw, ph, 20.0 * ui, panel);
        text(font, "ORBS", px + 20.0 * ui, py + 30.0 * ui, 18.0 * ui, Color::new(0.45, 0.35, 0.65, 1.0));
        let s = format!("{} / {}", g.collected, ORBS_TO_WIN);
        text(font, &s, px + 20.0 * ui, py + 62.0 * ui, 34.0 * ui, PLUM);
        bar(
            px + 150.0 * ui,
            py + 44.0 * ui,
            100.0 * ui,
            14.0 * ui,
            g.collected as f32 / ORBS_TO_WIN as f32,
            Color::new(1.0, 0.8, 0.35, 1.0),
        );

        // hearts
        let hp_y = py + ph + 30.0 * ui;
        for i in 0..MAX_HP {
            let c = if i < g.player.hp {
                Color::new(1.0, 0.45, 0.62, 1.0)
            } else {
                Color::new(0.88, 0.84, 0.95, 1.0)
            };
            heart(px + 24.0 * ui + i as f32 * 44.0 * ui, hp_y, 34.0 * ui, c);
        }

        // time
        let ts = fmt_time(g.time);
        let (tw, th) = (170.0 * ui, 52.0 * ui);
        rrect(sw - tw - 18.0 * ui, 18.0 * ui, tw, th, 20.0 * ui, panel);
        text_c(font, &ts, sw - tw * 0.5 - 18.0 * ui, 18.0 * ui + 36.0 * ui, 30.0 * ui, PLUM);

        // stamina + shockwave
        let bw = 240.0 * ui;
        let bx = sw * 0.5 - bw * 0.5;
        rrect(bx - 16.0 * ui, sh - 82.0 * ui, bw + 32.0 * ui, 52.0 * ui, 18.0 * ui, panel);
        bar(bx, sh - 70.0 * ui, bw, 14.0 * ui, g.player.stamina, Color::new(0.45, 0.82, 0.78, 1.0));
        let ready = g.player.burst_cd <= 0.0;
        bar(
            bx,
            sh - 48.0 * ui,
            bw,
            14.0 * ui,
            1.0 - g.player.burst_cd / 3.0,
            if ready {
                Color::new(0.72, 0.58, 1.0, 1.0)
            } else {
                Color::new(0.8, 0.72, 0.95, 1.0)
            },
        );

        text(
            font,
            "WASD move · Space jump (x2) · Shift sprint · Mouse/Arrows/Q E orbit · Wheel zoom · F/Click shockwave",
            18.0 * ui,
            sh - 10.0 * ui,
            15.0 * ui,
            Color::new(0.35, 0.27, 0.55, 0.85),
        );
    }

    let veil = Color::new(0.92, 0.86, 1.0, 0.35);
    let cx = sw * 0.5;
    let cy = sh * 0.5;
    match g.state {
        State::Title => {
            draw_rectangle(0.0, 0.0, sw, sh, Color::new(0.95, 0.88, 1.0, 0.22));
            let (w, h) = (720.0 * ui, 400.0 * ui);
            rrect(cx - w * 0.5, cy - h * 0.5, w, h, 34.0 * ui, Color::new(1.0, 1.0, 1.0, 0.50));
            text_c(font, "LUMEN DRIFT", cx, cy - 100.0 * ui, 84.0 * ui, PLUM);
            text_c(
                font,
                "Gather the glowing orbs. Keep away from the gloom.",
                cx,
                cy - 52.0 * ui,
                24.0 * ui,
                Color::new(0.45, 0.35, 0.65, 1.0),
            );
            let lines = [
                "WASD  move        Space  jump (twice!)        Shift  sprint",
                "Mouse / Arrows / Q E  orbit the camera        Wheel  zoom",
                "F or Click  soft shockwave — stuns the gloomlings",
                "Pink hearts heal you. Collect 20 orbs to win.",
            ];
            for (i, l) in lines.iter().enumerate() {
                text_c(font, l, cx, cy - 8.0 * ui + i as f32 * 30.0 * ui, 20.0 * ui, PLUM);
            }
            let pulse = 0.65 + 0.35 * (g.t * 3.0).sin();
            text_c(font, "Press ENTER to begin", cx, cy + 150.0 * ui, 32.0 * ui, Color::new(0.55, 0.38, 0.85, pulse));
            if let Some(b) = g.best {
                text_c(
                    font,
                    &format!("Best time  {}", fmt_time(b)),
                    cx,
                    cy + 184.0 * ui,
                    20.0 * ui,
                    Color::new(0.45, 0.35, 0.65, 1.0),
                );
            }
        }
        State::Paused => {
            draw_rectangle(0.0, 0.0, sw, sh, veil);
            let (w, h) = (460.0 * ui, 220.0 * ui);
            rrect(cx - w * 0.5, cy - h * 0.5, w, h, 30.0 * ui, Color::new(1.0, 1.0, 1.0, 0.6));
            text_c(font, "Paused", cx, cy - 20.0 * ui, 58.0 * ui, PLUM);
            text_c(font, "Esc / P  resume      Q  quit", cx, cy + 40.0 * ui, 22.0 * ui, PLUM);
        }
        State::Won => {
            draw_rectangle(0.0, 0.0, sw, sh, Color::new(1.0, 0.95, 0.85, 0.25));
            let (w, h) = (560.0 * ui, 280.0 * ui);
            rrect(cx - w * 0.5, cy - h * 0.5, w, h, 30.0 * ui, Color::new(1.0, 1.0, 1.0, 0.6));
            text_c(font, "All orbs gathered!", cx, cy - 40.0 * ui, 52.0 * ui, PLUM);
            text_c(font, &format!("Time  {}", fmt_time(g.time)), cx, cy + 10.0 * ui, 32.0 * ui, PLUM);
            if g.new_best {
                text_c(font, "New best time!", cx, cy + 48.0 * ui, 24.0 * ui, Color::new(0.85, 0.5, 0.2, 1.0));
            } else if let Some(b) = g.best {
                text_c(font, &format!("Best  {}", fmt_time(b)), cx, cy + 48.0 * ui, 22.0 * ui, PLUM);
            }
            text_c(
                font,
                "ENTER  play again      Esc  quit",
                cx,
                cy + 96.0 * ui,
                22.0 * ui,
                Color::new(0.55, 0.38, 0.85, 1.0),
            );
        }
        State::Lost => {
            draw_rectangle(0.0, 0.0, sw, sh, Color::new(0.6, 0.5, 0.85, 0.30));
            let (w, h) = (560.0 * ui, 260.0 * ui);
            rrect(cx - w * 0.5, cy - h * 0.5, w, h, 30.0 * ui, Color::new(1.0, 1.0, 1.0, 0.6));
            text_c(font, "Swallowed by the gloom", cx, cy - 30.0 * ui, 46.0 * ui, PLUM);
            text_c(
                font,
                &format!("Orbs gathered  {} / {}", g.collected, ORBS_TO_WIN),
                cx,
                cy + 18.0 * ui,
                28.0 * ui,
                PLUM,
            );
            text_c(
                font,
                "ENTER / R  try again      Esc  quit",
                cx,
                cy + 84.0 * ui,
                22.0 * ui,
                Color::new(0.55, 0.38, 0.85, 1.0),
            );
        }
        State::Playing => {}
    }
}

// ------------------------------------------------------------------------------------
// Post-processing shader: edge-aware softening + bloom + vignette + dither
// ------------------------------------------------------------------------------------
const POST_VERT: &str = "#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
varying lowp vec2 uv;
varying lowp vec4 color;
uniform mat4 Model;
uniform mat4 Projection;
void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    color = color0 / 255.0;
    uv = texcoord;
}";

const POST_FRAG: &str = "#version 100
precision mediump float;
varying vec2 uv;
uniform sampler2D Texture;
uniform vec2 texel;
uniform float time;
uniform float flash;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    vec3 col = texture2D(Texture, uv).rgb;
    
    vec2 q = uv - 0.5;
    float v = 1.0 - dot(q, q) * 0.9;
    col *= 0.86 + 0.14 * smoothstep(0.0, 1.0, v);
    col = mix(col, vec3(1.0, 0.35, 0.45), flash * 0.35);
    
    gl_FragColor = vec4(col, 1.0);
}";

fn make_target(w: u32, h: u32) -> RenderTarget {
    let rt = render_target_ex(
        w,
        h,
        RenderTargetParams {
            sample_count: 1,
            depth: true,
        },
    );
    rt.texture.set_filter(FilterMode::Linear);
    rt
}

async fn load_font() -> Option<Font> {
    let paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ];
    for p in paths {
        if let Ok(mut f) = load_ttf_font(p).await {
            f.set_filter(FilterMode::Linear);
            return Some(f);
        }
    }
    None
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Lumen Drift".to_owned(),
        window_width: 1200,
        window_height: 700,
        sample_count: 1,
        high_dpi: false,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    srand(macroquad::miniquad::date::now() as u64);
    let font = load_font().await;
    let ground = build_ground();

    let post = load_material(
        ShaderSource::Glsl {
            vertex: POST_VERT,
            fragment: POST_FRAG,
        },
        MaterialParams {
            uniforms: vec![
                UniformDesc::new("texel", UniformType::Float2),
                UniformDesc::new("time", UniformType::Float1),
                UniformDesc::new("flash", UniformType::Float1),
            ],
            ..Default::default()
        },
    )
    .expect("post-process shader failed to compile");

    let mut rt_size = (0u32, 0u32);
    let mut rt = make_target(16, 16);
    let mut g = Game::new();

    loop {
        let dt = get_frame_time().min(0.05);
        g.update(dt);
        if g.want_quit {
            break;
        }

        // (re)create the supersampled MSAA target when the window size changes
        let want = (
            ((screen_width() * SUPERSAMPLE) as u32).max(64),
            ((screen_height() * SUPERSAMPLE) as u32).max(64),
        );
        if want != rt_size {
            rt = make_target(want.0, want.1);
            rt_size = want;
        }

        // ---- 3D pass into the render target -------------------------------------
        let (cam_pos, cam_target) = g.camera_pos_target();
        let env = Env {
            cam: cam_pos,
            light: vec3(-0.5, 0.55, -0.65).normalize(),
        };
        set_camera(&Camera3D {
            position: cam_pos,
            target: cam_target,
            up: Vec3::Y,
            fovy: 55f32.to_radians(),
            render_target: Some(rt.clone()),
            z_near: 0.4,
            z_far: 500.0,
            ..Default::default()
        });
        clear_background(col(FOG, 1.0));
        render_world(&g, &env, &ground);

        // ---- soft post-process to the screen ------------------------------------
        set_default_camera();
        clear_background(col(FOG, 1.0));
        post.set_uniform("texel", vec2(1.0 / rt_size.0 as f32, 1.0 / rt_size.1 as f32));
        post.set_uniform("time", (get_time() % 100.0) as f32);
        post.set_uniform("flash", g.flash);
        gl_use_material(&post);
        draw_texture_ex(
            &rt.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                flip_y: true,
                ..Default::default()
            },
        );
        gl_use_default_material();

        // ---- HUD ----------------------------------------------------------------
        draw_hud(&g, font.as_ref());

        next_frame().await;
    }
}
