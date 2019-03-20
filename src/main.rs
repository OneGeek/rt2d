use rand::prelude::*;
use cgmath::prelude::*;

use cgmath::{Vector2, Point2, Matrix2, vec2};
use collision::{Line2, Ray2, Ray, Continuous};

use graphics;
mod gfx_prelude {
    use gfx_device_gl::Resources;
    use gfx_graphics::GfxGraphics;

    pub use graphics::Graphics;

    pub use gfx_graphics::TextureSettings;

    pub type Window = glutin_window::GlutinWindow;
    pub use piston::window::OpenGLWindow;


    pub type Device = gfx_device_gl::Device;

    pub type Factory = gfx_device_gl::Factory;
    pub type CommandBuffer = gfx_device_gl::CommandBuffer;
    pub type Context = graphics::Context;

    pub type Encoder = gfx::Encoder<Resources, CommandBuffer>;

    pub type Gfx2d = gfx_graphics::Gfx2d<Resources>;

    pub type G2d<'a> = GfxGraphics<'a, Resources, CommandBuffer>;

    pub type DepthStencilView = gfx::handle::DepthStencilView<Resources, gfx::format::DepthStencil>;
    pub type RenderTargetView = gfx::handle::RenderTargetView<Resources, gfx::format::Srgba8>;

    pub type RenderBuffer = graphics_buffer::RenderBuffer;
    pub use graphics_buffer::IDENTITY;
}

use gfx_prelude::*;
use piston::input;
use piston::window::Window as PistonWindowTrait; // gives .draw_size()
use glutin_window::GlutinWindow;
use piston_window::{PistonWindow, Glyphs};


use find_folder;
use snowflake::ProcessUniqueId;

const GL_V: glutin_window::OpenGL = glutin_window::OpenGL::V3_2;
const SAMPLES: u8 = 4;

struct DrawSystem {
    frame: i64,
    encoder: Encoder,
    g2d: Gfx2d,
    device: Device,
    factory: Factory,
    output_color: RenderTargetView,
    output_stencil: DepthStencilView,
    light_buffer: RenderBuffer,
    glyphs: Glyphs,
}

use graphics::{Graphics, ImageSize, DrawState};
use graphics::math::Matrix2d;
use graphics::character::CharacterCache;

impl DrawSystem {

    fn create_color_and_stencil(window: &Window) -> (RenderTargetView, DepthStencilView) {
        use gfx::{
            format::{DepthStencil, Formatted, Srgba8},
            memory::Typed,
        };
        use piston::window::Window;

        let draw_size = window.draw_size();
        let aa = SAMPLES as gfx::texture::NumSamples;

        let dim = (
            draw_size.width as u16,
            draw_size.height as u16,
            1,
            aa.into(),
        );

        let color_format = <Srgba8 as Formatted>::get_format();
        let depth_format = <DepthStencil as Formatted>::get_format();

        let (output_color, output_stencil) =
            gfx_device_gl::create_main_targets_raw(dim, color_format.0, depth_format.0);

        let output_color = Typed::new(output_color);
        let output_stencil = Typed::new(output_stencil);

        (output_color, output_stencil)
    }

    pub fn new(window: &mut Window) -> Self {
        let frame = 0i64;
        let (device, mut factory) =
            gfx_device_gl::create(|s| window.get_proc_address(s) as *const std::os::raw::c_void);

        let g2d = Gfx2d::new(GL_V, &mut factory);

        let (output_color, output_stencil) = Self::create_color_and_stencil(&*window);

        let encoder = factory.create_command_buffer().into();

        let size = window.draw_size();
        let light_buffer = RenderBuffer::new(size.width as u32, size.height as u32);

        let assets = find_folder::Search::ParentsThenKids(3, 3)
            .for_folder("assets").expect("Failed to find assets");

        let font = assets.join("FiraSans-Regular.ttf");
        let glyphs = Glyphs::new(&font, factory.clone(), TextureSettings::new())
            .expect(&(String::from("Failed to load ") + font.to_str().unwrap()));

        DrawSystem { frame, device, factory, encoder, g2d, glyphs, output_color, output_stencil, light_buffer }
    }

    pub fn draw(&mut self, state: &mut MainState, args: input::RenderArgs) {
        use graphics::Graphics;

        let DrawSystem {
            mut frame,
            ref mut encoder,
            ref mut g2d,
            ref mut glyphs,
            ref output_color,
            ref output_stencil,
            ref mut device,
            ref mut factory,
            ref mut light_buffer
        } = self;

        if state.clear_light {
            light_buffer.clear([0.0,0.0,0.0,1.0]);
            state.clear_light = false;

        }

        for n in 0..10 {
            gen_ray(&state.light_position).draw(state, IDENTITY, light_buffer);
        }

        let light_texture =
            light_buffer.to_g2d_texture(factory,&TextureSettings::new())
                .expect("Failed to render to light texture");

        g2d.draw(encoder, output_color, output_stencil, args.viewport(),
            |c, g| {
                g.clear_color([0.0,0.0,0.0,1.0]);

                graphics::image(&light_texture, c.transform, g);


                for &wall in state.scene.walls.iter() {
                    wall.draw(state, c.transform, g);
                }
            },
        );

        encoder.flush(device);

        frame = frame + 1;
    }
}

type Color = [f32; 4];

trait Drawable {
    fn draw<G, T>(&self, state: &MainState, transform: Matrix2d, g: &mut G) where G: Graphics<Texture = T>, T: ImageSize;
}

#[derive(Debug, Clone, Copy)]
struct LightRay {
    origin: CgPoint,
    direction: CgVec,
    color: Color,
    max_bounces: i32,
    no_collide: Option<ProcessUniqueId>,
    current_media_refractive_index: f32,
}

type CollisionRay = collision::Ray2<f32>;
type CollisionLine = collision::Line2<f32>;
type CgPoint = cgmath::Point2<f32>;
type CgVec = cgmath::Vector2<f32>;

#[derive(Debug, Clone, Copy)]
struct RtLine {
    a: CgPoint,
    b: CgPoint
}

impl RtLine {
    fn new((ax, ay): (f32, f32), (bx, by): (f32, f32)) -> Self {
        RtLine{ a: CgPoint{x:ax, y:ay}, b: CgPoint{x:bx, y:by} }
    }
    fn collidable(&self) -> CollisionLine {
        Line2::new(self.a, self.b)
    }

    fn length(&self) -> f32 {
        self.a.distance(self.b)
    }

    fn normalized(&self) -> RtLine {
        let len = self.length();
        RtLine::new((0.0, 0.0), ((self.b.x - self.a.x) / len, (self.b.y - self.a.y) / len))
    }

    fn normal(&self) -> CgVec {
        let normalized = self.normalized();
        vec2(-normalized.b.y, normalized.b.x)
    }
}

impl LightRay {
    fn collidable(&self) -> CollisionRay {
        Ray::new(self.origin, self.direction)
    }

    fn intersection(&self, wall: &Wall) -> Option<CgPoint> {
        self.collidable().intersection(&wall.collidable())
    }
}


impl Drawable for LightRay {
    fn draw<G, T>(&self, state: &MainState, transform: Matrix2d, g: &mut G) where G: Graphics<Texture = T>, T: ImageSize {

        let mut point_hit: Option<CgPoint> = None;
        let mut surface_hit: Option<Wall> = None;

        for &wall in &state.scene.walls {
            let skip = self.no_collide.map_or(false, |id| wall.id.eq(&id));
            if !skip {
                point_hit = self.intersection(&wall).map_or(point_hit, |new_point_hit| {
                    point_hit.map_or(Some(new_point_hit), |old_point_hit| {
                        let new_distance = RtLine { a: self.origin, b: new_point_hit }.length();
                        let old_distance = RtLine { a: self.origin, b: old_point_hit }.length();

                        if new_distance < old_distance {
                            surface_hit = Some(wall);
                            Some(new_point_hit)
                        } else {
                            Some(old_point_hit)
                        }
                    })
                });
            }
        }
        point_hit.map(|point| {
            draw_line(RtLine { a: self.origin, b: point }, self.color, 0.5, state, transform, g);

            surface_hit.map(|surface| {
                if surface.reflected_fraction > 0.0 && self.max_bounces > 0 {
                    let surface_normal = surface.line.normal();
                    let cos_incident = cgmath::dot(self.direction, surface_normal);

                    LightRay {
                        origin: point,
                        direction: vec2(
                            self.direction.x - (2.0 * surface_normal.x * cos_incident),
                            self.direction.y - (2.0 * surface_normal.y * cos_incident)
                        ),
                        color: [self.color[0], self.color[1], self.color[2], self.color[3] * surface.reflected_fraction],
                        max_bounces: self.max_bounces - 1,
                        no_collide: Some(surface.id),
                        current_media_refractive_index: self.current_media_refractive_index,
                    }.draw(state, transform, g);

                    debug_direction(point, vec2(-self.direction.x, -self.direction.y), [1.0, 0.0, 1.0, 1.0], state, transform, g);
                    debug_direction(point, surface_normal, [0.0, 1.0, 1.0, 1.0], state, transform, g);
                }

                if surface.refracted_fraction > 0.0 && self.max_bounces > 0 {
                    let mut surface_normal = surface.line.normal();
                    let mut cos_incident = cgmath::dot(self.direction, surface_normal);

                    if cos_incident < 0.0 {
                        surface_normal.x = -surface_normal.x;
                        surface_normal.y = -surface_normal.y;
                        cos_incident = cgmath::dot(self.direction, surface_normal);
                    }

                    let r = self.current_media_refractive_index / surface.refractive_index;
                    let c = cos_incident;
                    let n = surface_normal;
                    let q = self.direction;

                    use std::f64;
                    let inner_term = (r * c) * ((1.0 - ((r*r) * (1.0 - (c*c))) as f64).sqrt()) as f32;
                    LightRay {
                        origin: point,
                        direction: vec2(
                            (r * q.x) + (inner_term * n.x),
                            (r * q.y) + (inner_term * n.y)
                        ),
                        color: [self.color[0], self.color[1], self.color[2], self.color[3] * surface.refracted_fraction],
                        max_bounces: self.max_bounces - 1,
                        no_collide: Some(surface.id),
                        current_media_refractive_index: surface.refractive_index,
                    }.draw(state, transform, g);
                }
            });
        });
    }
}

fn debug_direction<G, T>(point: CgPoint, direction: CgVec, color: Color, state: &MainState, transform: Matrix2d, g: &mut G)
    where G: Graphics<Texture = T>, T: ImageSize {

    let debug_line = RtLine::new(
        (point.x, point.y),
        (point.x + (direction.x * 50.0), point.y + (direction.y * 50.0))
    );

    //draw_line(debug_line, color, 1.5, state, transform, g);
}


#[derive(Debug, Clone, Copy)]
struct Wall {
    id: ProcessUniqueId,
    line: RtLine,
    refractive_index: f32,
    reflected_fraction: f32,
    refracted_fraction: f32,
    color: Color,
}

impl Wall {

    fn new(line: RtLine, color: Color) -> Self {
        let id = ProcessUniqueId::new();
        let is_reflective = false;
        Wall{id, line, color, reflected_fraction: 0.0, refracted_fraction: 0.0, refractive_index: 0.0}
    }

    fn red(line: RtLine) -> Self {
        Wall::new(line, [1.0, 0.0, 0.0, 1.0])
    }

    fn mirror(line: RtLine) -> Self {
        let id = ProcessUniqueId::new();
        let color = [0.7, 0.7, 0.7, 1.0];
        Wall{id, line, color, reflected_fraction: 1.0, refracted_fraction: 0.0, refractive_index: 0.0}
    }

    fn glass(line: RtLine) -> Self {
        let id = ProcessUniqueId::new();
        let color = [0.6, 0.6, 1.0, 1.0];
        Wall{id, line, color, reflected_fraction: 0.3, refracted_fraction: 0.7, refractive_index: 1.5}
    }

    fn collidable(&self) -> CollisionLine {
        self.line.collidable()
    }

}

impl Drawable for Wall {
    fn draw<G, T>(&self, state: &MainState, transform: Matrix2d, g: &mut G) where G: Graphics<Texture = T>, T: ImageSize {
        draw_line(self.line, self.color, 1.0, state, transform, g);

        let normalized = self.line.normalized();

        debug_direction(self.line.a, vec2(normalized.b.x, normalized.b.y), [0.0, 1.0, 0.0, 1.0], state, transform, g);
        debug_direction(self.line.a, self.line.normal(), [1.0, 0.0, 1.0, 1.0], state, transform, g);
    }
}

fn draw_line<G, T>(line: RtLine, color: Color, radius: f32, state: &MainState, transform: Matrix2d, g: &mut G)  where G: Graphics<Texture = T>, T: ImageSize {
    graphics::line(
        color, radius as f64,
        [line.a.x as f64, line.a.y as f64, line.b.x as f64, line.b.y as f64],
        transform, g
    );
}

fn gen_ray(p: &Point2<f32>) -> LightRay {
    let rnd1: f32 = rand::random();
    let rnd2: f32 = rand::random();

    let direction = RtLine::new((0.0, 0.0), (rnd1 - 0.5, rnd2 - 0.5)).normalized();

    LightRay {
        origin: *p,
        direction: vec2(direction.b.x, direction.b.y),
        color: [1.0, 1.0, 1.0, 0.1],
        max_bounces: 4,
        current_media_refractive_index: 1.0,
        no_collide: None
    }
}

fn make_walls(width: f32, height: f32) -> Vec<Wall> {
    let bottom_left = Point2 { x: 0.0, y: 0.0 };
    let bottom_right = Point2 { x: width, y: 0.0 };
    let top_left = Point2 { x: 0.0, y: height };
    let top_right = Point2 { x: width, y: height };

    let top_wall = RtLine{a: top_left, b: top_right};
    let right_wall = RtLine{a: top_right, b: bottom_right};
    let bottom_wall = RtLine{a: bottom_left, b: bottom_right};
    let left_wall = RtLine{a: top_left, b: bottom_left};

    let mut walls: Vec<Wall> = Vec::new();
    let red = [1.0,0.0,0.0,1.0];
    let green = [0.0,1.0,0.0,1.0];
    walls.push(Wall::red(top_wall));
    walls.push(Wall::red(right_wall));
    walls.push(Wall::red(bottom_wall));
    walls.push(Wall::red(left_wall));
    walls.push(Wall::red(RtLine::new(
       (width * 0.4, height * 0.25),
       (width * 0.45, height * 0.4)
    )));
    walls.push(Wall::red(RtLine::new(
        (width * 0.3, height * 0.5),
        (width * 0.5, height * 0.5)
    )));
    walls.push(Wall::red(RtLine::new(
        (width * 0.95, height * 1.0),
        (width * 0.95, height * 0.85)
    )));
    walls.push(Wall::mirror(RtLine::new(
        (width * 0.53, height * 0.5),
        (width * 0.8, height * 0.5)
    )));
    walls.push(Wall::glass(RtLine::new(
        (width * 0.60, height * 0.65),
        (width * 0.75, height * 0.65)
    )));

    walls
}

struct Scene {
    walls: Vec<Wall>
}

impl Scene {
    pub fn new() -> Self {
        let walls = make_walls(1024.0, 500.0);
        Scene { walls }
    }
}

struct MainState {
    scene: Scene,
    light_position: Point2<f32>,
    mouse_cursor: Point2<f32>,
    clear_light: bool
}

impl MainState {
    fn new() -> Self {
        let scene = Scene::new();
        let light_position = Point2{x:10.0, y: 10.0};
        let mouse_cursor = Point2{x:10.0, y: 10.0};
        let clear_light = true;
        MainState{scene, light_position, mouse_cursor, clear_light}
    }
}

fn main() {

    let mut state = MainState::new();

    let (width, height) = (1024, 500);

    let mut window: PistonWindow<GlutinWindow> = piston::window::WindowSettings::new("rt2d", [width,height])
        .exit_on_esc(true)
        .samples(SAMPLES)
        .vsync(true)
        .build()
        .expect("Failed to create window");

    let mut draw_system = DrawSystem::new(&mut window.window);


    while let Some(e) = window.next() {
        // Handle input
        use piston::input::{Button, ReleaseEvent, MouseCursorEvent, RenderEvent};
        use piston::input::keyboard::Key;
        use piston::input::mouse::MouseButton;

        if let Some(Button::Keyboard(Key::Space)) = e.release_args() {
            state.clear_light = true;
        }

        if let Some(Button::Mouse(MouseButton::Left)) = e.release_args() {
            state.clear_light = true;
            state.light_position = state.mouse_cursor.clone();
        }

        if let Some(pos) = e.mouse_cursor_args() {
            //state.clear_light = true;
            state.mouse_cursor = Point2 { x: pos[0] as f32, y: pos[1] as f32};
        }

        if let Some(args) = e.render_args() {
            draw_system.draw(&mut state, args);
        }
    }
}
