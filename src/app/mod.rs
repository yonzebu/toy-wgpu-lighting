use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use glam::*;
use wgpu::{include_wgsl, util::DeviceExt};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod render;
use render::*;
mod util;
use util::*;

const RECT_VERTICES: &[ColoredVert] = &[
    ColoredVert {
        pos: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
    ColoredVert {
        pos: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 1.0],
    },
    ColoredVert {
        pos: [-0.5, 0.5, 0.0],
        color: [1.0, 0.0, 1.0],
    },
    ColoredVert {
        pos: [-0.5, -0.5, 0.0],
        color: [1.0, 1.0, 1.0],
    },
];
const RECT_INDICES: &[u16] = &[2, 1, 0, 1, 2, 3];

const TRIANGLE_VERTICES: &[ColoredVert] = &[
    ColoredVert {
        pos: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 0.0],
    },
    ColoredVert {
        pos: [0.0, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    ColoredVert {
        pos: [-0.5, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
];
const TRIANGLE_INDICES: &[u16] = &[2, 1, 0];

const CUBE_VERTICES: &[ColoredVert] = &[
    // left bottom front
    ColoredVert {
        pos: [-0.5, -0.5, -0.5],
        color: [0.0, 0.0, 0.0],
    },
    // left bottom back
    ColoredVert {
        pos: [-0.5, -0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    // left top front
    ColoredVert {
        pos: [-0.5, 0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    // left top back
    ColoredVert {
        pos: [-0.5, 0.5, 0.5],
        color: [0.0, 1.0, 1.0],
    },
    // right bottom front
    ColoredVert {
        pos: [0.5, -0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    // right bottom back
    ColoredVert {
        pos: [0.5, -0.5, 0.5],
        color: [1.0, 0.0, 1.0],
    },
    // right top front
    ColoredVert {
        pos: [0.5, 0.5, -0.5],
        color: [1.0, 1.0, 0.0],
    },
    // right top back
    ColoredVert {
        pos: [0.5, 0.5, 0.5],
        color: [1.0, 1.0, 1.0],
    },
];
const CUBE_INDICES: &[u16] = &[
    // front
    // ur front
    2, 4, 6,
    // dl front
    4, 2, 0,
    // back
    // ur back
    1, 3, 7,
    // dl back
    7, 5, 1,

    // right
    // ul right
    7, 6, 4,
    // dr right
    4, 5, 7,
    // left
    // ul left
    1, 2, 3,
    // dr left
    2, 1, 0,

    // top
    // ur top
    3, 6, 7,
    // dl top
    6, 3, 2,
    // bottom
    // ur bottom
    5, 4, 0,
    // dl bottom
    0, 1, 5,
];

const VERTICES: &[ColoredVert] = CUBE_VERTICES;
const INDICES: &[u16] = CUBE_INDICES;

const MOVE_SPEED: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Motion {
    Pos,
    Neg,
    None,
}

impl Motion {
    pub fn combine(self, rhs: Self) -> Self {
        match self {
            Motion::Neg => rhs.dec(),
            Motion::None => rhs,
            Motion::Pos => rhs.inc(),
        }
    }

    pub fn inv(self) -> Self {
        match self {
            Motion::Pos => Motion::Neg,
            Motion::None => Motion::None,
            Motion::Neg => Motion::Pos,
        }
    }

    pub fn inc(self) -> Self {
        match self {
            Self::Neg => Self::None,
            _ => Self::Pos,
        }
    }
    pub fn dec(self) -> Self {
        match self {
            Self::Pos => Self::None,
            _ => Self::Neg,
        }
    }

    pub fn as_multiplier(self) -> i32 {
        match self {
            Self::Neg => -1,
            Self::None => 0,
            Self::Pos => 1,
        }
    }
}

pub struct App {
    should_quit: bool,
    last_update: Instant,
    queued_move: [Motion; 3],
    camera: Dirtiable<Camera>,
    basic_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_uniform: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    base: WgpuBase,
}

impl App {
    pub async fn new(window: Window) -> Self {
        let base = WgpuBase::new(window).await;

        let aspect_ratio = base.size.width as f32 / base.size.height as f32;
        let camera = Dirtiable::new(Camera::perspective(
            -Vec3::Z,
            Quat::IDENTITY,
            0.0,
            10.0,
            80_f32.to_radians(),
            aspect_ratio,
        ));
        // let basic_shader = base
        //     .device
        //     .create_shader_module(include_wgsl!("base_buffer.wgsl"));
        // let basic_pipeline_layout =
        //     base.device
        //         .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //             label: Some("Basic pipeline layout"),
        //             bind_group_layouts: &[],
        //             push_constant_ranges: &[],
        //         });
        let camera_shader = base
            .device
            .create_shader_module(include_wgsl!("base_camera.wgsl"));
        let camera_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Camera bind group layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let camera_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Camera pipeline layout"),
                    bind_group_layouts: &[&camera_bind_group_layout],
                    push_constant_ranges: &[],
                });
        let basic_pipeline = base
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Basic render pipeline"),
                layout: Some(&camera_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &camera_shader,
                    entry_point: "vs_main",
                    buffers: &[ColoredVert::LAYOUT],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &camera_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: base.surface_config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Front),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        let vertex_buffer = base
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = base
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });

        let camera_uniform = base
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera uniform buffer"),
                contents: bytemuck::cast_slice(&camera.as_uniform_data()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let camera_bind_group = base.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_uniform,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        Self {
            should_quit: false,
            last_update: Instant::now(),
            queued_move: [Motion::None; 3],
            base,
            camera,
            basic_pipeline,
            vertex_buffer,
            index_buffer,
            camera_uniform,
            camera_bind_group,
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        if self.queued_move[0] != Motion::None {
            let sign = self.queued_move[0].as_multiplier() as f32;
            self.camera
                .modify(|camera| camera.pos.x += sign * MOVE_SPEED * dt);
        }
        if self.queued_move[1] != Motion::None {
            let sign = self.queued_move[1].as_multiplier() as f32;
            self.camera
                .modify(|camera| camera.pos.z += sign * MOVE_SPEED * dt);
        }
        if self.camera.is_dirty() {
            println!(
                "camera moved, rectangle vertices are now:\n\
                \tupper left: {:?}\n\
                \tupper right: {:?}\n\
                \tlower left: {:?}\n\
                \tlower right: {:?}\n",
                self.camera.transform(VERTICES[2].pos.into()),
                self.camera.transform(VERTICES[0].pos.into()),
                self.camera.transform(VERTICES[2].pos.into()),
                self.camera.transform(VERTICES[1].pos.into())
            );
        }
        self.last_update = now;
    }

    /// Prepares a resize but does not execute it (yet). Only one resize can be queued at a
    /// time, and old resizes will be overwritten by new resizes.
    pub fn queue_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.base.queue_resize(new_size)
    }

    // Applies a queued resize if one exists.
    pub fn apply_resize(&mut self) {
        self.base.apply_resize()
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.base.resize(new_size)
    }

    pub fn render(&mut self) {
        // this is the main source of early termination of rendering a frame
        let surface_texture = match self.base.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                self.resize(self.base.size);
                // try to get the texture again but only once
                match self.base.get_current_texture() {
                    Ok(texture) => texture,
                    Err(e) => {
                        eprintln!("{:?}", e);
                        return;
                    }
                }
            }
            Err(e) => {
                eprintln!("{:?}", e);
                return;
            }
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.base
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render encoder"),
                });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.3,
                            g: 0.2,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.basic_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(0..));
            render_pass.set_index_buffer(self.index_buffer.slice(0..), wgpu::IndexFormat::Uint16);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
        }
        self.camera.if_dirty(|camera| {
            self.base.queue.write_buffer(
                &self.camera_uniform,
                0,
                bytemuck::cast_slice(&camera.as_uniform_data()),
            )
        });

        self.base.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }

    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    pub fn process_event(&mut self, event: Event<()>, control_flow: &mut ControlFlow) {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == self.base.window.id() => {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => self.should_quit = true,
                    // can't actually resize here, since winit seems to have a bug where it
                    // sends resize events at startup that don't actually result in anything
                    // besides causing errors for a Vulkan backend
                    WindowEvent::Resized(new_size) => {
                        self.queue_resize(*new_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        self.queue_resize(**new_inner_size);
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                        ..
                    } => {
                        if let Some((mut x_move, mut y_move)) = match keycode {
                            VirtualKeyCode::W => Some((Motion::None, Motion::Pos)),
                            VirtualKeyCode::S => Some((Motion::None, Motion::Neg)),
                            VirtualKeyCode::A => Some((Motion::Neg, Motion::None)),
                            VirtualKeyCode::D => Some((Motion::Pos, Motion::None)),
                            _ => None,
                        } {
                            if *state == ElementState::Released {
                                x_move = x_move.inv();
                                y_move = y_move.inv();
                            }
                            self.queued_move[0] = self.queued_move[0].combine(x_move);
                            self.queued_move[1] = self.queued_move[1].combine(y_move);
                        }
                    }
                    _ => {}
                }
            }
            Event::DeviceEvent { event, .. } => {
                match event {
                    DeviceEvent::MouseMotion { delta } => {
                        self.camera.modify(|camera| {
                            let (x, y, _) = camera
                                .rot
                                // .mul_quat(Quat::from_axis_angle(
                                //     Vec3::Y,
                                //     delta.0.to_radians() as f32,
                                // ))
                                // .mul_quat(Quat::from_axis_angle(
                                //     Vec3::X,
                                //     delta.1.to_radians() as f32,
                                // ))
                                .normalize()
                                .to_euler(EulerRot::XYZ);
                            camera.rot = Quat::from_euler(
                                EulerRot::XYZ,
                                (x + delta.1.to_radians() as f32).clamp(f32::to_radians(-60.0), f32::to_radians(60.0)), 
                                y + delta.0.to_radians() as f32, 
                                0.0
                            );
                        });

                        // println!("mouse moved, camera z is now: {:?}", self.camera.rot.mul_vec3(Vec3::Z));
                        println!(
                            "mouse moved, rectangle vertices are now:\n\
                            \tupper left: {:?}\n\
                            \tupper right: {:?}\n\
                            \tlower left: NOT PRESENT\n\
                            \tlower right: {:?}\n",
                            self.camera.transform(VERTICES[2].pos.into()),
                            self.camera.transform(VERTICES[0].pos.into()),
                            // self.camera.transform(VERTICES[3].pos.into()),
                            self.camera.transform(VERTICES[1].pos.into())
                        );
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(window_id) if window_id == self.base.window.id() => {
                self.update();
                self.render();
                if self.should_quit() {
                    *control_flow = ControlFlow::Exit;
                }
            }
            Event::MainEventsCleared => {
                self.base.window.request_redraw();
            }
            _ => {}
        }
    }
}




fn init_logger() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }
}


pub async fn run() {
    init_logger();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("toy wgpu renderer").build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("toy-renderer")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut app = App::new(window).await;

    event_loop.run(move |event, _, control_flow| {
        app.process_event(event, control_flow);
    })
}
