use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use chrono::{DateTime, Utc};

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
mod utils;
use utils::*;
mod camera;
use camera::*;

const _RECT_VERTICES: &[ColoredVertex] = &[
    ColoredVertex {
        pos: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
    ColoredVertex {
        pos: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 1.0],
    },
    ColoredVertex {
        pos: [-0.5, 0.5, 0.0],
        color: [1.0, 0.0, 1.0],
    },
    ColoredVertex {
        pos: [-0.5, -0.5, 0.0],
        color: [1.0, 1.0, 1.0],
    },
];
const _RECT_INDICES: &[u16] = &[2, 1, 0, 1, 2, 3];

const _TRIANGLE_VERTICES: &[ColoredVertex] = &[
    ColoredVertex {
        pos: [0.5, 0.5, 0.0],
        color: [0.0, 0.0, 0.0],
    },
    ColoredVertex {
        pos: [0.0, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    ColoredVertex {
        pos: [-0.5, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
];
const _TRIANGLE_INDICES: &[u16] = &[2, 1, 0];

const CUBE_VERTICES: &[ColoredVertex] = &[
    // left bottom front
    ColoredVertex {
        pos: [-0.5, -0.5, -0.5],
        color: [0.0, 0.0, 0.0],
    },
    // left bottom back
    ColoredVertex {
        pos: [-0.5, -0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    // left top front
    ColoredVertex {
        pos: [-0.5, 0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    // left top back
    ColoredVertex {
        pos: [-0.5, 0.5, 0.5],
        color: [0.0, 1.0, 1.0],
    },
    // right bottom front
    ColoredVertex {
        pos: [0.5, -0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    // right bottom back
    ColoredVertex {
        pos: [0.5, -0.5, 0.5],
        color: [1.0, 0.0, 1.0],
    },
    // right top front
    ColoredVertex {
        pos: [0.5, 0.5, -0.5],
        color: [1.0, 1.0, 0.0],
    },
    // right top back
    ColoredVertex {
        pos: [0.5, 0.5, 0.5],
        color: [1.0, 1.0, 1.0],
    },
];
const CUBE_INDICES: &[u16] = &[
    // front
    // ur front
    2, 4, 6, // dl front
    4, 2, 0, // back
    // ur back
    1, 3, 7, // dl back
    7, 5, 1, // right
    // ul right
    7, 6, 4, // dr right
    4, 5, 7, // left
    // ul left
    1, 2, 3, // dr left
    2, 1, 0, // top
    // ur top
    3, 6, 7, // dl top
    6, 3, 2, // bottom
    // ur bottom
    5, 4, 0, // dl bottom
    0, 1, 5,
];

const VERTICES: &[ColoredVertex] = CUBE_VERTICES;
const INDICES: &[u16] = CUBE_INDICES;

/// Only one output color target, uses a depth buffer
// fn make_basic_colored_vertex_pipeline_desc<'a>(
//     shader: &'a wgpu::ShaderModule,
//     layout: Option<&'a wgpu::PipelineLayout>,
//     color_targets: &'a [Option<wgpu::ColorTargetState>],
// ) -> wgpu::RenderPipelineDescriptor<'a> {
//     wgpu::RenderPipelineDescriptor {
//         label: Some("Basic colored vertex render pipeline"),
//         layout,
//         vertex: wgpu::VertexState {
//             module: &shader,
//             entry_point: "vs_main",
//             buffers: &[ColoredVertex::LAYOUT],
//         },
//         fragment: Some(wgpu::FragmentState {
//             module: &shader,
//             entry_point: "fs_main",
//             targets: color_targets,
//         }),
//         primitive: wgpu::PrimitiveState {
//             topology: wgpu::PrimitiveTopology::TriangleList,
//             strip_index_format: None,
//             front_face: wgpu::FrontFace::Ccw,
//             cull_mode: Some(wgpu::Face::Back),
//             polygon_mode: wgpu::PolygonMode::Fill,
//             unclipped_depth: false,
//             conservative: false,
//         },
//         depth_stencil: Some(wgpu::DepthStencilState {
//             format: Texture::DEPTH_FORMAT,
//             depth_write_enabled: true,
//             depth_compare: wgpu::CompareFunction::LessEqual,
//             stencil: wgpu::StencilState::default(),
//             bias: wgpu::DepthBiasState::default(),
//         }),
//         multisample: wgpu::MultisampleState {
//             count: 1,
//             mask: !0,
//             alpha_to_coverage_enabled: false,
//         },
//         multiview: None,
//     }
// }

const MOVE_SPEED: f32 = 1.0;
const COLORED_CUBE_ID: usize = 0;
const TRIANGLE_ID: usize = 1;
const RECT_ID: usize = 2;

pub struct App {
    should_quit: bool,
    last_update: DateTime<Utc>,
    camera: BasicCameraController,
    // colored_vertex_pipeline: wgpu::RenderPipeline,
    // model_pipeline: wgpu::RenderPipeline,
    // vertex_buffer: wgpu::Buffer,
    // index_buffer: wgpu::Buffer,
    // camera_uniform: wgpu::Buffer,
    // camera_bind_group: wgpu::BindGroup,
    z_buffer: Texture,
    objects: HashMap<usize, RenderObject>,
    base: WgpuBase,
}

impl App {
    pub async fn new(window: Window) -> Self {
        let base = WgpuBase::new(window).await;

        let aspect_ratio = base.size.width as f32 / base.size.height as f32;
        let camera = Camera::perspective(
            -Vec3::Z,
            Quat::IDENTITY,
            0.0,
            1.0,
            70_f32.to_radians(),
            aspect_ratio,
        );
        // let camera = Camera::orthographic(
        //     -Vec3::Z,
        //     Quat::IDENTITY,
        //     0.0,
        //     vec3(2.0*aspect_ratio, 2.0, 2.0)
        // );
        let camera = BasicCameraController::new(camera, MOVE_SPEED * Vec3::ONE, 1.0);
        let colored_vertex_shader = base
            .device
            .create_shader_module(include_wgsl!("colored_vertex.wgsl"));
        let model_vertex_shader = base
            .device
            .create_shader_module(include_wgsl!("model_vertex.wgsl"));
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
        let model_mat_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Model matrix bind group layout"),
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
        let colored_vertex_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Colored vertex pipeline layout"),
                    bind_group_layouts: &[&model_mat_bind_group_layout],
                    push_constant_ranges: &[],
                });
        let colored_vertex_pipeline = Arc::new(base.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Colored vertex render pipeline"),
                layout: Some(&colored_vertex_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &colored_vertex_shader,
                    entry_point: "vs_main",
                    buffers: &[ColoredVertex::LAYOUT],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &colored_vertex_shader,
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
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            },
        ));

        let model_vertex_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Model vertex pipeline layout"),
                    bind_group_layouts: &[&model_mat_bind_group_layout],
                    push_constant_ranges: &[],
                });
        let model_vertex_pipeline = Arc::new(base.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Model vertex render pipeline"),
                layout: Some(&model_vertex_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &model_vertex_shader,
                    entry_point: "vs_main",
                    buffers: &[ModelVertex::LAYOUT],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &model_vertex_shader,
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
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            },
        ));

        let colored_cube_model_uniform =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera uniform buffer"),
                    // this is fine because initial transform is the identity transform
                    contents: bytemuck::cast_slice(&camera.as_uniform_data()),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let colored_cube = RenderObject {
            vertex_buffer: Arc::new(base.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Colored cube vertex buffer"),
                    contents: bytemuck::cast_slice(CUBE_VERTICES),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            )),
            num_vertices: CUBE_VERTICES.len(),
            index_buffer: Arc::new(base.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Colored cube index buffer"),
                    contents: bytemuck::cast_slice(CUBE_INDICES),
                    usage: wgpu::BufferUsages::INDEX,
                },
            )),
            num_indices: CUBE_INDICES.len(),
            pipeline: colored_vertex_pipeline.clone(),
            model_bind_group: base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Camera bind group"),
                layout: &model_mat_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: colored_cube_model_uniform.as_entire_binding(),
                }],
            }),
            model_uniform: colored_cube_model_uniform,
            transform: Transform::default().into(),
            render_hook: None,
        };

        // let camera_uniform = base
        //     .device
        //     .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //         label: Some("Camera uniform buffer"),
        //         contents: bytemuck::cast_slice(&camera.as_uniform_data()),
        //         usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        //     });

        let z_buffer = Texture::create_depth_texture(&base, false);

        let objects = HashMap::from([(COLORED_CUBE_ID, colored_cube)]);

        Self {
            should_quit: false,
            last_update: Utc::now(),
            camera,
            objects,
            // colored_vertex_pipeline,
            // vertex_buffer,
            // index_buffer,
            // camera_uniform,
            // camera_bind_group,
            z_buffer,
            base,
        }
    }

    pub fn update(&mut self) {
        let now = Utc::now();
        let dt = now
            .signed_duration_since(self.last_update)
            .num_nanoseconds()
            .unwrap() as f32
            / 1_000_000_000.0;

        self.camera.update_with_dt(dt);

        self.last_update = now;
    }

    /// Prepares a resize but does not execute it (yet). Only one resize can be queued at a
    /// time, and old resizes will be overwritten by new resizes.
    pub fn queue_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.base.queue_resize(new_size)
    }

    // Applies a queued resize if one exists.
    pub fn apply_resize(&mut self) {
        self.base.apply_resize();
        self.z_buffer = Texture::create_depth_texture(&self.base, self.z_buffer.sampler.is_some());
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.queue_resize(new_size);
        self.apply_resize();
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
                // depth_stencil_attachment: None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.z_buffer.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            for object in self.objects.values_mut() {
                if let Some(mut render_hook) = object.render_hook.take() {
                    // this is kinda just hacking around the borrow checker
                    // not breaking any rules though! 
                    // it's just maybe a tiny bit of a footgun to structure code like this
                    render_hook(object, &mut render_pass, &self.base.queue);
                    object.render_hook = Some(render_hook);
                } else {
                    render_pass.set_pipeline(&object.pipeline);
                    render_pass.set_vertex_buffer(0, object.vertex_buffer.slice(0..));
                    render_pass.set_index_buffer(
                        object.index_buffer.slice(0..),
                        wgpu::IndexFormat::Uint16,
                    );
                    render_pass.set_bind_group(0, &object.model_bind_group, &[]);
                    render_pass.draw_indexed(0..object.num_indices as u32, 0, 0..1);
                }
                if self
                    .camera
                    .if_dirty(|camera| object.update_uniforms(camera, true, &self.base.queue))
                    .is_none()
                {
                    object.update_uniforms(&self.camera, false, &self.base.queue)
                }
            }

            // render_pass.set_pipeline(&self.colored_vertex_pipeline);
            // render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(0..));
            // render_pass.set_index_buffer(self.index_buffer.slice(0..), wgpu::IndexFormat::Uint16);
            // render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            // render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
        }

        self.base.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }

    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    pub fn process_event(&mut self, event: Event<()>, control_flow: &mut ControlFlow) {
        *control_flow = ControlFlow::Poll;
        // process only the window events that apply to this window, but all other events pass through
        if let &Event::WindowEvent { window_id, .. } = &event {
            if window_id == self.base.window.id() {
                self.camera.process_event(&event);
            }
        } else {
            self.camera.process_event(&event);
        }
        // now do other event processing
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
    let window = WindowBuilder::new()
        .with_title("toy wgpu renderer")
        .build(&event_loop)
        .unwrap();

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
