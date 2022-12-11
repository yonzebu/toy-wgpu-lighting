use std::{collections::HashMap, sync::Arc};

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

mod camera;
use camera::{BasicCameraController, Camera};
mod render;
use render::*;
mod utils;
use utils::*;
mod resources;
use resources::*;

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

const _CUBE_VERTICES: &[ColoredVertex] = &[
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
const _CUBE_INDICES: &[u16] = &[
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

const MOVE_SPEED: f32 = 1.0;
// const COLORED_CUBE_ID: usize = 0;
// const TRIANGLE_ID: usize = 1;
// const RECT_ID: usize = 2;
// const MODEL_CUBE_ID: usize = 3;
const WHITE_CUBES_ID: usize = 4;
const BLACK_CUBES_ID: usize = 5;
const WHITE_QUEENS_ID: usize = 6;
const BLACK_QUEENS_ID: usize = 7;

const CELL_WIDTH: f32 = 0.1;
const CELL_HEIGHT: f32 = 0.05;
const BOARD_WIDTH: usize = 8;
const QUEEN_SCALE: Vec3 = vec3(0.04, 0.04, 0.04);

const INITIAL_LIGHT_POS: Vec3 = vec3(0.0, 1.0, -1.0);

fn board_x_to_world_x(x: f32) -> f32 {
    let offset = (BOARD_WIDTH / 2) as f32 - 0.5;
    (x - offset) * CELL_WIDTH
}
// it's literally the same thing
fn board_y_to_world_y(y: f32) -> f32 {
    let offset = (BOARD_WIDTH / 2) as f32 - 0.5;
    (y - offset) * CELL_WIDTH
}

pub struct App {
    should_quit: bool,
    last_update: DateTime<Utc>,
    camera: BasicCameraController,
    empty_bind_group: wgpu::BindGroup,
    light: PointLight,
    // pov: you are a point light
    light_camera: Dirtiable<Camera>,
    /// Contains light info (only uses view/projection matrix though)
    shadow_pass_bind_group: wgpu::BindGroup,
    /// Contains camera info, light info and shadow texture/sampler
    standard_pass_bind_group: wgpu::BindGroup,
    /// Contains camera info
    camera_uniform: wgpu::Buffer,
    /// Contains light info
    light_uniform: wgpu::Buffer,
    z_buffer: render::Texture,
    shadow_z_buffer: render::Texture,
    objects: HashMap<usize, ShadowedRenderObject>,
    keystates: HashMap<VirtualKeyCode, bool>,
    base: WgpuBase,
}

impl App {
    pub async fn new(window: Window) -> Self {
        let base = WgpuBase::new(window).await;

        let aspect_ratio = base.size.width as f32 / base.size.height as f32;
        let camera = Camera::perspective(
            -Vec3::Z,
            Quat::IDENTITY,
            0.01,
            100.0,
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

        let light_camera: Dirtiable<Camera> = Camera {
            pos: INITIAL_LIGHT_POS,
            rot: Quat::IDENTITY,
            near: 0.01,
            far: 100.0,
            proj: camera::Projection::Perspective {
                fov_y: 120_f32,
                aspect_ratio: 1.0,
            },
        }
        .look_at(Vec3::Z)
        .into();
        let light = render::PointLight {
            view_proj: light_camera.view_proj_matrix().to_cols_array(),
            pos: INITIAL_LIGHT_POS.into(),
            specular_color: [0.5, 0.5, 0.5],
            specular_power: 5.0,
            ambient_power: 0.1,
            exponent: 50.0,
        };

        let shadow_z_buffer = render::Texture::create_shadow_texture(
            &base,
            Some((
                2 * base.surface_config.width,
                2 * base.surface_config.height,
            )),
        );
        let camera_uniform = base
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera uniform buffer"),
                contents: bytemuck::cast_slice(&camera.as_uniform_data()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let mut raw_light_bytes = [0_u8; round_size_of_up_to_align::<render::PointLight>(16)];
        raw_light_bytes[0..std::mem::size_of::<render::PointLight>()]
            .copy_from_slice(bytemuck::cast_slice(&[light]));
        let light_uniform = base
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Light uniform buffer"),
                contents: &raw_light_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let model_instanced_shader = base
            .device
            .create_shader_module(include_wgsl!("model_instanced.wgsl"));
        let empty_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Empty bind group layout"),
                    entries: &[],
                });
        let standard_pass_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Standard pass bind group layout"),
                    entries: &[
                        // camera uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // point light uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // shadow texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // shadow sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
        let shadow_pass_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Shadow pass bind group layout"),
                    entries: &[
                        // camera uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // point light uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let model_instanced_bind_group_layout =
            base.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Model instanced bind group layout"),
                    entries: &[
                        // texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // texture sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let empty_bind_group = base.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Empty bind group"),
            layout: &empty_bind_group_layout,
            entries: &[],
        });
        let standard_pass_bind_group = base.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Standard pass bind group"),
            layout: &standard_pass_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &camera_uniform,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &light_uniform,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_z_buffer.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&shadow_z_buffer.sampler),
                },
            ],
        });
        let shadow_pass_bind_group = base.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow pass bind group"),
            layout: &shadow_pass_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &camera_uniform,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &light_uniform,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let model_instanced_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Model instanced pipeline layout"),
                    bind_group_layouts: &[
                        &empty_bind_group_layout,
                        &empty_bind_group_layout,
                        &standard_pass_bind_group_layout,
                        &model_instanced_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let model_instanced_pipeline = Arc::new(base.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Model instanced render pipeline"),
                layout: Some(&model_instanced_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &model_instanced_shader,
                    entry_point: "vs_main",
                    buffers: &[
                        render::ModelVertex::LAYOUT,
                        render::ShadowedModelInstanced::INSTANCE_LAYOUT,
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &model_instanced_shader,
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
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: Some(wgpu::Face::Back),
                    // cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                // depth_stencil: None,
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: render::Texture::DEPTH_FORMAT,
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
        let model_instanced_shadow_pipeline_layout =
            base.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Model vertex pipeline layout"),
                    bind_group_layouts: &[
                        &empty_bind_group_layout,
                        &empty_bind_group_layout,
                        &shadow_pass_bind_group_layout,
                        &model_instanced_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let model_instanced_shadow_pipeline = Arc::new(base.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Model instanced shadow render pipeline"),
                layout: Some(&model_instanced_shadow_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &model_instanced_shader,
                    entry_point: "shadow_vs_main",
                    buffers: &[
                        render::ModelVertex::LAYOUT,
                        render::ShadowedModelInstanced::INSTANCE_LAYOUT,
                    ],
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: Some(wgpu::Face::Back),
                    // cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                // depth_stencil: None,
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: render::Texture::DEPTH_FORMAT,
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

        // let colored_cube_matrix_uniform =
        //     base.device
        //         .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //             label: Some("Colored cube matrix uniform buffer"),
        //             // this is fine because initial transform is the identity transform
        //             contents: bytemuck::cast_slice(&camera.as_uniform_data()),
        //             usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        //         });
        // let colored_cube = RenderObject::RawVertex(RawVertexObject {
        //     vertex_buffer: base
        //         .device
        //         .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //             label: Some("Colored cube vertex buffer"),
        //             contents: bytemuck::cast_slice(CUBE_VERTICES),
        //             usage: wgpu::BufferUsages::VERTEX,
        //         }),
        //     index_buffer: base
        //         .device
        //         .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //             label: Some("Colored cube index buffer"),
        //             contents: bytemuck::cast_slice(CUBE_INDICES),
        //             usage: wgpu::BufferUsages::INDEX,
        //         }),
        //     num_indices: CUBE_INDICES.len() as u32,
        //     pipeline: colored_vertex_pipeline.clone(),
        //     bind_group: base.device.create_bind_group(&wgpu::BindGroupDescriptor {
        //         label: Some("Colored cube bind group"),
        //         layout: &colored_vertex_bind_group_layout,
        //         entries: &[
        //             wgpu::BindGroupEntry {
        //                 binding: 0,
        //                 resource: colored_cube_matrix_uniform.as_entire_binding(),
        //             }
        //         ],
        //     }),
        //     matrix_uniform: colored_cube_matrix_uniform,
        //     transform: Transform::default().into(),
        // });

        let mut white_cube_transforms = Vec::with_capacity(32);
        for i in 0..32 {
            let yi = i / 4;
            let y = yi as f32 * CELL_WIDTH;
            let x = (i % 4) as f32 * CELL_WIDTH * 2.0 + ((1 - yi % 2) as f32 * CELL_WIDTH);
            white_cube_transforms.push(Transform {
                pos: vec3(x - 3.5 * CELL_WIDTH, 0.0, y as f32 - 3.5 * CELL_WIDTH),
                scale: vec3(CELL_WIDTH / 2.0, CELL_HEIGHT / 2.0, CELL_WIDTH / 2.0),
                ..Default::default()
            });
        }
        let mut black_cube_transforms = Vec::with_capacity(32);
        for i in 0..32 {
            let yi = i / 4;
            let y = yi as f32 * CELL_WIDTH;
            let x = (i % 4) as f32 * CELL_WIDTH * 2.0 + ((yi % 2) as f32 * CELL_WIDTH);
            black_cube_transforms.push(Transform {
                pos: vec3(x - 3.5 * CELL_WIDTH, 0.0, y as f32 - 3.5 * CELL_WIDTH),
                scale: vec3(CELL_WIDTH / 2.0, CELL_HEIGHT / 2.0, CELL_WIDTH / 2.0),
                ..Default::default()
            });
        }
        let mut initial_white_cube_matrices = Vec::with_capacity(32);
        initial_white_cube_matrices.extend(white_cube_transforms.iter().map(|transform| {
            let mut matrices = [0.0; 32];
            let (mvp_mat, normal_mat) = matrices.split_at_mut(16);
            let mvp = camera.view_proj_matrix() * transform.to_matrix();
            mvp_mat.copy_from_slice(&mvp.to_cols_array());
            normal_mat.copy_from_slice(&mvp.inverse().transpose().to_cols_array());
            matrices
        }));
        let mut initial_black_cube_matrices = Vec::with_capacity(32);
        initial_black_cube_matrices.extend(black_cube_transforms.iter().map(|transform| {
            let mut matrices = [0.0; 32];
            let (mvp_mat, normal_mat) = matrices.split_at_mut(16);
            let mvp = camera.view_proj_matrix() * transform.to_matrix();
            mvp_mat.copy_from_slice(&mvp.to_cols_array());
            normal_mat.copy_from_slice(&mvp.inverse().transpose().to_cols_array());
            matrices
        }));
        let white_cube_instance_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("White cube instance buffer"),
                    contents: bytemuck::cast_slice(&initial_white_cube_matrices),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
                });
        let white_cube_instanced_model = Arc::new(
            resources::load_model(
                "cube-white.obj",
                &base.device,
                &base.queue,
                &model_instanced_bind_group_layout,
                |tex, material_name, layout| {
                    base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("White cube material bind group ({material_name})")),
                        layout: layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&tex.sampler),
                            },
                        ],
                    })
                },
            )
            .await
            .unwrap(),
        );
        let black_cube_instance_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Black cube instance buffer"),
                    contents: bytemuck::cast_slice(&initial_black_cube_matrices),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
                });
        let black_cube_instanced_model = Arc::new(
            resources::load_model(
                "cube-black.obj",
                &base.device,
                &base.queue,
                &model_instanced_bind_group_layout,
                |tex, material_name, layout| {
                    base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Black cube material bind group ({material_name})")),
                        layout: layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&tex.sampler),
                            },
                        ],
                    })
                },
            )
            .await
            .unwrap(),
        );
        let white_cube = ShadowedRenderObject::ModelInstanced(ShadowedModelInstanced {
            model: white_cube_instanced_model.clone(),
            pipeline: model_instanced_pipeline.clone(),
            shadow_pipeline: model_instanced_shadow_pipeline.clone(),
            instance_buffer: white_cube_instance_buffer,
            transform_matrices: initial_white_cube_matrices,
            transforms: white_cube_transforms.into(),
        });
        let black_cube = ShadowedRenderObject::ModelInstanced(ShadowedModelInstanced {
            model: black_cube_instanced_model.clone(),
            pipeline: model_instanced_pipeline.clone(),
            shadow_pipeline: model_instanced_shadow_pipeline.clone(),
            instance_buffer: black_cube_instance_buffer,
            transform_matrices: initial_black_cube_matrices,
            transforms: black_cube_transforms.into(),
        });
        println!("made model cubes!");

        // let black_pawn_transforms = (0..8).map(|i| {
        //     Transform {
        //         pos: vec3(board_x_to_world_x(i as f32), board_y_to_world_y(6.0), CELL_HEIGHT / 2.0),
        //         scale: CELL_HEIGHT * Vec3::ONE,
        //         ..Default::default()
        //     }
        // }).collect::<Vec<Transform>>();
        // let black_pawn_matrices = black_pawn_transforms.iter().map(|transform| {
        //     transform.to_matrix().to_cols_array()
        // }).collect::<Vec<[f32; 16]>>();
        // let black_bishop_transforms = vec![
        //     Transform {
        //         pos: vec3(board_x_to_world_x(2.0), board_y_to_world_y(7.0), CELL_HEIGHT / 2.0),
        //         scale: CELL_HEIGHT * Vec3::ONE,
        //         ..Default::default()
        //     },
        //     Transform {
        //         pos: vec3(board_x_to_world_x(5.0), board_y_to_world_y(7.0), CELL_HEIGHT / 2.0),
        //         scale: CELL_HEIGHT * Vec3::ONE,
        //         ..Default::default()
        //     }
        // ];
        // let black_rook_transforms = Vec::with_capacity()
        let black_queen_transforms = vec![
            Transform {
                pos: vec3(
                    board_x_to_world_x(2.0),
                    CELL_HEIGHT / 2.0,
                    board_y_to_world_y(0.0),
                ),
                scale: QUEEN_SCALE,
                ..Default::default()
            },
            Transform {
                pos: vec3(
                    board_x_to_world_x(0.0),
                    CELL_HEIGHT / 2.0,
                    board_y_to_world_y(4.0),
                ),
                scale: QUEEN_SCALE,
                ..Default::default()
            },
            Transform {
                pos: vec3(
                    board_x_to_world_x(5.0),
                    CELL_HEIGHT / 2.0,
                    board_y_to_world_y(7.0),
                ),
                scale: QUEEN_SCALE,
                ..Default::default()
            },
            Transform {
                pos: vec3(
                    board_x_to_world_x(7.0),
                    CELL_HEIGHT / 2.0,
                    board_y_to_world_y(3.0),
                ),
                scale: QUEEN_SCALE,
                ..Default::default()
            },
        ];
        let black_queen_transform_matrices = black_queen_transforms
            .iter()
            .map(|transform| {
                let mut matrices = [0.0; 32];
                let (mvp_mat, normal_mat) = matrices.split_at_mut(16);
                let mvp = camera.view_proj_matrix() * transform.to_matrix();
                mvp_mat.copy_from_slice(&mvp.to_cols_array());
                normal_mat.copy_from_slice(&mvp.inverse().transpose().to_cols_array());
                matrices
            })
            .collect::<Vec<_>>();
        let white_queen_transforms = vec![
            Transform {
                pos: vec3(
                    board_x_to_world_x(1.0),
                    CELL_HEIGHT / 2.0,
                    board_y_to_world_y(2.0),
                ),
                scale: QUEEN_SCALE,
                ..Default::default()
            },
            Transform {
                pos: vec3(
                    board_x_to_world_x(3.0),
                    CELL_HEIGHT / 2.0,
                    board_y_to_world_y(6.0),
                ),
                scale: QUEEN_SCALE,
                ..Default::default()
            },
            Transform {
                pos: vec3(
                    board_x_to_world_x(6.0),
                    CELL_HEIGHT / 2.0,
                    board_y_to_world_y(5.0),
                ),
                scale: QUEEN_SCALE,
                ..Default::default()
            },
            Transform {
                pos: vec3(
                    board_x_to_world_x(4.0),
                    CELL_HEIGHT / 2.0,
                    board_y_to_world_y(1.0),
                ),
                scale: QUEEN_SCALE,
                ..Default::default()
            },
        ];
        let white_queen_transform_matrices = white_queen_transforms
            .iter()
            .map(|transform| {
                let mut matrices = [0.0; 32];
                let (mvp_mat, normal_mat) = matrices.split_at_mut(16);
                let mvp = camera.view_proj_matrix() * transform.to_matrix();
                mvp_mat.copy_from_slice(&mvp.to_cols_array());
                normal_mat.copy_from_slice(&mvp.inverse().transpose().to_cols_array());
                matrices
            })
            .collect::<Vec<_>>();
        let white_queen_instance_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("White queen instance buffer"),
                    contents: bytemuck::cast_slice(&white_queen_transform_matrices),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
                });
        let white_queen_instanced_model = Arc::new(
            resources::load_model(
                "queen-white.obj",
                &base.device,
                &base.queue,
                &model_instanced_bind_group_layout,
                |tex, material_name, layout| {
                    base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!(
                            "White queen material bind group ({material_name})"
                        )),
                        layout: layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&tex.sampler),
                            },
                        ],
                    })
                },
            )
            .await
            .unwrap(),
        );
        let black_queen_instance_buffer =
            base.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Black queen instance buffer"),
                    contents: bytemuck::cast_slice(&black_queen_transform_matrices),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
                });
        let black_queen_instanced_model = Arc::new(
            resources::load_model(
                "queen-black.obj",
                &base.device,
                &base.queue,
                &model_instanced_bind_group_layout,
                |tex, material_name, layout| {
                    base.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!(
                            "Black queen material bind group ({material_name})"
                        )),
                        layout: layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&tex.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&tex.sampler),
                            },
                        ],
                    })
                },
            )
            .await
            .unwrap(),
        );
        let white_queen = ShadowedRenderObject::ModelInstanced(ShadowedModelInstanced {
            model: white_queen_instanced_model.clone(),
            pipeline: model_instanced_pipeline.clone(),
            shadow_pipeline: model_instanced_shadow_pipeline.clone(),
            instance_buffer: white_queen_instance_buffer,
            transform_matrices: white_queen_transform_matrices,
            transforms: white_queen_transforms.into(),
        });
        let black_queen = ShadowedRenderObject::ModelInstanced(ShadowedModelInstanced {
            model: black_queen_instanced_model.clone(),
            pipeline: model_instanced_pipeline.clone(),
            shadow_pipeline: model_instanced_shadow_pipeline.clone(),
            instance_buffer: black_queen_instance_buffer,
            transform_matrices: black_queen_transform_matrices,
            transforms: black_queen_transforms.into(),
        });

        let z_buffer = render::Texture::create_depth_texture(&base, None);

        let objects = HashMap::from([
            // (COLORED_CUBE_ID, colored_cube),
            (WHITE_CUBES_ID, white_cube),
            (BLACK_CUBES_ID, black_cube),
            (WHITE_QUEENS_ID, white_queen),
            (BLACK_QUEENS_ID, black_queen),
        ]);

        Self {
            should_quit: false,
            last_update: Utc::now(),
            camera,
            objects,
            empty_bind_group,
            light,
            light_camera,
            light_uniform,
            camera_uniform,
            shadow_pass_bind_group,
            standard_pass_bind_group,
            shadow_z_buffer,
            // colored_vertex_pipeline,
            // vertex_buffer,
            // index_buffer,
            // camera_uniform,
            // camera_bind_group,
            keystates: HashMap::new(),
            z_buffer,
            base,
        }
    }

    fn swap_queens(&mut self) {
        let black_queens = self.objects.remove(&BLACK_QUEENS_ID).unwrap();
        let white_queens = self.objects.remove(&WHITE_QUEENS_ID).unwrap();
        if let (
            ShadowedRenderObject::ModelInstanced(mut black_queens),
            ShadowedRenderObject::ModelInstanced(mut white_queens),
        ) = (black_queens, white_queens)
        {
            std::mem::swap(&mut black_queens.transforms, &mut white_queens.transforms);
            std::mem::swap(
                &mut black_queens.transform_matrices,
                &mut white_queens.transform_matrices,
            );
            self.objects.insert(
                BLACK_QUEENS_ID,
                ShadowedRenderObject::ModelInstanced(black_queens),
            );
            self.objects.insert(
                WHITE_QUEENS_ID,
                ShadowedRenderObject::ModelInstanced(white_queens),
            );
        } else {
            unreachable!();
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
        // if *self.keystates.get(&VirtualKeyCode::R).unwrap_or(&false) {
        // }

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
        self.z_buffer = render::Texture::create_depth_texture(&self.base, None);
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
        let frame_state = FrameInfo {
            // view_proj: self.camera.view_proj_matrix(),
            queue: &self.base.queue,
        };
        let mut encoder =
            self.base
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render encoder"),
                });

        let light_camera_changed = self.light_camera.is_dirty();
        let camera_changed = self.camera.is_dirty();
        self.camera.if_either_dirty(&self.light_camera, |camera, light_camera| {
            let light = PointLight {
                    view_proj: /* MODIFIED HERE */ light_camera.view_proj_matrix().to_cols_array(),
                    pos: (camera.view_proj_matrix() * Vec3::from(/* MODIFIED HERE */ self.light.pos).extend(1.0)).xyz().into(),
                    ..self.light
            };
            self.base.queue.write_buffer(&self.light_uniform, 0, bytemuck::cast_slice(&[light]));
            self.base.queue.write_buffer(&self.camera_uniform, 0, bytemuck::cast_slice(&camera.as_uniform_data()));
        });

        {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow pass"),
                color_attachments: &[],
                // depth_stencil_attachment: None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_z_buffer.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            shadow_pass.set_bind_group(0, &self.empty_bind_group, &[]);
            shadow_pass.set_bind_group(1, &self.empty_bind_group, &[]);
            shadow_pass.set_bind_group(2, &self.shadow_pass_bind_group, &[]);
            let pass_info = PassInfo {
                frame: &frame_state,
                view: self. /* MODIFIED HERE */ light_camera.view_matrix(),
                proj: self. /* MODIFIED HERE */ light_camera.proj_matrix(),
                view_proj_changed: camera_changed,
                // pass: &mut shadow_pass,
                index: 0,
            };
            // self.light_camera.mark_clean();

            for object in self.objects.values_mut() {
                object.render_in_pass(&mut shadow_pass, &pass_info);
            }
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.769,
                            g: 0.869,
                            b: 0.875,
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
            // ideally we'd have per frame bindings, per pass bindings, per pipeline/material
            // bindings, and per object bindings for a basic forward rendering system (and
            // possibly many other kinds of rendering systems?)
            // however, i'm not currently grouping objects by pipeline, so that's one wasted
            // binding slot, and i'm not even sure if i'm going to
            render_pass.set_bind_group(0, &self.empty_bind_group, &[]);
            render_pass.set_bind_group(1, &self.empty_bind_group, &[]);
            render_pass.set_bind_group(2, &self.standard_pass_bind_group, &[]);
            let pass_info = PassInfo {
                frame: &frame_state,
                view: self.camera.view_matrix(),
                proj: self.camera.proj_matrix(),
                view_proj_changed: camera_changed,
                // pass: &mut render_pass,
                index: 1,
            };

            for object in self.objects.values_mut() {
                object.render_in_pass(&mut render_pass, &pass_info);
            }
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
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(virtual_keycode),
                                ..
                            },
                        ..
                    } => match state {
                        ElementState::Pressed => {
                            self.keystates.insert(*virtual_keycode, true);
                            if *virtual_keycode == VirtualKeyCode::R {
                                self.swap_queens()
                            }
                        }
                        ElementState::Released => {
                            self.keystates.insert(*virtual_keycode, false);
                        }
                    },
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
        window.set_inner_size(PhysicalSize::new(800, 600));

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
