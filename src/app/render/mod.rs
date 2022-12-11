use std::sync::Arc;

use glam::*;
use winit::window::Window;

use super::utils::Dirtiable;

mod texture;
pub use texture::*;
mod model;
pub use model::*;

// #[derive(Debug)]
// pub enum Pipeline {
//     Render(wgpu::RenderPipeline),
//     Compute(wgpu::ComputePipeline),
// }
// impl From<wgpu::RenderPipeline> for Pipeline {
//     fn from(value: wgpu::RenderPipeline) -> Self {
//         Self::Render(value)
//     }
// }
// impl From<wgpu::ComputePipeline> for Pipeline {
//     fn from(value: wgpu::ComputePipeline) -> Self {
//         Self::Compute(value)
//     }
// }

pub struct WgpuBase {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,

    pub(crate) surface: wgpu::Surface,
    /// Cached result of calling `surface.get_supported_formats()`
    pub(crate) _surface_supported_formats: Vec<wgpu::TextureFormat>,
    /// Holds the size according to the wgpu::Surface
    pub(crate) surface_config: wgpu::SurfaceConfiguration,
    pub(crate) resize_queued: bool,

    /// Holds the size according to windowing events, may differ from `window.inner_size()` or
    /// from size as reported by the `width` and `height` fields of `surface_config`.
    pub(crate) size: winit::dpi::PhysicalSize<u32>,
    // for safety purposes this MUST be dropped last
    pub(crate) window: Window,
}

impl WgpuBase {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        // SAFETY: window gets moved into the returned value, and drop order guarantees it
        // will be dropped after surface (and everything else)
        // in the time between it getting moved into the return value and window being used
        // here, it's the longest-lived temporary in this function and therefore will be dropped last
        let surface = unsafe { instance.create_surface(&window) };
        #[cfg(not(target_arch = "wasm32"))]
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .filter(|adapter| adapter.is_surface_supported(&surface))
            .next()
            .unwrap();
        #[cfg(target_arch = "wasm32")]
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let device_desc = wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            },
            ..Default::default()
        };
        let (device, queue) = adapter.request_device(&device_desc, None).await.unwrap();

        let surface_supported_formats = surface.get_supported_formats(&adapter);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            // TODO: can this be avoided? maybe store the returned Vec for later use so it's not an alloc/free call in quick succession??
            format: surface_supported_formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &surface_config);

        Self {
            device,
            queue,
            surface,
            _surface_supported_formats: surface_supported_formats,
            surface_config,
            resize_queued: false,
            size,
            window,
        }
    }

    /// Prepares a resize but does not execute it (yet). Only one resize can be queued at a
    /// time, and old resizes will be overwritten by new resizes.
    pub fn queue_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.resize_queued = true;
        }
    }

    // Applies a queued resize if one exists.
    pub fn apply_resize(&mut self) {
        if self.resize_queued {
            self.surface_config.width = self.size.width;
            self.surface_config.height = self.size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.resize_queued = false;
        }
    }

    pub fn _resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.queue_resize(new_size);
        self.apply_resize()
    }

    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

pub trait Vertex {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute];
    const LAYOUT: wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColoredVertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex for ColoredVertex {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            shader_location: 1,
        },
    ];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBUTES,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub pos: Vec3,
    pub rot: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn _translation(pos: Vec3) -> Self {
        Self {
            pos,
            ..Default::default()
        }
    }
    pub fn _rotation(rot: Quat) -> Self {
        Self {
            rot,
            ..Default::default()
        }
    }
    pub fn _scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Default::default()
        }
    }

    pub fn _translate_by(self, v: Vec3) -> Self {
        Self {
            pos: self.pos + v,
            ..self
        }
    }
    pub fn _rotate_by(self, q: Quat) -> Self {
        Self {
            rot: (self.rot * q).normalize(),
            ..self
        }
    }
    pub fn _scale_by(self, s: Vec3) -> Self {
        Self {
            scale: self.scale * s,
            ..self
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        // let x_axis = self.rot.mul_vec3(Vec3::X * self.scale.x);
        // let y_axis = self.rot.mul_vec3(Vec3::Y * self.scale.y);
        // let z_axis = self.rot.mul_vec3(Vec3::Z * self.scale.z);
        // Mat4::from_cols(x_axis, y_axis, z_axis, w_axis);
        Mat4::from_scale_rotation_translation(self.scale, self.rot, self.pos)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

// pub struct RenderObject {
//     pub vertex_buffer: Arc<wgpu::Buffer>,
//     pub num_vertices: usize,
//     pub index_buffer: Arc<wgpu::Buffer>,
//     pub num_indices: usize,
//     pub pipeline: Arc<wgpu::RenderPipeline>,
//     pub model_bind_group: wgpu::BindGroup,
//     pub model_uniform: wgpu::Buffer,
//     pub transform: Dirtiable<Transform>,
//     // low-effort but it works
//     pub render_hook: Option<Box<dyn FnMut(&Self, &mut wgpu::RenderPass, &wgpu::Queue)>>,
// }

// impl RenderObject {
//     pub fn update_uniforms(&self, view_proj: Mat4, always_update: bool, queue: &wgpu::Queue) {
//         let update_fn = |transform: &Transform| {
//             let mat = view_proj * transform.to_matrix();
//             queue.write_buffer(
//                 &self.model_uniform,
//                 0,
//                 bytemuck::cast_slice(&mat.to_cols_array()),
//             );
//         };
//         if always_update {
//             update_fn(&self.transform);
//         } else {
//             self.transform.if_dirty(update_fn);
//         }
//     }
// }

#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
pub struct PointLight {
    // diffuse color is based on the texture in the shader
    pub pos: [f32; 3],
    pub ambient_power: f32,
    pub specular_color: [f32; 3],
    pub specular_power: f32,
    pub exponent: f32,
}

pub struct PerFrameState<'a> {
    pub queue: &'a wgpu::Queue,
}

pub struct PerPassState<'a, 'pass: 'a> {
    pub frame: &'a PerFrameState<'pass>,
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj_changed: bool,
    pub pass: &'a mut wgpu::RenderPass<'pass>,
    // pub index: u32,
    // pub light_buffer: &'a wgpu::Buffer,
    // pub light: PointLight,
}

pub trait Render {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    );
}

pub struct RawVertexObject {
    // may be sharing buffers with other objects
    // then why not instance it?
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub transform: Dirtiable<Transform>,
    pub pipeline: Arc<wgpu::RenderPipeline>,
    pub shadow_pipeline: Arc<wgpu::RenderPipeline>,
    pub bind_group: wgpu::BindGroup,
    pub matrix_uniform: wgpu::Buffer,
}

impl RawVertexObject {
    pub fn update_uniforms<'a, 'pass>(&mut self, state: &mut PerPassState<'a, 'pass>) {
        let update_fn = |transform: &Transform| {
            let mat = state.proj * state.view * transform.to_matrix();
            state.frame.queue.write_buffer(
                &self.matrix_uniform,
                0,
                bytemuck::cast_slice(&mat.to_cols_array()),
            );
        };
        if state.view_proj_changed {
            update_fn(&self.transform);
        } else {
            self.transform.if_dirty(update_fn);
        }
    }
}

impl Render for RawVertexObject {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    ) {
        self.update_uniforms(state);
        if pass_index == 0 {
            state.pass.set_pipeline(&self.shadow_pipeline);
        } else if pass_index == 1 {
            state.pass.set_pipeline(&self.pipeline);
        } else {
            unreachable!();
        }

        state.pass.set_pipeline(&self.pipeline);
        state
            .pass
            .set_vertex_buffer(0, self.vertex_buffer.slice(0..));
        state
            .pass
            .set_index_buffer(self.index_buffer.slice(0..), wgpu::IndexFormat::Uint16);
        state.pass.set_bind_group(3, &self.bind_group, &[]);
        state.pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}

// pub struct RawVertexInstanced {
//     // may be sharing buffers with other objects
//     // then why not instance it?
//     pub vertex_buffer: wgpu::Buffer,
//     pub instance_buffer: wgpu::Buffer,
//     pub index_buffer: wgpu::Buffer,
//     pub num_indices: u32,
//     pub num_instances: u32,
//     pub transforms: Dirtiable<Vec<Transform>>,
//     pub pipeline: Arc<wgpu::RenderPipeline>,
//     pub bind_group: wgpu::BindGroup,
//     pub uniform: wgpu::Buffer,
// }

// impl RawVertexInstanced {
//     pub fn update_uniforms<'a, 'pass>(&mut self, pass: &MultipassState<'a, 'pass>) {
//         let update_fn = |transforms: &Vec<Transform>| {
//             for (i, transform) in transforms.iter().enumerate() {
//                 let mat = pass.frame.view_proj * transform.to_matrix();
//                 pass.frame.queue.write_buffer(
//                     &self.uniform,
//                     (std::mem::size_of::<Mat4>() * i) as wgpu::BufferAddress,
//                     bytemuck::cast_slice(&mat.to_cols_array()),
//                 );
//             }
//         };
//         if pass.frame.view_proj_changed {
//             update_fn(&self.transforms);
//         } else {
//             self.transforms.if_dirty(update_fn);
//         }
//     }
// }

// impl Render for RawVertexInstanced {
//     fn render_in_pass<'a, 'pass: 'a>(&'pass mut self, state: &mut PerPassState<'a, 'pass>) {
//         self.update_uniforms(&state);

//         state.pass.set_pipeline(&self.pipeline);
//         state.pass
//             .set_vertex_buffer(0, self.vertex_buffer.slice(0..));
//         state.pass
//             .set_vertex_buffer(1, self.instance_buffer.slice(0..));
//         state.pass
//             .set_index_buffer(self.index_buffer.slice(0..), wgpu::IndexFormat::Uint16);
//         state.pass.set_bind_group(3, &self.bind_group, &[]);
//         state.pass
//             .draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
//     }
// }

pub struct ModelObject {
    pub model: Arc<Model>,
    pub transform: Dirtiable<Transform>,
    pub pipeline: Arc<wgpu::RenderPipeline>,
    pub shadow_pipeline: Arc<wgpu::RenderPipeline>,
    pub uniform: wgpu::Buffer,
}

impl ModelObject {
    pub fn update_uniforms<'a, 'pass>(&mut self, state: &PerPassState<'a, 'pass>) {
        let update_fn = |transform: &Transform| {
            let mat = state.proj * state.view * transform.to_matrix();
            state.frame.queue.write_buffer(
                &self.uniform,
                0,
                bytemuck::cast_slice(&mat.to_cols_array()),
            );
        };
        if state.view_proj_changed {
            update_fn(&self.transform);
        } else {
            self.transform.if_dirty(update_fn);
        }
    }
}

impl Render for ModelObject {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    ) {
        self.update_uniforms(state);
        if pass_index == 0 {
            state.pass.set_pipeline(&self.shadow_pipeline);
        } else if pass_index == 1 {
            state.pass.set_pipeline(&self.pipeline);
        } else {
            unreachable!();
        }

        state.pass.set_pipeline(&self.pipeline);
        for mesh in &self.model.meshes {
            let material = &self.model.materials[mesh.material];
            state.pass.set_bind_group(3, &material.bind_group, &[]);
            state
                .pass
                .set_vertex_buffer(0, mesh.vertex_buffer.slice(0..));
            // TODO: make this Uint16 by fixing the model loading? or does u32 make more sense anyways?
            state
                .pass
                .set_index_buffer(mesh.index_buffer.slice(0..), wgpu::IndexFormat::Uint32);
            state.pass.draw_indexed(0..mesh.num_elements, 0, 0..1);
        }
    }
}

/// Weirdly this is only really suited to static objects bc I don't have time to make it better
pub struct ModelInstanced {
    // if i were taking more time or care i wouldn't just be spamming Arc everywhere but i need
    // shared ownership and i need it fast and i haven't yet hit a performance wall and don't
    // expect to anytime soon
    pub model: Arc<Model>,
    pub transforms: Dirtiable<Vec<Transform>>,
    // this is just raw memory that gets preallocated and reused to reduce buffer copies
    // this is bc each buffer copy in wgpu has potential to create the overhead of an additional
    // synchronization primitive in one of the backends
    pub transform_matrices: Vec<[f32; 32]>,
    pub instance_buffer: wgpu::Buffer,
    pub pipeline: Arc<wgpu::RenderPipeline>,
    pub shadow_pipeline: Arc<wgpu::RenderPipeline>,
}

impl ModelInstanced {
    pub const INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<[f32; 32]>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (1 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (2 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 5,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (3 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 6,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (4 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 7,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (5 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 8,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (6 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 9,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (7 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 10,
            },
        ],
    };

    pub fn update_uniforms<'a, 'pass>(&mut self, state: &PerPassState<'a, 'pass>) {
        let mut update_fn = |transforms: &Vec<Transform>| {
            // this is probably more efficient than multiple writes, but it's also a known size
            // can this be preallocated and just reuse that memory repeatedly?
            self.transform_matrices
                .iter_mut()
                .zip(transforms)
                .for_each(|(matrices, transform)| {
                    let (mvp_mat, normal_mat) = matrices.split_at_mut(16);
                    let m = transform.to_matrix();
                    let mv = state.view * m;
                    let mvp = state.proj * mv;
                    mvp_mat.copy_from_slice(&mvp.to_cols_array());
                    // hm is there away to avoid a matrix inverse per frame
                    // maybe 4x4 inverses are cheapish??? hopefully???
                    normal_mat.copy_from_slice(&mv.inverse().transpose().to_cols_array());
                });
            state.frame.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.transform_matrices),
            );
        };
        if state.view_proj_changed {
            self.transforms.clean_with(update_fn);
        } else {
            self.transforms.if_dirty(update_fn);
        }
    }
}

impl Render for ModelInstanced {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    ) {
        self.update_uniforms(state);
        if pass_index == 0 {
            state.pass.set_pipeline(&self.shadow_pipeline);
        } else if pass_index == 1 {
            state.pass.set_pipeline(&self.pipeline);
        } else {
            unreachable!();
        }

        for mesh in &self.model.meshes {
            let material = &self.model.materials[mesh.material];
            state.pass.set_bind_group(3, &material.bind_group, &[]);
            state
                .pass
                .set_vertex_buffer(0, mesh.vertex_buffer.slice(0..));
            state
                .pass
                .set_vertex_buffer(1, self.instance_buffer.slice(0..));
            // TODO: make this Uint16 by fixing the model loading? or does u32 make more sense anyways?
            state
                .pass
                .set_index_buffer(mesh.index_buffer.slice(0..), wgpu::IndexFormat::Uint32);
            state
                .pass
                .draw_indexed(0..mesh.num_elements, 0, 0..self.transforms.len() as u32);
        }
    }
}

pub enum RenderObject {
    RawVertex(RawVertexObject),
    // RawVertexInstanced(RawVertexInstanced),
    _Model(ModelObject),
    ModelInstanced(ModelInstanced),
    _Dynamic(Box<dyn Render>),
}

impl Render for RenderObject {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    ) {
        match self {
            Self::RawVertex(raw) => raw.render_in_pass(state, pass_index),
            // Self::RawVertexInstanced(raw) => raw.render_in_pass(state),
            Self::_Model(model) => model.render_in_pass(state, pass_index),
            Self::ModelInstanced(model) => model.render_in_pass(state, pass_index),
            Self::_Dynamic(dynamic) => dynamic.render_in_pass(state, pass_index),
        }
    }
}

// impl RenderObject {
//     pub fn update_uniforms(&self) {
//         match &self {

//         }
//         let update_fn = |transform: &Transform| {
//             let mat = view_proj * transform.to_matrix();
//             queue.write_buffer(
//                 &self.model_uniform,
//                 0,
//                 bytemuck::cast_slice(&mat.to_cols_array()),
//             );
//         };
//         if always_update {
//             update_fn(&self.transform);
//         } else {
//             self.transform.if_dirty(update_fn);
//         }
//     }
// }

/*

use std::sync::Arc;

use glam::*;
use winit::window::Window;

use super::utils::Dirtiable;

mod texture;
pub use texture::*;
mod model;
pub use model::*;

// #[derive(Debug)]
// pub enum Pipeline {
//     Render(wgpu::RenderPipeline),
//     Compute(wgpu::ComputePipeline),
// }
// impl From<wgpu::RenderPipeline> for Pipeline {
//     fn from(value: wgpu::RenderPipeline) -> Self {
//         Self::Render(value)
//     }
// }
// impl From<wgpu::ComputePipeline> for Pipeline {
//     fn from(value: wgpu::ComputePipeline) -> Self {
//         Self::Compute(value)
//     }
// }

pub struct WgpuBase {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,

    pub(crate) surface: wgpu::Surface,
    /// Cached result of calling `surface.get_supported_formats()`
    pub(crate) _surface_supported_formats: Vec<wgpu::TextureFormat>,
    /// Holds the size according to the wgpu::Surface
    pub(crate) surface_config: wgpu::SurfaceConfiguration,
    pub(crate) resize_queued: bool,

    /// Holds the size according to windowing events, may differ from `window.inner_size()` or
    /// from size as reported by the `width` and `height` fields of `surface_config`.
    pub(crate) size: winit::dpi::PhysicalSize<u32>,
    // for safety purposes this MUST be dropped last
    pub(crate) window: Window,
}

impl WgpuBase {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        // SAFETY: window gets moved into the returned value, and drop order guarantees it
        // will be dropped after surface (and everything else)
        // in the time between it getting moved into the return value and window being used
        // here, it's the longest-lived temporary in this function and therefore will be dropped last
        let surface = unsafe { instance.create_surface(&window) };
        #[cfg(not(target_arch = "wasm32"))]
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .filter(|adapter| adapter.is_surface_supported(&surface))
            .next()
            .unwrap();
        #[cfg(target_arch = "wasm32")]
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let device_desc = wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            },
            ..Default::default()
        };
        let (device, queue) = adapter.request_device(&device_desc, None).await.unwrap();

        let surface_supported_formats = surface.get_supported_formats(&adapter);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            // TODO: can this be avoided? maybe store the returned Vec for later use so it's not an alloc/free call in quick succession??
            format: surface_supported_formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &surface_config);

        Self {
            device,
            queue,
            surface,
            _surface_supported_formats: surface_supported_formats,
            surface_config,
            resize_queued: false,
            size,
            window,
        }
    }

    /// Prepares a resize but does not execute it (yet). Only one resize can be queued at a
    /// time, and old resizes will be overwritten by new resizes.
    pub fn queue_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.resize_queued = true;
        }
    }

    // Applies a queued resize if one exists.
    pub fn apply_resize(&mut self) {
        if self.resize_queued {
            self.surface_config.width = self.size.width;
            self.surface_config.height = self.size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.resize_queued = false;
        }
    }

    pub fn _resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.queue_resize(new_size);
        self.apply_resize()
    }

    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

pub trait Vertex {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute];
    const LAYOUT: wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColoredVertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex for ColoredVertex {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            shader_location: 1,
        },
    ];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBUTES,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub pos: Vec3,
    pub rot: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn _translation(pos: Vec3) -> Self {
        Self {
            pos,
            ..Default::default()
        }
    }
    pub fn _rotation(rot: Quat) -> Self {
        Self {
            rot,
            ..Default::default()
        }
    }
    pub fn _scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Default::default()
        }
    }

    pub fn _translate_by(self, v: Vec3) -> Self {
        Self {
            pos: self.pos + v,
            ..self
        }
    }
    pub fn _rotate_by(self, q: Quat) -> Self {
        Self {
            rot: (self.rot * q).normalize(),
            ..self
        }
    }
    pub fn _scale_by(self, s: Vec3) -> Self {
        Self {
            scale: self.scale * s,
            ..self
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        // let x_axis = self.rot.mul_vec3(Vec3::X * self.scale.x);
        // let y_axis = self.rot.mul_vec3(Vec3::Y * self.scale.y);
        // let z_axis = self.rot.mul_vec3(Vec3::Z * self.scale.z);
        // Mat4::from_cols(x_axis, y_axis, z_axis, w_axis);
        Mat4::from_scale_rotation_translation(self.scale, self.rot, self.pos)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            rot: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

// pub struct RenderObject {
//     pub vertex_buffer: Arc<wgpu::Buffer>,
//     pub num_vertices: usize,
//     pub index_buffer: Arc<wgpu::Buffer>,
//     pub num_indices: usize,
//     pub pipeline: Arc<wgpu::RenderPipeline>,
//     pub model_bind_group: wgpu::BindGroup,
//     pub model_uniform: wgpu::Buffer,
//     pub transform: Dirtiable<Transform>,
//     // low-effort but it works
//     pub render_hook: Option<Box<dyn FnMut(&Self, &mut wgpu::RenderPass, &wgpu::Queue)>>,
// }

// impl RenderObject {
//     pub fn update_uniforms(&self, view_proj: Mat4, always_update: bool, queue: &wgpu::Queue) {
//         let update_fn = |transform: &Transform| {
//             let mat = view_proj * transform.to_matrix();
//             queue.write_buffer(
//                 &self.model_uniform,
//                 0,
//                 bytemuck::cast_slice(&mat.to_cols_array()),
//             );
//         };
//         if always_update {
//             update_fn(&self.transform);
//         } else {
//             self.transform.if_dirty(update_fn);
//         }
//     }
// }

#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
pub struct PointLight {
    // diffuse color is based on the texture in the shader
    pub pos: [f32; 3],
    pub ambient_power: f32,
    pub specular_color: [f32; 3],
    pub specular_power: f32,
    pub exponent: f32,
}

pub struct PerFrameState<'a> {
    pub queue: &'a wgpu::Queue,
}

pub struct PerPassState<'a, 'pass: 'a> {
    pub frame: &'a PerFrameState<'pass>,
    pub view: Mat4,
    pub proj: Mat4,
    // pub view_proj_changed: bool,
    pub pass: &'a mut wgpu::RenderPass<'pass>,
    // pub index: u32,
    // pub light_buffer: &'a wgpu::Buffer,
    // pub light: PointLight,
}

pub trait Render {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    );
}

pub struct RawVertexObject {
    // may be sharing buffers with other objects
    // then why not instance it?
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub transform: Dirtiable<Transform>,
    pub pipeline: Arc<wgpu::RenderPipeline>,
    pub shadow_pipeline: Arc<wgpu::RenderPipeline>,
    pub bind_group: wgpu::BindGroup,
    pub matrix_uniform: wgpu::Buffer,
}

impl RawVertexObject {
    pub fn update_uniforms<'a, 'pass>(&mut self, state: &mut PerPassState<'a, 'pass>) {
        let update_fn = |transform: &Transform| {
            let mat = transform.to_matrix();
            state.frame.queue.write_buffer(
                &self.matrix_uniform,
                0,
                bytemuck::cast_slice(&mat.to_cols_array()),
            );
        };
        // if state.view_proj_changed {
        //     update_fn(&self.transform);
        // } else {
            self.transform.if_dirty(update_fn);
        // }
    }
}

impl Render for RawVertexObject {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    ) {
        self.update_uniforms(state);
        if pass_index == 0 {
            state.pass.set_pipeline(&self.shadow_pipeline);
        } else if pass_index == 1 {
            state.pass.set_pipeline(&self.pipeline);
        } else {
            unreachable!();
        }

        state.pass.set_pipeline(&self.pipeline);
        state
            .pass
            .set_vertex_buffer(0, self.vertex_buffer.slice(0..));
        state
            .pass
            .set_index_buffer(self.index_buffer.slice(0..), wgpu::IndexFormat::Uint16);
        state.pass.set_bind_group(3, &self.bind_group, &[]);
        state.pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}

// pub struct RawVertexInstanced {
//     // may be sharing buffers with other objects
//     // then why not instance it?
//     pub vertex_buffer: wgpu::Buffer,
//     pub instance_buffer: wgpu::Buffer,
//     pub index_buffer: wgpu::Buffer,
//     pub num_indices: u32,
//     pub num_instances: u32,
//     pub transforms: Dirtiable<Vec<Transform>>,
//     pub pipeline: Arc<wgpu::RenderPipeline>,
//     pub bind_group: wgpu::BindGroup,
//     pub uniform: wgpu::Buffer,
// }

// impl RawVertexInstanced {
//     pub fn update_uniforms<'a, 'pass>(&mut self, pass: &MultipassState<'a, 'pass>) {
//         let update_fn = |transforms: &Vec<Transform>| {
//             for (i, transform) in transforms.iter().enumerate() {
//                 let mat = pass.frame.view_proj * transform.to_matrix();
//                 pass.frame.queue.write_buffer(
//                     &self.uniform,
//                     (std::mem::size_of::<Mat4>() * i) as wgpu::BufferAddress,
//                     bytemuck::cast_slice(&mat.to_cols_array()),
//                 );
//             }
//         };
//         if pass.frame.view_proj_changed {
//             update_fn(&self.transforms);
//         } else {
//             self.transforms.if_dirty(update_fn);
//         }
//     }
// }

// impl Render for RawVertexInstanced {
//     fn render_in_pass<'a, 'pass: 'a>(&'pass mut self, state: &mut PerPassState<'a, 'pass>) {
//         self.update_uniforms(&state);

//         state.pass.set_pipeline(&self.pipeline);
//         state.pass
//             .set_vertex_buffer(0, self.vertex_buffer.slice(0..));
//         state.pass
//             .set_vertex_buffer(1, self.instance_buffer.slice(0..));
//         state.pass
//             .set_index_buffer(self.index_buffer.slice(0..), wgpu::IndexFormat::Uint16);
//         state.pass.set_bind_group(3, &self.bind_group, &[]);
//         state.pass
//             .draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
//     }
// }

pub struct ModelObject {
    pub model: Arc<Model>,
    pub transform: Dirtiable<Transform>,
    pub pipeline: Arc<wgpu::RenderPipeline>,
    pub shadow_pipeline: Arc<wgpu::RenderPipeline>,
    pub uniform: wgpu::Buffer,
}

impl ModelObject {
    pub fn update_uniforms<'a, 'pass>(&mut self, state: &PerPassState<'a, 'pass>) {
        let update_fn = |transform: &Transform| {
            // let mat = state.proj * state.view * transform.to_matrix();
            let mat = transform.to_matrix();
            state.frame.queue.write_buffer(
                &self.uniform,
                0,
                bytemuck::cast_slice(&mat.to_cols_array()),
            );
        };
        // if state.view_proj_changed {
        //     update_fn(&self.transform);
        // } else {
            self.transform.if_dirty(update_fn);
        // }
    }
}

impl Render for ModelObject {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    ) {
        self.update_uniforms(state);
        if pass_index == 0 {
            state.pass.set_pipeline(&self.shadow_pipeline);
        } else if pass_index == 1 {
            state.pass.set_pipeline(&self.pipeline);
        } else {
            unreachable!();
        }

        state.pass.set_pipeline(&self.pipeline);
        for mesh in &self.model.meshes {
            let material = &self.model.materials[mesh.material];
            state.pass.set_bind_group(3, &material.bind_group, &[]);
            state
                .pass
                .set_vertex_buffer(0, mesh.vertex_buffer.slice(0..));
            // TODO: make this Uint16 by fixing the model loading? or does u32 make more sense anyways?
            state
                .pass
                .set_index_buffer(mesh.index_buffer.slice(0..), wgpu::IndexFormat::Uint32);
            state.pass.draw_indexed(0..mesh.num_elements, 0, 0..1);
        }
    }
}

/// Weirdly this is only really suited to static objects bc I don't have time to make it better
pub struct ModelInstanced {
    // if i were taking more time or care i wouldn't just be spamming Arc everywhere but i need
    // shared ownership and i need it fast and i haven't yet hit a performance wall and don't
    // expect to anytime soon
    pub model: Arc<Model>,
    pub transforms: Dirtiable<Vec<Transform>>,
    // this is just raw memory that gets preallocated and reused to reduce buffer copies
    // this is bc each buffer copy in wgpu has potential to create the overhead of an additional
    // synchronization primitive in one of the backends
    pub transform_matrices: Vec<[f32; 32]>,
    pub instance_buffer: wgpu::Buffer,
    pub pipeline: Arc<wgpu::RenderPipeline>,
    pub shadow_pipeline: Arc<wgpu::RenderPipeline>,
}

impl ModelInstanced {
    pub const INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<[f32; 32]>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (1 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (2 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 5,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (3 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 6,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (4 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 7,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (5 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 8,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (6 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 9,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: (7 * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress,
                shader_location: 10,
            },
        ],
    };

    pub fn update_uniforms<'a, 'pass>(&mut self, state: &PerPassState<'a, 'pass>) {
        let mut update_fn = |transforms: &Vec<Transform>| {
            // this is probably more efficient than multiple writes, but it's also a known size
            // can this be preallocated and just reuse that memory repeatedly?
            self.transform_matrices
                .iter_mut()
                .zip(transforms)
                .for_each(|(matrices, transform)| {
                    let (mvp_mat, normal_mat) = matrices.split_at_mut(16);
                    let m = transform.to_matrix();
                    let mv = state.view * m;
                    let mvp = state.proj * mv;
                    mvp_mat.copy_from_slice(&m.to_cols_array());
                    // hm is there away to avoid a matrix inverse per frame
                    // maybe 4x4 inverses are cheapish??? hopefully???
                    normal_mat.copy_from_slice(&mv.inverse().transpose().to_cols_array());
                });
            state.frame.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.transform_matrices),
            );
        };
        // if state.view_proj_changed {
        //     self.transforms.clean_with(update_fn);
        // } else {
            self.transforms.if_dirty(update_fn);
        // }
    }
}

impl Render for ModelInstanced {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    ) {
        self.update_uniforms(state);
        if pass_index == 0 {
            state.pass.set_pipeline(&self.shadow_pipeline);
        } else if pass_index == 1 {
            state.pass.set_pipeline(&self.pipeline);
        } else {
            unreachable!();
        }

        for mesh in &self.model.meshes {
            let material = &self.model.materials[mesh.material];
            state.pass.set_bind_group(3, &material.bind_group, &[]);
            state
                .pass
                .set_vertex_buffer(0, mesh.vertex_buffer.slice(0..));
            state
                .pass
                .set_vertex_buffer(1, self.instance_buffer.slice(0..));
            // TODO: make this Uint16 by fixing the model loading? or does u32 make more sense anyways?
            state
                .pass
                .set_index_buffer(mesh.index_buffer.slice(0..), wgpu::IndexFormat::Uint32);
            state
                .pass
                .draw_indexed(0..mesh.num_elements, 0, 0..self.transforms.len() as u32);
        }
    }
}

pub enum RenderObject {
    RawVertex(RawVertexObject),
    // RawVertexInstanced(RawVertexInstanced),
    _Model(ModelObject),
    ModelInstanced(ModelInstanced),
    _Dynamic(Box<dyn Render>),
}

impl Render for RenderObject {
    fn render_in_pass<'a, 'pass: 'a>(
        &'pass mut self,
        state: &mut PerPassState<'a, 'pass>,
        pass_index: u32,
    ) {
        match self {
            Self::RawVertex(raw) => raw.render_in_pass(state, pass_index),
            // Self::RawVertexInstanced(raw) => raw.render_in_pass(state),
            Self::_Model(model) => model.render_in_pass(state, pass_index),
            Self::ModelInstanced(model) => model.render_in_pass(state, pass_index),
            Self::_Dynamic(dynamic) => dynamic.render_in_pass(state, pass_index),
        }
    }
}

// impl RenderObject {
//     pub fn update_uniforms(&self) {
//         match &self {

//         }
//         let update_fn = |transform: &Transform| {
//             let mat = view_proj * transform.to_matrix();
//             queue.write_buffer(
//                 &self.model_uniform,
//                 0,
//                 bytemuck::cast_slice(&mat.to_cols_array()),
//             );
//         };
//         if always_update {
//             update_fn(&self.transform);
//         } else {
//             self.transform.if_dirty(update_fn);
//         }
//     }
// }

*/