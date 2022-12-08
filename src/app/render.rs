use std::{sync::Arc, any::Any};

use glam::*;
use winit::window::Window;

use super::{camera::Camera, utils::Dirtiable};

// use super::utils::{IdMap, IsId};

// #[derive(Clone, PartialEq, Eq, Debug)]
// pub enum FrameSkipCause {
//     SurfaceLost,
//     SurfaceOutdated,
//     SurfaceTimeout,
// }

// #[derive(Clone, PartialEq, Eq, Debug)]
// pub enum RenderError {
//     /// corresponds to only wgpu::SurfaceError::OutOfMemory right now
//     OutOfMemory,
//     /// frame was skipped but it isn't a fatal error and is expected to be fixed in the next couple of frames
//     FrameSkipped(FrameSkipCause),
// }

// impl From<wgpu::SurfaceError> for RenderError {
//     fn from(value: wgpu::SurfaceError) -> Self {
//         match value {
//             wgpu::SurfaceError::Lost => Self::FrameSkipped(FrameSkipCause::SurfaceLost),
//             wgpu::SurfaceError::Outdated => Self::FrameSkipped(FrameSkipCause::SurfaceOutdated),
//             wgpu::SurfaceError::Timeout => Self::FrameSkipped(FrameSkipCause::SurfaceTimeout),
//             wgpu::SurfaceError::OutOfMemory => Self::OutOfMemory,
//         }
//     }
// }

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

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.queue_resize(new_size);
        self.apply_resize()
    }

    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    // fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    //     if self.is_window_init && new_size.width > 0 && new_size.height > 0 {
    //         self.size = new_size;
    //         self.surface_config.width = self.size.width;
    //         self.surface_config.height = self.size.height;
    //         self.surface.configure(&self.device, &self.surface_config);
    //     }
    // }

    //     pub fn render(&mut self, _: &World) -> Result<(), RenderError> {
    //         // this is the main source of early termination of rendering a frame
    //         let surface_texture = match self.surface.get_current_texture() {
    //             Ok(texture) => Ok(texture),
    //             Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
    //                 self.resize(self.size);
    //                 // try to get the texture again but only once
    //                 self.surface
    //                     .get_current_texture()
    //                     // have to put a type annotation somewhere, and this is ugly but it works i guess
    //                     .map_err(|e| Into::<RenderError>::into(e))
    //             }
    //             Err(e) => Err(e.into()),
    //         }?;

    //         let view = surface_texture
    //             .texture
    //             .create_view(&wgpu::TextureViewDescriptor::default());
    //         let mut encoder = self
    //             .device
    //             .create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //                 label: Some("Render Encoder"),
    //             });
    //         {
    //             let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    //                 label: Some("Render Pass"),
    //                 color_attachments: &[Some(wgpu::RenderPassColorAttachment {
    //                     view: &view,
    //                     resolve_target: None,
    //                     ops: wgpu::Operations {
    //                         load: wgpu::LoadOp::Clear(wgpu::Color {
    //                             r: 0.1,
    //                             g: 0.2,
    //                             b: 0.3,
    //                             a: 1.0,
    //                         }),
    //                         store: true,
    //                     },
    //                 })],
    //                 depth_stencil_attachment: None,
    //             });
    //         }

    //         self.queue.submit(std::iter::once(encoder.finish()));
    //         surface_texture.present();

    //         Ok(())
    //     }
}

pub struct Texture {
    pub handle: wgpu::Texture,
    pub sampler: Option<wgpu::Sampler>,
    pub view: wgpu::TextureView,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(base: &WgpuBase, make_sampler: bool) -> Self {
        let handle = base.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: base.surface_config.width,
                height: base.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let view = handle.create_view(&wgpu::TextureViewDescriptor {
            ..Default::default() // label: None,
                                 // format: ,
                                 // dimension: ,
                                 // aspect: wgpu::TextureAspect::DepthOnly,
                                 // base_mip_level:,
                                 // mip_level_count,
                                 // base_array_layer,
                                 // array_layer_count,
        });

        Texture {
            handle,
            sampler: None,
            view,
        }
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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub pos: [f32; 3],
    pub tex_coords: [f32; 3],
    pub normals: [f32; 3],
}

impl Vertex for ModelVertex {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute] = &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
            shader_location: 2,
        },
    ];
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBUTES,
    };
}

// #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct PipelineId(usize);
// impl From<usize> for PipelineId {
//     fn from(value: usize) -> Self {
//         Self(value)
//     }
// }
// impl From<PipelineId> for usize {
//     fn from(id: PipelineId) -> Self {
//         id.0
//     }
// }
// impl IsId for PipelineId {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub pos: Vec3,
    pub rot: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn translate_by(self, v: Vec3) -> Self {
        Self {
            pos: self.pos + v,
            ..self
        }
    }
    pub fn rotate_by(self, q: Quat) -> Self {
        Self {
            rot: (self.rot * q).normalize(),
            ..self
        }
    }
    pub fn scale_by(self, s: Vec3) -> Self {
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

pub struct RenderObject {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub num_vertices: usize,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub num_indices: usize,
    pub pipeline: Arc<wgpu::RenderPipeline>,
    pub model_bind_group: wgpu::BindGroup,
    pub model_uniform: wgpu::Buffer,
    pub transform: Dirtiable<Transform>,
    // low-effort but it works
    pub render_hook: Option<Box<dyn FnMut(&Self, &mut wgpu::RenderPass, &wgpu::Queue)>>,
}

impl RenderObject {
    pub fn update_uniforms(&self, camera: &Camera, always_update: bool, queue: &wgpu::Queue) {
        let update_fn = |transform: &Transform| {
            let mat = camera.transform_matrix() * transform.to_matrix();
            queue.write_buffer(
                &self.model_uniform,
                0,
                bytemuck::cast_slice(&mat.to_cols_array()),
            );
        };
        if always_update {
            update_fn(&self.transform);
        } else {
            self.transform.if_dirty(update_fn);
        }
    }
}
