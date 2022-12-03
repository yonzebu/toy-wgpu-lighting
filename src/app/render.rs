use std::ops::Neg;

use glam::*;
use winit::window::Window;

// use super::world::World;

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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColoredVert {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

impl ColoredVert {
    pub const ATTRIBUTES: &'static [wgpu::VertexAttribute] = &[
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
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBUTES,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Orthographic { w: f32, h: f32 },
    Perspective { fov_y: f32, aspect_ratio: f32 },
}

pub struct Camera {
    pub pos: Vec3,
    pub rot: Quat,
    pub near: f32,
    pub far: f32,
    pub proj: Projection,
}

impl Camera {
    pub fn as_uniform_data(&self) -> [[f32; 4]; 4] {
        self.transform_matrix().to_cols_array_2d()
    }

    /// Creates a camera using an orthographic projection
    ///
    /// # Panics
    /// Panics if either of `scale.x` or `scale.y` is 0.
    pub fn orthographic(pos: Vec3, rot: Quat, near: f32, scale: Vec3) -> Self {
        assert_ne!(scale.x, 0.0);
        assert_ne!(scale.y, 0.0);

        Self {
            pos,
            rot: rot.normalize(),
            near,
            far: near + scale.z,
            proj: Projection::Orthographic {
                w: scale.x,
                h: scale.y,
            },
        }
    }

    /// `fov_y` is in radians.
    pub fn perspective(
        pos: Vec3,
        rot: Quat,
        near: f32,
        far: f32,
        fov_y: f32,
        aspect_ratio: f32,
    ) -> Self {
        Self {
            pos,
            rot: rot.normalize(),
            near,
            far,
            proj: Projection::Perspective {
                fov_y,
                aspect_ratio,
            },
        }
    }

    /// Converts the camera space to the view space (camera looks at +z, +x is right and
    /// +y is up)
    pub fn view_matrix(&self) -> Mat4 {
        let t = self.rot.mul_vec3(Vec3::Y);
        let g = self.rot.mul_vec3(-Vec3::Z);
        let gxt = g.cross(t);
        Mat4::from_cols(
            // gxt.extend(gxt.dot(self.pos)).into(),
            // t.extend(t.dot(self.pos)).into(),
            // g.extend(g.dot(self.pos)).into(),
            gxt.extend(0.0),
            t.extend(0.0),
            g.extend(0.0),
            [gxt.dot(self.pos), t.dot(self.pos), g.dot(self.pos), 1.0].into()
        )
        // TODO: this is a fixed multiplication, can figure this out based on g, t, and g x t
        // Mat4::from_quat(self.rot) * Mat4::from_translation(self.pos)
    }

    /// Gets this camera's projection matrix, which converts view space coordinates
    /// (camera looks at +z, +x is right, +y is up) to wgpu normalized device coordinates.
    pub fn proj_matrix(&self) -> Mat4 {
        let n = self.near;
        let f = self.far;
        match self.proj {
            Projection::Orthographic { w, h } => {
                Mat4::orthographic_lh(-w / 2.0, w / 2.0, -h / 2.0, h / 2.0, self.near, self.far)
            }
            // Mat4::from_cols(
            //     2.0 / w * Vec4::X,
            //     2.0 / h * Vec4::Y,
            //     2.0 / (f - n) * Vec4::Z,
            //     [0.0, 0.0, (n + f) / (n - f), 1.0].into(),
            // ),
            Projection::Perspective {
                fov_y,
                aspect_ratio,
            } => {
                // i tried to implement it myself and then realized glam already provides methods for various projections
                Mat4::perspective_lh(fov_y, aspect_ratio, self.near, self.far)

                // let t = f32::tan(fov_y / 2.0) * self.near.abs();
                // let r = t * aspect_ratio;
                // // this is the transpose of the perspective transform
                // Mat4::from_cols_array_2d(&[
                //     [n / r, 0.0, 0.0, 0.0],
                //     [0.0, n / t, 0.0, 0.0],
                //     [0.0, 0.0, (n + f) / (f - n), 1.0],
                //     [0.0, 0.0, -2.0 * n * f / (n - f), 0.0],
                // ])
            }
        }
    }

    pub fn transform_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Converts a world/camera space position to normalized device coordinates
    pub fn transform(&self, pos: Vec3) -> Vec3 {
        self.transform_matrix().project_point3(pos)
    }

    /// This is probably unnecessary micro-optimization
    pub fn get_transform(&self) -> impl Fn(Vec3) -> Vec3 {
        let transform_mat = self.transform_matrix();
        move |vec| transform_mat.project_point3(vec)
    }
}

pub struct WgpuBase {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,

    pub(crate) surface: wgpu::Surface,
    /// Cached result of calling `surface.get_supported_formats()`
    pub(crate) surface_supported_formats: Vec<wgpu::TextureFormat>,
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
        // SAFETY: window gets moved into the returned value, and drop order guarantees it will be dropped after surface (and everything else)
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
            surface_supported_formats,
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
