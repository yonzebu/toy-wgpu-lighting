use crate::app::render;

pub struct Material {
    pub name: String,
    pub diffuse_texture: super::Texture,
    pub bind_group: wgpu::BindGroup,
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub pos: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normals: [f32; 3],
}

impl render::Vertex for ModelVertex {
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
        array_stride: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: Self::ATTRIBUTES,
    };
}

// pub trait DrawModel<'a> {
//     fn draw_mesh(&mut self, mesh: &'a Mesh);
//     fn draw_mesh_instanced(&mut self, mesh: &'a Mesh, instances: Range<u32>);
// }
// impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
// where
//     'b: 'a,
// {
//     fn draw_mesh(&mut self, mesh: &'b Mesh) {
//         self.draw_mesh_instanced(mesh, 0..1);
//     }

//     fn draw_mesh_instanced(&mut self, mesh: &'b Mesh, instances: Range<u32>) {
//         self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
//         self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
//         self.draw_indexed(0..mesh.num_elements, 0, instances);
//     }
// }
