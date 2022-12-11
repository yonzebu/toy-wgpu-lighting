// simple shader for vertices with a color at each vertex
// more complex shaders to come in the future!

// struct CameraUniform {
//     view_proj: mat4x4<f32>,
// };
// @group(0) @binding(0)
// var<uniform> camera: CameraUniform;

struct ModelUniform {
    data: mat4x4<f32>,
};
@group(3) @binding(0)
var<uniform> model: ModelUniform;
@group(3) @binding(1)
var model_tex: texture_2d<f32>;
@group(3) @binding(2)
var tex_sampler: sampler;

struct ModelVertex {
    @location(0) position: vec3<f32>,
    @location(1) texture: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    vertex: ModelVertex
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = vertex.texture;
    out.clip_position = model.data * vec4<f32>(vertex.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(model_tex, tex_sampler, in.tex_coords);
}