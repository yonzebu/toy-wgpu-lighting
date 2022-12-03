// simple shader for vertices with a color at each vertex
// more complex shaders to come in the future!

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct ColoredVertex {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    vertex: ColoredVertex
) -> VertexOutput {
    var out: VertexOutput;
    out.color = vertex.color;
    out.clip_position = camera.view_proj * vec4<f32>(vertex.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}