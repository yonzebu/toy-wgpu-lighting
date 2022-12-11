
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// TODO: change to const whenever wgpu (technically naga but that's an implementation detail) 
// updates to allow const expressions
// also you could probably remove the <> when that happens
let vertices = array<vec2<f32>, 3>(
    vec2<f32>(0.5, -0.5),
    vec2<f32>(0.0, 0.5),
    vec2<f32>(-0.5, -0.5)
);

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(in_vertex_index & 1u) - 0.5; 
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.2, 0.1, 1.0);
}