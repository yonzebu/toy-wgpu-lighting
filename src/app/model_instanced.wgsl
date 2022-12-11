// simple shader for vertices with a color at each vertex
// more complex shaders to come in the future!

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(2) @binding(0)
var<uniform> camera: CameraUniform;

struct PointLight {
    view_proj: mat4x4<f32>,
    // diffuse_color is based on the texture sampling, not part of the uniform
    pos: vec3<f32>,
    ambient_power: f32,
    specular_color: vec3<f32>,
    specular_power: f32,
    exponent: f32,
}
@group(2) @binding(1)
var<uniform> light: PointLight;
// @group(2) @binding(0)
// var<uniform> shadow_pass_light: PointLight;

@group(2) @binding(2)
var shadow_map: texture_depth_2d;
@group(2) @binding(3)
var shadow_sampler: sampler;

// struct ModelUniform {
//     normal_matrix: mat4x4<f32>,
// };
// @group(3) @binding(0)
// var<uniform> model: ModelUniform;
@group(3) @binding(0)
var model_tex: texture_2d<f32>;
@group(3) @binding(1)
var tex_sampler: sampler;

struct ModelVertex {
    @location(0) position: vec3<f32>,
    @location(1) texture: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct ModelInstance {
    @location(3) model_matrix_col0: vec4<f32>,
    @location(4) model_matrix_col1: vec4<f32>,
    @location(5) model_matrix_col2: vec4<f32>,
    @location(6) model_matrix_col3: vec4<f32>,
    @location(7) normal_matrix_col0: vec4<f32>,
    @location(8) normal_matrix_col1: vec4<f32>,
    @location(9) normal_matrix_col2: vec4<f32>,
    @location(10) normal_matrix_col3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pos: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) shadow_pos: vec4<f32>,
};

@vertex
fn shadow_vs_main(
    vertex: ModelVertex,
    instance: ModelInstance
) -> @builtin(position) vec4<f32> {
    let model_matrix = mat4x4(
        instance.model_matrix_col0,
        instance.model_matrix_col1,
        instance.model_matrix_col2,
        instance.model_matrix_col3
    );
    // let out = light.view_proj * model_matrix * vec4<f32>(vertex.position, 1.0);
    // return vec4<f32>(out.xy, out.z / 4.0, out.w);
    return light.view_proj * model_matrix * vec4<f32>(vertex.position, 1.0);
}

@vertex
fn vs_main(
    vertex: ModelVertex,
    instance: ModelInstance
) -> VertexOutput {
    var out: VertexOutput;
    let model_matrix = mat4x4(
        instance.model_matrix_col0,
        instance.model_matrix_col1,
        instance.model_matrix_col2,
        instance.model_matrix_col3
    );
    let normal_matrix = mat4x4(
        instance.normal_matrix_col0,
        instance.normal_matrix_col1,
        instance.normal_matrix_col2,
        instance.normal_matrix_col3
    );
    out.tex_coords = vertex.texture;
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(vertex.position, 1.0);
    out.pos = out.clip_position.xyz;
    let normal = (normal_matrix * vec4(vertex.normal, 0.0)).xyz;
    out.normal = normal * vec3(1.0, 1.0, 1.0);
    // out.shadow_pos = light.view_proj * model_matrix * vec4<f32>(vertex.position, 1.0);
    
    // let shadow_pos_out = light.view_proj * model_matrix * vec4<f32>(vertex.position, 1.0);
    // out.shadow_pos = vec4<f32>(shadow_pos_out.xy, shadow_pos_out.z / 4.0, shadow_pos_out.w);
    out.shadow_pos = light.view_proj * model_matrix * vec4<f32>(vertex.position, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let kd = textureSample(model_tex, tex_sampler, in.tex_coords);
    let nhat = normalize(in.normal);
    let i = light.pos - in.pos;
    let ihat = normalize(i);
    let I = light.specular_power / (length(i) * length(i) / 5.0 + 5.0);
    let ohat = -normalize(in.pos);
    let hhat = normalize(ihat + ohat);

    // let shadow_pos = light.view_proj * vec4(in.pos, 1.0);
    // let shadow_map_sample = textureSample(shadow_map, shadow_sampler, in.shadow_pos.xy / in.shadow_pos.w);
    // let is_shadowed = in.shadow_pos.z >= textureSample(shadow_map, shadow_sampler, in.shadow_pos.xy);

    // return vec4(vec3(shadow_map_sample), 1.0);
    // return vec4(vec3(1.0 / in.shadow_pos.w), 1.0);

    if (dot(nhat, ihat) > 0.0) {
        let diffuse = kd.xyz * (I * dot(nhat, ihat) + light.ambient_power);
        let specular = vec3(I * pow(dot(nhat, hhat), light.exponent));
        return vec4(diffuse + specular, 1.0);
    } else {
        return vec4(kd.xyz * light.ambient_power, 1.0);
    }
}

