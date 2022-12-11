// simple shader for vertices with a color at each vertex
// more complex shaders to come in the future!

// struct CameraUniform {
//     view_proj: mat4x4<f32>,
// };
// @group(2) @binding(3)
// var<uniform> camera: CameraUniform;


struct PointLight {
    // camera_transform: mat4x4<f32>,
    // diffuse_color is based on the texture sampling
    pos: vec3<f32>,
    ambient_power: f32,
    specular_color: vec3<f32>,
    specular_power: f32,
    exponent: f32,
}
@group(2) @binding(0)
var<uniform> light: PointLight;
@group(2) @binding(1)
var shadow_map: texture_depth_2d;
@group(2) @binding(2)
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
    @location(3) transform_col0: vec4<f32>,
    @location(4) transform_col1: vec4<f32>,
    @location(5) transform_col2: vec4<f32>,
    @location(6) transform_col3: vec4<f32>,
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
};

@vertex
fn vs_main(
    vertex: ModelVertex,
    instance: ModelInstance
) -> VertexOutput {
    var out: VertexOutput;
    let transform = mat4x4(
        instance.transform_col0,
        instance.transform_col1,
        instance.transform_col2,
        instance.transform_col3
    );
    let normal_matrix = mat4x4(
        instance.normal_matrix_col0,
        instance.normal_matrix_col1,
        instance.normal_matrix_col2,
        instance.normal_matrix_col3
    );
    out.tex_coords = vertex.texture;
    out.clip_position = transform * vec4<f32>(vertex.position, 1.0);
    out.pos = out.clip_position.xyz;
    let normal = (normal_matrix * vec4(vertex.normal, 0.0)).xyz;
    out.normal = normal * vec3(1.0, 1.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let kd = textureSample(model_tex, tex_sampler, in.tex_coords);
    // ambient power
    let c0 = light.ambient_power;
    // light source power
    let I0 = 5.0;
    // exponent
    let a = 50.0;
    let nhat = normalize(in.normal);
    let i = light.pos - in.pos;
    let ihat = normalize(i);
    let I = I0 / (length(i) * length(i) / 5.0 + 5.0);
    let ohat = -normalize(in.pos);
    let hhat = normalize(ihat + ohat);
    // return kd*c0 + max(0.0, I * (kd * dot(nhat, ihat) + pow(dot(nhat, hhat, a))));
    // let diffuse = kd.xyz * (I*max(dot(nhat, ihat), 0.0) + c0);
    // return vec4(abs(nhat), 1.0);

    // let shadow_pos = light.camera_transform * vec4(in.pos, 1.0);
    // let is_shadowed = shadow_pos.z >= textureSample(shadow_map, shadow_sampler, shadow_pos.xy);

    if (dot(nhat, ihat) > 0.0) {
        return vec4(kd.xyz * (I*dot(nhat, ihat) + c0) + vec3(I * pow(dot(nhat, hhat), a)), 1.0);
    } else {
        return vec4(kd.xyz*c0, 1.0);
    }
    // if (dot(nhat, ihat) >= 0.0) {
    //     return vec4(1.0, 1.0, 0.0, 1.0);
    //     // return vec4(diffuse + vec3(I * pow(dot(nhat, hhat), a)), 1.0);
    // } else if (dot(nhat, hhat) >= 0.0) {
    //     return vec4(0.0, 1.0, 0.0, 1.0);
    // } else if (dot(nhat, ihat) >= 0.0) {
    //     return vec4(1.0, 0.0, 0.0, 1.0);
    // } else {
    //     return vec4(0.0, 0.0, 0.0, 1.0);
    //     // return vec4(kd.xyz*c0, 1.0);
    // }
}


// struct CameraUniform {
//     view_proj: mat4x4<f32>,
// };
// @group(2) @binding(3)
// var<uniform> camera: CameraUniform;


// struct PointLight {
//     camera_transform: mat4x4<f32>,
//     // diffuse_color is based on the texture sampling
//     pos: vec3<f32>,
//     ambient_power: f32,
//     specular_color: vec3<f32>,
//     specular_power: f32,
//     exponent: f32,
// }
// @group(2) @binding(0)
// var<uniform> light: PointLight;
// @group(2) @binding(1)
// var shadow_map: texture_depth_2d;
// @group(2) @binding(2)
// var shadow_sampler: sampler;

// // struct ModelUniform {
// //     normal_matrix: mat4x4<f32>,
// // };
// // @group(3) @binding(0)
// // var<uniform> model: ModelUniform;
// @group(3) @binding(0)
// var model_tex: texture_2d<f32>;
// @group(3) @binding(1)
// var tex_sampler: sampler;

// struct ModelVertex {
//     @location(0) position: vec3<f32>,
//     @location(1) texture: vec2<f32>,
//     @location(2) normal: vec3<f32>,
// };

// struct ModelInstance {
//     @location(3) transform_col0: vec4<f32>,
//     @location(4) transform_col1: vec4<f32>,
//     @location(5) transform_col2: vec4<f32>,
//     @location(6) transform_col3: vec4<f32>,
//     @location(7) normal_matrix_col0: vec4<f32>,
//     @location(8) normal_matrix_col1: vec4<f32>,
//     @location(9) normal_matrix_col2: vec4<f32>,
//     @location(10) normal_matrix_col3: vec4<f32>,
// }

// struct VertexOutput {
//     @builtin(position) clip_position: vec4<f32>,
//     @location(0) pos: vec3<f32>,
//     @location(1) tex_coords: vec2<f32>,
//     @location(2) normal: vec3<f32>,
//     // @location(3) shadow_pos: vec3<f32>
// };

// @vertex
// fn vs_main(
//     vertex: ModelVertex,
//     instance: ModelInstance
// ) -> VertexOutput {
//     var out: VertexOutput;
//     let model_matrix = mat4x4(
//         instance.transform_col0,
//         instance.transform_col1,
//         instance.transform_col2,
//         instance.transform_col3
//     );
//     let normal_matrix = mat4x4(
//         instance.normal_matrix_col0,
//         instance.normal_matrix_col1,
//         instance.normal_matrix_col2,
//         instance.normal_matrix_col3
//     );
//     out.tex_coords = vertex.texture;
//     out.clip_position = camera.view_proj * model_matrix * vec4<f32>(vertex.position, 1.0);
//     out.pos = out.clip_position.xyz;
//     out.normal = (normal_matrix * vec4(vertex.normal, 0.0)).xyz;
//     // out.shadow_pos = ( * model_matrix * vec4(out.pos, 1.0)).xyz;
//     return out;
// }

// @fragment
// fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
//     let kd = textureSample(model_tex, tex_sampler, in.tex_coords);
//     // ambient power
//     let c0 = light.ambient_power;
//     // light source power
//     let I0 = 5.0;
//     // exponent
//     let a = 50.0;
//     let nhat = normalize(in.normal);
//     let i = light.pos - in.pos;
//     let ihat = normalize(i);
//     let I = I0 / (length(i) * length(i) / 5.0 + 5.0);
//     let ohat = -normalize(in.pos);
//     let hhat = normalize(ihat + ohat);
//     // return kd*c0 + max(0.0, I * (kd * dot(nhat, ihat) + pow(dot(nhat, hhat, a))));
//     // let diffuse = kd.xyz * (I*max(dot(nhat, ihat), 0.0) + c0);
//     // return vec4(abs(nhat), 1.0);

//     let shadow_pos = light.camera_transform * vec4(in.pos, 1.0);
//     let is_shadowed = shadow_pos.z >= textureSample(shadow_map, shadow_sampler, shadow_pos.xy);

//     if (dot(nhat, ihat) > 0.0) {
//         return vec4(kd.xyz * (I*dot(nhat, ihat) + c0) + vec3(I * pow(dot(nhat, hhat), a)), 1.0);
//     } else {
//         return vec4(kd.xyz*c0, 1.0);
//     }
//     // if (dot(nhat, ihat) >= 0.0) {
//     //     return vec4(1.0, 1.0, 0.0, 1.0);
//     //     // return vec4(diffuse + vec3(I * pow(dot(nhat, hhat), a)), 1.0);
//     // } else if (dot(nhat, hhat) >= 0.0) {
//     //     return vec4(0.0, 1.0, 0.0, 1.0);
//     // } else if (dot(nhat, ihat) >= 0.0) {
//     //     return vec4(1.0, 0.0, 0.0, 1.0);
//     // } else {
//     //     return vec4(0.0, 0.0, 0.0, 1.0);
//     //     // return vec4(kd.xyz*c0, 1.0);
//     // }
// }