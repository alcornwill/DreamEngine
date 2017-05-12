
struct Texture {
    vec4 transform;  // offset.x, offset.y, scale.x, scale.y,
    // #default [0.0, 0.0, 1.0, 1.0]?
    #ifdef TEXTURE_INDEXED
    int index;
    #endif
    sampler2DArray sampler;
    // blend_fact
};


vec2 transform_tex(vec2 texcoord, Texture tex) {
    return texcoord * tex.transform.zw + tex.transform.xy;
}

vec4 sample_tex(Texture tex, vec2 uv) {
    // abstracts usage of tex.index
    #ifdef TEXTURE_INDEXED
    return texture(tex.sampler, vec3(uv, tex.index));
    #else
    return texture(tex.sampler, uv);
    #endif
}

//vec4 sample_tex(Texture tex) {
//    // assume 'in vec2 texcoord' is defined...
//    vec2 uv = transform_tex(texcoord, tex);
//    return sample_tex(tex, uv);
//}

in vec3 vs_fs_normal;
in vec3 vs_fs_position;
in vec2 texcoord;
in float fog_amount;
in vec4 Color;

out vec4 outputColor;

uniform float cutoff;
uniform Texture tex0;
// #enum BLEND_MODE MIX_TEX0 ADD_TEX0 MULT_TEX0
uniform Texture tex1;
uniform Texture tex2;
uniform float t2_mix;  // the amount to mix/add/multiply t2 (hmm)
uniform samplerCube cube;