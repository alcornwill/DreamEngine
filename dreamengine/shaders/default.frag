#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.frag

void main()
{
    vec4 outColor = Color;

    #ifdef USE_TEXTURES
    vec2 uv1 = transform_tex(texcoord, tex0);
    vec4 col1 = sample_tex(tex0, uv1);

    vec2 uv2 = transform_tex(texcoord,tex1);
    vec4 col2 = sample_tex(tex1, uv2);

    vec2 uv_mask = transform_tex(texcoord, tex2);
    vec4 col_mask = sample_tex(tex2, uv_mask);

    // todo abstract the blend equations into functions, (use preprocessor just do blend(col, tex)?)
    #ifdef OVER_T2
    vec4 col = mix(col1, col2, col_mask.a);
    col = mix(col1, col, t2_mix);
    #endif
    #ifdef MIX_T2
    vec4 col = mix(col1, col2, t2_mix);
    #endif
    #ifdef ADD_T2
    col2 = mix(vec4(0.0), col2, t2_mix);
    vec4 col = col1 + col2;
    #endif
    #ifdef MULT_T2
    vec4 col = col1 * col2;
    col = mix(col1, col, t2_mix);
    #endif
    outColor *= col;
    #endif

    #ifdef USE_ENV_MAP
    // Calculate the texture coordinate by reflecting the
    // view-space position around the surface normal.
    vec3 tc = reflect(-vs_fs_position, normalize(vs_fs_normal));
    // Sample the texture and color the resulting fragment
    // a golden color.
    outColor *= texture(cube, tc);
    #endif

    #ifndef UNLIT
    #ifdef USE_FOG
    outColor = mix(outColor, vec4(fogColor, 1.0), fog_amount);
    #endif
    #endif

    outputColor = outColor;
}

