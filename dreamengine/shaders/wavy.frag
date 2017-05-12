#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.frag
uniform sampler2D tex;

void main()
{
    vec2 uv = texcoord;
    uv.x += sin(uv.y * 4*2*3.14159 + t) / 100;
    vec4 col = texture(tex, uv);
    outputColor = col;
}

