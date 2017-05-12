#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.frag
uniform sampler2D tex;  // should use tex0

void main()
{
    outputColor = texture(tex, texcoord);
}

