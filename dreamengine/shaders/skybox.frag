#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.frag

in vec3 pos;

void main(void)
{
    outputColor = texture(cube, pos);
}