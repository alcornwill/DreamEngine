#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.vert

out vec3 pos;

void main(void)
{
    // Clip-space position
    gl_Position = MVP * vec4(position, 1.0);
    // View-space normal and position
    pos = -position;
}