#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.vert

void main()
{
    gl_Position = MVP * vec4(position, 1.0f);
	Color = vec4(color, 1.0);
}
