#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.vert

void main()
{
	gl_Position = vec4(position, 1.0f);
	texcoord = uv;  // can technically use position as uv? whatever
}
