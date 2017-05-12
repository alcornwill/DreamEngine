#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.vert

out vec4 pos;

void main()
{
    float f = gl_InstanceID * 0.3927;
    float sn = sin(f) * 2;
    float cs = cos(f) * 2;
    vec3 p = position + vec3(sn, cs, 0);
	pos = MVP * vec4(p, 1.0f);
	gl_Position = pos;
}
