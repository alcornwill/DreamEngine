#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.frag

in vec4 pos;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
	float val = rand(pos.xy);  // could also use gl_FragCoord
    outputColor = vec4(val, val, val, 1.0);
}

