#version 330

#include dreamengine/shaders/shared.glsl
#include dreamengine/shaders/dreamengine.vert

uniform vec4 diffuse_color;
uniform vec4 object_color;

void main()
{
    texcoord = uv;

    #ifdef USE_BONES
    mat4 bonetransform = bone_transform();
    vec4 v_pos = bonetransform * vec4(position, 1.0f);
    #endif
    #ifndef USE_BONES
    vec4 v_pos = vec4(position, 1.0f);
    #endif
	gl_Position = MVP * v_pos;

	vec3 col = diffuse_color.rgb;
	#ifdef USE_VERTEX_COLORS
	col *= color;
	#endif
	#ifdef USE_OBJECT_COLOR
	col *= object_color.rgb;
    #endif

    #ifndef UNLIT
	vec3 scatteredLight = ambient;
	vec3 surface_normal = normalize(NormalMatrix * normal);
	directional_light(surface_normal, scatteredLight);
    point_lights(v_pos, surface_normal, scatteredLight);
    col *= scatteredLight;

    #ifdef USE_FOG
	float difference = 1 / (fogEnd - fogStart);
	fog_amount = min((gl_Position.z - fogStart) * difference, 1.0);
    #endif
    #endif

    #ifdef USE_ENV_MAP
    // View-space normal and position
    vs_fs_normal = mat3(MP) * normal; // should use NormalMatrix? also surely we need to use V?
    vs_fs_position = (MP * v_pos).xyz;
    #endif

    // note this value is unsaturated, we would saturate with min(color, vec4(1.0)
	Color = vec4(col, diffuse_color.a);  // note: diffuse_color alpha is the output alpha! (for now)
}
