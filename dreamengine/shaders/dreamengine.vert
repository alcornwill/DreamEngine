// contains all builtin attributes and uniforms
// (anything that isn't used will be optimized away by compiler)

// i am relying on compiler optimization quite a lot here... (it does make the code neater)
in vec3 position;
in vec3 normal;
in vec3 color;
in vec2 uv;
in ivec4 boneid;
in vec4 weight;

const int MAX_BONE_WEIGHTS_PER_VERTEX = 4;
const int MAX_BONES = 24;

out vec3 vs_fs_normal;  // surface normal and view-space position
out vec3 vs_fs_position;
out vec2 texcoord;
out float fog_amount;
out vec4 Color;

uniform mat4 MVP;
uniform mat4 MP;
uniform mat4 M;
uniform mat3 NormalMatrix;
uniform mat4 bones[MAX_BONES];

mat4 bone_transform()
{
    mat4 transform = bones[boneid[0]] * weight[0];
    transform += bones[boneid[1]] * weight[1];
    transform += bones[boneid[2]] * weight[2];
    transform += bones[boneid[3]] * weight[3];
    return transform;
}

void directional_light(vec3 surface_normal, inout vec3 scatteredLight)
{
    vec3 direction = normalize(directionalLight.direction);
    float diffuse = max(0.0, dot(surface_normal, direction));
    scatteredLight += directionalLight.ambient * directionalLight.color;
	scatteredLight += directionalLight.color * diffuse;
}

void point_lights(vec4 v_pos, vec3 surface_normal, inout vec3 scatteredLight)
{
    vec4 p_pos = M * v_pos;
	for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
	    PointLight light = pointLights[i];
        vec3 direction = light.position - vec3(p_pos);
        float distance = length(direction);
        float attenuation = 1.0 / (light.attenuation * distance * distance);
        float diffuse = max(0.0, dot(surface_normal, direction));
        scatteredLight += light.ambient * light.color;
        scatteredLight += light.color * diffuse * attenuation;
    }
}