
#define OVER_T2
//#define MIX_T2
//#define ADD_T2
//#define MULT_T2
#define TEXTURE_INDEXED  // when defined, all textures must be indexed...

const int MAX_POINT_LIGHTS = 4;

struct DirectionalLight {
    vec3 color;
    // unused float here
    vec3 direction;
    float ambient;
};

struct PointLight {
    vec3 color;
    float attenuation;
    vec3 position;
    float ambient;
};

// not just lighting anymore...
layout(std140) uniform lighting {
    // unused vec3 here
    float t;
    vec3 ambient;
    float fogStart;
    vec3 fogColor;
    float fogEnd;
    DirectionalLight directionalLight;
    PointLight pointLights[MAX_POINT_LIGHTS];
};

