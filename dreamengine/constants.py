
# megashader defines
UNLIT = 'UNLIT'
USE_VERTEX_COLORS = 'USE_VERTEX_COLORS'
USE_OBJECT_COLOR = 'USE_OBJECT_COLOR'
#USE_TRANSPARENCY = 'USE_TRANSPARENCY'
USE_FOG = 'USE_FOG'
USE_BONES = 'USE_BONES'
USE_TEXTURES = 'USE_TEXTURES'  # would be better if was like NUM_TEXTURES?
USE_ENV_MAP = 'USE_ENV_MAP'

DEFAULT_SWITCHES = [
    UNLIT,
    # USE_VERTEX_COLORS,
    # USE_OBJECT_COLOR,
    # USE_FOG,
    # USE_BONES,
    # USE_TEXTURES,
    # USE_ENV_MAP,
]

# all programs use the same switches (can't define custom like in Unity)
SWITCH_MAP = {
    UNLIT:              0b0000001,
    USE_VERTEX_COLORS:  0b0000010,
    USE_OBJECT_COLOR:   0b0000100,
    USE_FOG:            0b0001000,
    USE_BONES:          0b0010000,
    USE_TEXTURES:       0b0100000,
    USE_ENV_MAP:        0b1000000
}

DEBUG_SHADERS = False
INVALIDATE_BONE_DATA = False
IMPORT_MATERIALS = False  # import materials from .fbx file

MATERIAL_BLACKLIST = [
    # use to ignore certain materials when importing
    'DefaultMaterial'  # blender's default material
]

SIZE_OF_FLOAT = 4
SIZE_OF_SHORT = 2

#MIPMAPS = 3  # hmm, everything is black ('out of memory' if use 4...)
MIPMAPS = 0
SAMPLES = 4
MAX_BONE_WEIGHTS_PER_VERTEX = 4
MAX_BONES = 24
MAX_POINT_LIGHTS = 4
# shader struct sizes
SIZEOF_FOG = 16 * 3
SIZEOF_DIRECTIONAL_LIGHT = 16 * 3
SIZEOF_POINT_LIGHT = 16 * 4

# these should be Settings?
FPS = 60
NATIVE_RESOLUTION = (320, 240)
SCREEN_SIZE = (320*2, 240*2)
GRAVITY = (0, -10, 0)

WHITE = (1.0, 1.0, 1.0, 1.0)
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
X_AXIS = (1, 0, 0)
Y_AXIS = (0, 1, 0)
Z_AXIS = (0, 0, 1)

MASK_DEFAULT = 1
MASK_UI = 1 << 1
MASK_SKYBOX = 1 << 2
MASK_OVERLAY = 1 << 3

# shader attributes are keywords
POSITION_ATTRIBUTE = b"position"
NORMAL_ATTRIBUTE = b"normal"
COLOR_ATTRIBUTE = b"color"
UV_ATTRIBUTE = b"uv"
BONE_INDEX_ATTRIBUTE = b"boneid"
BONE_WEIGHT_ATTRIBUTE = b"weight"

POSITION_STRIDE = 3
NORMAL_STRIDE = 3
COLOR_STRIDE = 4
UV_STRIDE = 2
BONE_INDEX_STRIDE = 4
BONE_WEIGHT_STRIDE = 4

# hmm
BUILTIN_ATTRIBUTES = {
    POSITION_ATTRIBUTE: POSITION_STRIDE,
    NORMAL_ATTRIBUTE: NORMAL_STRIDE,
    COLOR_ATTRIBUTE: COLOR_STRIDE,
    UV_ATTRIBUTE: UV_STRIDE,
    BONE_INDEX_ATTRIBUTE: BONE_INDEX_STRIDE,
    BONE_WEIGHT_ATTRIBUTE: BONE_WEIGHT_STRIDE
}

MVP_UNIFORM = b"MVP"
MV_UNIFORM = b"MV"
M_UNIFORM = b"M"
MP_UNIFORM = b"MP"
NORMALMATRIX_UNIFORM = b"NormalMatrix"
BONES_UNIFORM = b"bones[0]"

BUILTIN_UNIFORMS = [
    MVP_UNIFORM,
    MV_UNIFORM,
    M_UNIFORM,
    MP_UNIFORM,
    NORMALMATRIX_UNIFORM
]

NEAREST = 'nearest'
LINEAR = 'linear'
REPEAT = 'repeat'
MIRROR = 'mirror'
CLAMP_EDGE = 'clamp_edge'
CLAMP_BORDER = 'clamp_border'

FILTER_BINARY_MAP = {
    NEAREST: 0,
    LINEAR: 1
}

WRAP_BINARY_MAP = {
    REPEAT: 0,
    MIRROR: 1,
    CLAMP_EDGE: 2,
    CLAMP_BORDER: 3
}

DEFAULT_SAMPLER = 0b10000
