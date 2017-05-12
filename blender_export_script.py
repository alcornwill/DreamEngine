# export blender 3d scene into format accepted by DreamEngine (.json)
# todo a lot of validation / default values

from os.path import abspath, dirname
from os import getcwd, chdir
import bpy
import json

C = bpy.context
D = bpy.data

class SaveFileDialog(bpy.types.Operator):
    bl_idname = "object.custom_path"
    bl_label = "Save File"
    __doc__ = ""

    filename_ext = ".json"
    filter_glob = bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    filepath = bpy.props.StringProperty(name="File Path",
                                        description="Filepath used to export DreamEngine resources .json",
                                        maxlen=1024,
                                        default='')
    files = bpy.props.CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )

    def execute(self, context):
        # for file in self.files:
        #     print(file.name)  # selected files

        export_json(self.properties.filepath)

        return {'FINISHED'}

    def draw(self, context):
        self.layout.operator('file.select_all_toggle')

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

def write_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4)

MAP_EXTENSION = {
    'REPEAT': 'repeat',
    'CLIP': 'clamp_edge'
}
DEFAULT_EXTENSION = 'repeat'
MAP_ALPHA_BLEND = {
    'OPAQUE': 'opaque',
    'ADD': 'additive'
}
DEFAULT_ALPHA_BLEND = 'opaque'
MAP_BLEND_MODE = {
    'MIX': 'mix',
    'ADD': 'add',
    'MULTIPLY': 'multiply'
}
DEFAULT_BLEND_MODE = 'mix'

def map_blend_type(value, map_type, default):
    return map_type[value] if value in map_type else default

def get_texture_slot_data(texture_slot):
    if texture_slot is None: return None
    offset = texture_slot.offset
    scale = texture_slot.scale
    transform = [offset[0], offset[1], scale[0], scale[1]]
    blend_mode = texture_slot.blend_type  # 'MIX' 'ADD' 'MULTIPLY'
    blend_fact = texture_slot.diffuse_color_factor
    texture_name = texture_slot.texture.image.name  # NOTE: using the IMAGE name not texture name...

    blend_mode = map_blend_type(blend_mode, MAP_BLEND_MODE, DEFAULT_BLEND_MODE)

    return {
        "name": texture_name,
        "transform": transform,
        "blend_mode": blend_mode,
        "blend_fact": blend_fact
    }

def export_json(json_filepath):
    chdir(dirname(D.filepath))  # change directory so that relative paths work
    # json_filepath = D.filepath + '.json'
    scene = C.scene
    print('exporting all resources in scene "{}" to "{}"'.format(scene.name, json_filepath))
    if scene.render.engine != 'BLENDER_GAME':
        print('WARNING: {} is not compatible with DreamEngine export. Use BLENDER_GAME'.format(scene.render.engine))

    data = {
        "_THIS FILE IS AUTO GENERATED. CHANGES WILL BE LOST": None,
        "materials": {},
        "textures": {}
    }

    for texture in D.textures:
        # texture_name = texture.name
        sampling = texture.extension  # 'REPEAT' 'CLIP'
        mir_x = texture.use_mirror_x
        mir_y = texture.use_mirror_y
        # brightness = texture.intensity
        # contrast = texture.contrast
        # saturation = texture.saturation
        # r, g, b = texture.factor_red, texture.factor_green, texture.factor_blue

        # HMM i don't distinguish between textures and images in my engine, but it might be nice...
        image = texture.image
        name = image.name
        filepath = image.filepath
        use_alpha = image.use_alpha

        sampling = map_blend_type(sampling, MAP_EXTENSION, DEFAULT_EXTENSION)
        filepath = filepath.replace('\\', '/')

        data["textures"][name] = {
            "filepath": filepath,
            "use_alpha": use_alpha,
            "sampling": sampling,
            "mir_x": mir_x,
            "mir_y": mir_y
        }

    for material in D.materials:
        name = material.name
        unlit = material.use_shadeless
        use_transparency = material.use_transparency

        use_vertex_colors = material.use_vertex_color_paint
        # pass_index = material.pass_index
        use_object_color = material.use_object_color
        diffuse_color = material.diffuse_color
        emit = material.emit
        ambient = material.ambient
        translucency = material.translucency
        opacity = material.alpha

        double_sided = not material.game_settings.use_backface_culling
        # face = material.game_settings.face_orientation  # 'NORMAL' 'BILLBOARD'
        # ui = material.game_settings.text  # good idea?
        #   probably use custom property instead
        alpha_blend = material.game_settings.alpha_blend  # 'OPAQUE' 'ADD'

        # use_physics = material.game_settings.physics
        # friction = material.game_settings.physics.friction
        # elasticity = material.game_settings.physics.elasticity

        alpha_blend = map_blend_type(alpha_blend, MAP_ALPHA_BLEND, DEFAULT_ALPHA_BLEND)

        mat = {
            "alpha_blend": alpha_blend,
            "double_sided": double_sided,
            "unlit": unlit,
            "use_vertex_colors": use_vertex_colors,
            "use_object_color": use_object_color,
            "use_transparency": use_transparency,
            "diffuse_color": tuple(diffuse_color),
            "opacity": opacity,
            "emit": emit,
            "ambient": ambient,
            "translucency": translucency
        }

        texture_slots = [
            get_texture_slot_data(material.texture_slots[0]),  # tex 1
            get_texture_slot_data(material.texture_slots[1]),  # tex 2
            get_texture_slot_data(material.texture_slots[2])  # mask tex
        ]
        mat["texture_slots"] = texture_slots

        # shadow = material.use_cast_shadow

        data["materials"][name] = mat

    write_json(json_filepath, data)
    print("done")

# first prompt user to export .fbx
# bpy.ops.export_scene.fbx()
# now prompt export .json
# (if we could get the .fbx filepath they just exported to, we wouldn't need to open file dialog...)
bpy.utils.register_class(SaveFileDialog)
bpy.ops.object.custom_path('INVOKE_DEFAULT')
