import logging
from os.path import isfile, abspath
import collections
import json
from functools import reduce
from pprint import pprint

import math

from dreamengine.quaternion import Quaternion
from dreamengine.utils import *
from dreamengine.constants import *
import dreamengine.log as log
import dreamengine.basegraphics as gfx
from dreamengine.mathutils import *  # too low level for here?

R = None

class Object3D:
    root = None
    # an Entity with 3D transformation, in a hierarchy
    def __init__(self, position=(0.0, 0.0, 0.0), rotation=(0, 0, 0), scale=(1.0, 1.0, 1.0),
                 parent=None, children=None, name='default', static=False):
        self.name = name
        self.parent = None
        self.children = []
        self._position = tuple(position)
        self._rotation = None
        self._scale = tuple(scale)
        self.transform_changed = True
        self.model = None
        self.mvp = None  # I wonder if vp stuff should be handled by VPComponent, but it's fine...
        self.mv = None
        self.mp = None
        self.static = static  # maybe could also have 'dynamic' and cache matrix, only change 'if self.moved'
        self.offset = None  # offset matrix used for skeletal animation bones
        self.inverse_global = None  # also used for skeletal animation
        self.anim_mat = None  # yet another matrix that doesn't really belong here...

        # init
        if isinstance(rotation, Quaternion):
            self.rotation = rotation
        else:
            self.rotation = Quaternion.from_euler(*rotation)

        if parent:
            self.set_parent(parent)
        elif self.root:
            self.set_parent(self.root)
            # else is root, has no parent
        if children:
            self.add_children(children)

    def get_position(self):
        return self._position

    def set_position(self, value):
        self.transform_changed = True
        self._position = value

    def get_rotation(self):
        return self._rotation

    def set_rotation(self, value):
        self.transform_changed = True
        self._rotation = value

    def get_scale(self):
        return self._scale

    def set_scale(self, value):
        self.transform_changed = True
        self._scale = value

    position = property(get_position, set_position)
    rotation = property(get_rotation, set_rotation)
    scale = property(get_scale, set_scale)

    def delete(self):
        for child in self.children:
            child.delete()
        if self.parent:
            self.parent.children.remove(self)

    def set_bone_mats(self, offset, inverse_global):
        self.offset = offset
        self.inverse_global = inverse_global

    def set_parent(self, parent):
        if self.parent:
            self.parent.children.remove(self)
        self.parent = parent
        parent.children.append(self)

    def add_children(self, children):
        for child in children:
            child.set_parent(self)

    def move(self, x, y, z):
        p = self.position
        p2 = (p[0] + x, p[1] + y, p[2] + z)
        self.position = p2

    def add_scale(self, x, y, z):
        s = self.scale
        s2 = (s[0] + x, s[1] + y, s[2] + z)
        self.scale = s2

    def update_matrix(self):
        mats = []
        if self.parent:
            mats.append(self.parent.model)
        mats.append(translate(self.position))
        mats.append(self.rotation.to_matrix())
        mats.append(scale(self.scale))
        if self.anim_mat is not None:
            mats.append(self.anim_mat)
        self.model = matrix_multiply(mats)

    def update_matrix_r(self, force=False):
        force = force or self.transform_changed
        if not self.static or force:
            self.update_matrix()
        self.transform_changed = False

        for c in self.children:
            c.update_matrix_r(force)

    def update_mvp(self, camera):
        self.mv = camera.view @ self.model
        self.mvp = camera.projection @ self.mv
        self.mp = camera.projection @ self.model  # hmm, not every object needs this

    def update_mvp_r(self, camera):
        self.update_mvp(camera)
        for c in self.children: c.update_mvp_r(camera)

    def frustrum_test(self):
        # todo should use object bounds instead of object origin
        # also we could do like they do in majoras mask and make objects fade in and out
        # so we call fade_in or fade_out and it uses coroutines to animate
        return frustrum_test(self.mvp)

    def update_matrix_up_r(self, stop_at):
        # recursively update matrix up until reach stop_at
        if stop_at == self:
            # we aren't recursively updating all children, but we have changed
            for child in self.children:
                child.transform_changed = True
        else:
            self.parent.update_matrix_up_r(stop_at)
        self.update_matrix()
        self.transform_changed = False

    def parent_transform_changed_r(self, changed=None):
        # find the topmost parent that changed (if any)
        if self.transform_changed:
            changed = self
        if self.parent:
            return self.parent.parent_transform_changed_r(changed)
        return changed

    def update_matrix_up(self):
        # do minimal number of matrix updates to get our updated matrix
        changed = self.parent_transform_changed_r()
        if changed:
            self.update_matrix_up_r(stop_at=changed)

    def world_transform(self):
        # this is tricky because the transform may have changed, but haven't updated matrix yet
        self.update_matrix_up()
        return decompose_matrix(self.model)

class Counter:
    def __init__(self):
        self.count = -1

    def next(self):
        self.count += 1
        return self.count

class Sorted:
    def __init__(self, cls):
        self.id = None
        self.order = 0

        # init
        self.id = R.new_id(cls)
        # self.set_order(0)
        #   ...
        #   self.order = order
        #   self.index = self.order << 8 | self.id

    def on_set_job(self, queue):
        pass

    def on_unset_job(self, queue):
        pass

class Job:
    def __init__(self, func, desc):
        self.exec = func
        self.desc = desc if desc else ""  # description

class DrawJob(Job):
    renderer = None

    def __init__(self, func, sorters, desc):
        super().__init__(func, desc)
        self.order = 0
        self.id = 0
        self.sort = sorters

        # init
        self.init_id()
        DrawJob.renderer.draw_queue.append(self)

    def init_id(self):
        # this is pretty stupid, I would prefer if only had one sort index
        # all should be sorted by id, but only some need to be sorted by user specified 'order' as well
        #   (pass, rlayer, cam?)
        enc = 0
        order = 0
        stride = 16  # note: this means only works with id's up to 256... (could get stride from SortObj)
        for sort in self.sort:
            enc |= sort.order << order * stride
            order += 1
        self.order = enc

        id = 0
        order = 0
        stride = 16
        for sort in self.sort:
            id |= sort.id << order * stride
            order += 1
        self.id = id

class Renderer:
    def __init__(self):
        self.draw_queue = []
        self.passthrough = None
        self.width = None
        self.height = None

    def init_rendering(self):
        Sampler(NEAREST, REPEAT, REPEAT)  # this sampler is used for post processing
        PostProcessor()  # create passthrough postprocessor
        # should create own framebuffer?
        gfx.init()

    def draw(self):
        # process the draw_queue
        for job in self.draw_queue: job.exec()  # could this be used for other kinds of jobs?

    def sort_queue(self):
        self.draw_queue.sort(key=lambda job: job.id)
        self.draw_queue.sort(key=lambda job: job.order)  # this only works because queue is really draw_queue...
        self.queue_first_pass()

    def init_post_processing(self):
        for post in R.post_processors.values():
            post.init()  # we need to initialize after programs have been created (after 'load_scene')
        self.passthrough = R.post_processors['passthrough']

    def queue_first_pass(self):
        if not any(self.draw_queue): return []
        queue_copy = []
        prev = [None] * len(self.draw_queue[0].sort)  # list of previous sortobj
        for job in self.draw_queue:
            for i, sort in reversed(list(enumerate(job.sort))):
                if not prev[i]:
                    sort.on_set_job(queue_copy)
                    prev[i] = sort
                elif prev[i].id != sort.id:
                    prev[i].on_unset_job(queue_copy)
                    sort.on_set_job(queue_copy)
                    prev[i] = sort
            queue_copy.append(job)
        for sort in prev:
            sort.on_unset_job(queue_copy)
        self.draw_queue = queue_copy

    def blit(self):
        # (this could be a job)
        # this blends the framebuffers together in order, blasts the last one to window
        # hmm, if you wanted to post process a framebuffer and then use that render texture in the render
        gfx.enable_blending()
        gfx.set_additive_blending()
        l = list(R.render_layers.values())
        l.sort(key=lambda lyr: lyr.depth)
        prev = l[0]
        for rlayer in l[1:]:
            self.passthrough.draw(prev.render_texture, rlayer.fbuf)  # need to use passthrough shader to blend
            prev = rlayer
        # blit the last one to the window
        gfx.blit_framebuffer(prev.fbuf, prev.width, prev.height,
                             0, self.width, self.height)

    def resize(self, width, height):
        self.width = width
        self.height = height
        # todo reconstruct all framebuffers

class Group:
    def __init__(self):
        self.items = []

    def item_match(self, item):
        raise NotImplemented()

    @staticmethod
    def create(item):
        raise NotImplemented()

class Grouper:
    # arrays of things need to be 'grouped' by certain parameters (e.g. Meshes, Textures, Cubemaps)
    @staticmethod
    def group_items(cls, items):
        groups = []
        for item in items:
            group = Grouper.find_matching_group(groups, item)
            if not group:
                # create new
                group = cls.create(item)
                groups.append(group)
            group.items.append(item)
        return groups

    @staticmethod
    def find_matching_group(groups, texture):
        for group in groups:
            if group.item_match(texture):
                return group

class Resources:
    def __init__(self):
        global R
        R = self  # global

        # self.objects = {}  # if Entity3D inherited Object3D, could manage objects?
        self.passes = {}
        self.programdata = {}
        self.programs = {}
        # self.meshdata
        self.meshproperties = {}
        self.multimeshdata = DictionaryOfLists()
        self.pooledmeshdata = {}
        self.meshes = {}  # MeshBuffer?
        self.vaos = {}
        self.materials = {}
        self.texturedata = {}
        self.arraytextures = {}
        self.texture_arrays = {}
        self.cubemaps = {}
        self.render_layers = {}
        self.post_processors = {}
        self.animationdata = {}
        self.cameradata = {}
        self.lightdata = {}
        self.nodedata = {}  # or rootnodedata?
        # self.bonedata = {}

        self.uniform_blocks = {}
        self.samplers = {}

        self.counters = {}

        # init
        self.init_counters()

    @staticmethod
    def get_resource(lst, res):
        if isinstance(res, str):
            return lst[res]
        return res

    def init_counters(self):
        self.create_counter(Pass)
        self.create_counter(Material)
        self.create_counter(RenderLayer)
        self.create_counter(Vao)
        self.create_counter(Program)
        self.create_counter(Camera)

    def create_counter(self, cls):
        self.counters[cls] = Counter().next

    def new_id(self, cls):
        return self.counters[cls]()

    def load_resources(self, scene_json):
        # todo should do this in a coroutine, yield to defer loading times
        # I think you should be able to load a scene from pickled objects as well, or create them by hand if you want
        data = json.loads(scene_json)
        scene_name = data["name"]
        log.basic.info('loading resources "{}"'.format(scene_name))
        # programs
        # (if you were going for full GUI controlled engine it would probably be better to create programs automatically from material)
        #   but since I'm just writing json it actually helps to not have to duplicate paths everywhere...
        if "programs" in data:
            for name, program in data["programs"].items():
                defaults = dict_get_value(program, "defaults", {})
                ProgramData(program["vert"], program["frag"], defaults=defaults, name=name)

        # 3d files
        for file_3d in data["3D_files"]:
            path = file_3d["path"]
            flags = file_3d["flags"]

            # since materials and textures are 'engine specific' can only import from .json
            # look for 3d file .json extension resources
            json_path = change_extension(path, '.json')
            ext_data = None
            if "materials" in flags or "textures" in flags:
                if isfile(json_path):
                    ext_data = json.loads(read_file(json_path))

                    # materials
                    if "materials" in flags:
                        materials = ext_data["materials"]
                        for name, material in materials.items():
                            Material.create(name=name, **material)

                    if "textures" in flags:
                        textures = ext_data["textures"]
                        for name, texture in textures.items():
                            ArrayTexture.create(name=name, **texture)
                else:
                    log.basic.warning(
                        '.json resources extension file "{}" not found, materials and textures cannot be imported from 3d file'.format(
                            json_path))

            with gfx.MeshLoader(path) as loader:
                # todo shouldn't have to poke around in assimp types here
                #   meshloader should probably create assets automatically?
                #   but then doesn't know about our types, because it's a basegrahpics thing...

                # always import nodes
                if log.basic.isEnabledFor(logging.DEBUG):
                    gfx.MeshLoader.print_node_r(loader.scene.rootnode, "")
                rootnode = Node.create_r(None, loader.scene.rootnode)
                self.nodedata[path] = rootnode
                log.basic.info('created rootnode "{}"'.format(path))

                if "meshes" in flags:
                    for mesh in loader.scene.meshes:
                        unique_name = self.unique_mesh_name(mesh)
                        # if POOL_MESHES?
                        if unique_name in self.pooledmeshdata:
                            log.basic.warning(
                                'duplicate mesh name "{}"'.format(unique_name))  # todo do this for all resources?...
                        default_material = self.default_material_name(loader, mesh)
                        pooledmeshdata = PooledMeshData(unique_name, default_material, *loader.get_data(mesh))

                        if any(mesh.bones):
                            # it's weird that each mesh has a reference to same set of bones and rootbone...
                            pooledmeshdata.bones = mesh.bones  # todo should create own Bone class
                            # if a mesh has bones it must have an armature
                            root, armature, mesh = self.get_armature(rootnode, mesh.name)
                            pooledmeshdata.rootbone = armature.children[0]

                        # add an entry to the .multimeshdata dictionary
                        self.multimeshdata[mesh.name].append(unique_name)

                # create animation data
                if "animations" in flags:
                    for animation in loader.scene.animations:
                        AnimationData.create(animation)

                if "cameras" in flags:
                    for camera in loader.scene.cameras:
                        name = camera.name
                        aspect = camera.aspect
                        far = camera.clipplanefar
                        near = camera.clipplanenear
                        fov = camera.horizontalfov
                        # transform = camera.transformation
                        # up = camera.up
                        CameraData(name=name, aspect=aspect, far=far, near=near, fov=fov)

                if "lights" in flags:
                    for light in loader.scene.lights:
                        name = light.name
                        type_ = light.type  # 1: directional  2: point  (in blender at least)
                        # todo i think i might actually include these attenuation parameters in shader
                        attenuationconstant = light.attenuationconstant
                        attenuationlinear = light.attenuationlinear
                        attenuationquadratic = -light.attenuationquadratic  # this is negative...
                        ambient = light.colorambient  # numpy array (thought would be float?...)
                        diffuse = light.colordiffuse  # numpy array (a fucking big number)
                        direction = light.direction  # numpy array
                        # LightData(name=name, type_=type_, attenuation=attenuationquadratic, ambient=ambient,
                        #           diffuse=diffuse, direction=direction)
                        LightData(name=name, type_=type_, attenuation=attenuationquadratic)

        # textures
        if "textures" in data:
            for name, texture in data["textures"].items():
                ArrayTexture.create(name, **texture)

        # cubemaps
        if "cubemaps" in data:
            for name, textures in data["cubemaps"].items():
                faces = [TextureData(name + '.' + str(i), path) for i, path in enumerate(textures)]
                Cubemap(faces, name=name)

        # materials
        if "materials" in data:
            materials = data["materials"]
            for name, material in materials.items():
                Material.create(name=name, **material)

        # render layers
        if "render layers" in data:
            for name, rlayer in data["render layers"].items():
                RenderLayer(name=name, **rlayer)

    @staticmethod
    def find_mesh_node_r(node, mesh_name):
        for mesh in node.meshes:
            if mesh.name == mesh_name:
                return node
        for child in node.children:
            res = Resources.find_mesh_node_r(child, mesh_name)
            if res: return res

    @staticmethod
    def get_armature(rootnode, mesh_name):
        mesh = Resources.find_mesh_node_r(rootnode, mesh_name)
        # armature structure always like this: (always? only with blender?)
        #   (otherwise would be the (only?) 'sibling' of mesh)
        #   root
        #       armature
        #           hip, ...
        #       mesh
        root = mesh.parent
        armature = root.children[0]
        return root, armature, mesh

    @staticmethod
    def default_material_name(loader, mesh):
        # this is the only thing we use assimp materials for
        default_material = loader.scene.materials[mesh.materialindex].properties["name"]
        return default_material if default_material in R.materials else 'default'
        # assume material already loaded, else use 'default'

    @staticmethod
    def unique_mesh_name(mesh):
        # append material index
        return mesh.name + str(mesh.materialindex)

    def load_scene(self, scene_name):
        # create resources specific to the combination of loaded resources
        # Mesh.create_from_pooledmeshdata(self.MeshData, scene_name)
        Mesh.create_from_pooledmeshdata(self.pooledmeshdata.values(), scene_name)
        TextureArray.create_from_textures(self.arraytextures.values())
        for material in self.materials.values():
            material.init_textures()  # we can init now that array textures are created
            # we can create entities now as well

    def unload_resources(self):
        # todo it should probably delete all entities as well (override)
        #   that's actually really annoying though. need to serialize/deserialize?
        #       or probably easier to go round and replace references
        #       (which means we need to store resource _names_ in the entity or something)
        #   also need to set engine.initialized=False,
        # for pass_ in self.passes:
        #     pass_.delete()
        for program in self.programs.values():
            program.delete()
        for pooledmeshdata in self.pooledmeshdata.values():
            pooledmeshdata.delete()
        for mesh in self.meshes.values():
            mesh.delete()
        for vao in self.vaos.values():
            vao.delete()
        for material in self.materials.values():
            material.delete()
        for texture in self.texturedata.values():
            texture.delete()
        for texture_array in self.texture_arrays.values():
            texture_array.delete()
        # #for sampler in self.samplers.values():
        # #   del sampler
        for cubemap in self.cubemaps.values():
            cubemap.delete()
        # for uniform_block in self.uniform_blocks.values():
        #     # these are just integers...
        #     uniform_block.delete()
        for render_layer in self.render_layers.values():
            render_layer.delete()
        # for post_processor in self.post_processors:
        #     post_processor.delete()

        # passes and post processors arn't really resources...
        # self.passes = []
        self.programs.clear()
        # self.meshdata.clear()
        self.multimeshdata.clear()
        self.pooledmeshdata.clear()
        self.meshes.clear()
        self.vaos.clear()
        self.materials.clear()
        self.texturedata.clear()
        self.texture_arrays.clear()
        self.cubemaps.clear()
        # self.uniform_blocks.clear()
        self.render_layers.clear()
        # self.post_processors.clear()

class UniformBlock:
    name = 'default'
    index = 0
    size = 0

    def __init__(self):
        self.buf = 0

        self.buf = gfx.create_uniform_buffer(self.index, self.size)
        R.uniform_blocks[self.name] = self
        log.basic.debug('created ' + str(self))

    def __str__(self):
        return 'UniformBlock "{}" index {}'.format(self.name, self.index)

    def update(self):
        raise NotImplementedError()

class LightData:
    def __init__(self, name='default', type_=1, attenuation=0.1, ambient=0.0, diffuse=(0.1, 0.1, 0.1),
                 direction=(0.0, 1.0, 0.0)):
        self.name = name
        self.type_ = type_
        self.attenuation = attenuation
        self.ambient = ambient
        self.diffuse = diffuse
        self.direction = direction

        R.lightdata[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'LightData "{}"'.format(self.name)

class CameraData:
    def __init__(self, name='default', aspect=4.0 / 3.0, far=2000, near=0.5, fov=math.radians(60)):
        self.name = name
        self.aspect = aspect
        self.far = far
        self.near = near
        self.fov = fov

        R.cameradata[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'CameraData "{}"'.format(self.name)

class MeshProperties:
    # currently just lists the names of the materials it uses in order
    def __init__(self, name='default', materials=()):
        self.name = name
        self.materials = materials  # should be a list of names

        # init
        R.meshproperties[self.name] = self

class TextureSlot:
    # these are the properties of a texture specific to material
    def __init__(self, name='default', blend_fact=1.0, blend_mode='mix',
                 transform=(0.0, 0.0, 1.0, 1.0)):
        self.name = name
        self.blend_fact = blend_fact
        self.blend_mode = blend_mode
        self.transform = transform

# not 100% certain we need MaterialPass anymore, might just be one pass per material. but I'll leave for now
class Material(Sorted):
    def __init__(self, uniforms=None, pass_='opaque', program='default', switches=DEFAULT_SWITCHES, name='default',
                 texture_slots=(), cubemap=None, mask=MASK_DEFAULT, back=False, double_sided=False,
                 style=gfx.GL_TRIANGLES):
        super().__init__(Material)
        if uniforms is None:
            uniforms = {}
        self.name = name
        self.pass_ = None
        self.program = None
        self.texture_slots = texture_slots
        self.cubemap = R.get_resource(R.cubemaps, cubemap)  # only supports one for now
        self.uniforms = {}  # material uniforms are a subset of the program uniforms
        #   uniforms are classes which know how to update themselves
        self.mask = mask
        self.front = not back or double_sided
        self.back = back or double_sided
        self.style = style

        # init
        self.pass_ = R.get_resource(R.passes, pass_)
        self.program = Program.get_program(program, switches)
        self.default_uniform_values(uniforms)
        self.add_uniforms(uniforms)

        R.materials[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'Material "{}"'.format(self.name)

    def default_uniform_values(self, uniform_values):
        # programs can set default uniform values
        for d_uniform, d_args in self.program.programdata.defaults.items():
            if d_uniform not in uniform_values:
                # use default
                uniform_values[d_uniform] = d_args

    def add_uniforms(self, uniform_values):
        for uniform, value in uniform_values.items():
            self.create_uniform(uniform, value)

    def init_textures(self):
        for i, slot in enumerate(self.texture_slots):
            texture = R.arraytextures[slot.name]
            ident = 'tex' + str(i)
            self.create_uniform(ident + '.transform', slot.transform)
            self.create_uniform(ident + '.index', texture.index)
            self.create_texture_uniform(ident + '.sampler', texture.t_array_id, texture.sampler.sampler)
        if self.cubemap is not None:
            self.create_cubemap_uniform('cube', self.cubemap)  # always 'cube'

    def create_texture_uniform(self, key, t_array_id, sampler):
        # always texture array...
        info = self.get_uniform_info(key)
        if info is not None:
            self.uniforms[key] = gfx.Texture2DArrayUniform(info.index, info.t_pos, t_array_id, sampler)

    def create_cubemap_uniform(self, key, cubemap):
        info = self.get_uniform_info(key)
        if info is not None:
            self.uniforms[key] = gfx.CubemapUniform(info.index, info.t_pos, cubemap)

    def create_uniform(self, key, value):
        # !! this only works for non-opaque uniforms (not samplers)
        info = self.get_uniform_info(key)
        if info is not None:
            cls = gfx.get_basic_uniform_type(info.type)
            self.uniforms[key] = cls(info.index, value)

    def get_uniform_info(self, key):
        key = bytes(key, 'utf-8')
        if key not in self.program.uniforms:
            # log.basic.warning('uniform value "{}" ignored'.format(key))  # this was annoying
            return
        return self.program.uniforms[key]

    @staticmethod
    def split_identifier(full_identifier):
        split = full_identifier.split(b'.')
        return split[0], split[1]

    @staticmethod
    def get_texture_name(u_key, uniforms):
        # user should have supplied the texture name in uniforms
        t_name = None
        for unif, args in uniforms.items():
            if unif == u_key:
                t_name = args[0]
                break
        return t_name

    def update(self):
        # i don't particularly like this
        for u in self.uniforms.values():
            u.update()

    def delete(self):
        log.basic.info('deleted ' + str(self))

    def on_set_job(self, queue):
        queue.append(Job(self.update, "          change material {}".format(self.name)))

    @staticmethod
    def create(name='default', program='default', uniforms=None, alpha_blend='opaque', ambient=1.0, diffuse_color=None,
               double_sided=False, back=False, mask=MASK_DEFAULT, style='GL_TRIANGLES', emit=0.0, opacity=1.0,
               texture_slots=(), cubemap=None, translucency=0.0, unlit=False, use_object_color=False,
               use_transparency=False, use_vertex_colors=False, has_bones=False, use_fog=True, billboard=False):
        # create from raw properties
        if uniforms is None:
            uniforms = {}
        pass_ = alpha_blend  # hmm
        style = getattr(gfx, style)
        texture_slots = [TextureSlot(**texture_slot) for texture_slot in texture_slots if texture_slot is not None]
        switches = []
        if unlit:
            switches.append(UNLIT)
        if use_vertex_colors:
            switches.append(USE_VERTEX_COLORS)
        if use_object_color:
            switches.append(USE_OBJECT_COLOR)
        if len(texture_slots) > 0:
            switches.append(USE_TEXTURES)
        if use_fog:
            switches.append(USE_FOG)
        if has_bones:
            switches.append(USE_BONES)  # todo this shouldn't be a materials thing...
        if cubemap is not None:
            switches.append(USE_ENV_MAP)
        # todo billboard

        mat = Material(uniforms=uniforms, pass_=pass_, program=program, name=name, switches=switches,
                       texture_slots=texture_slots, cubemap=cubemap, double_sided=double_sided, back=back, mask=mask,
                       style=style)

        if len(texture_slots) > 1 and texture_slots[1] is not None:
            mat.create_uniform("t2_mix", texture_slots[1].blend_fact)  # hmm
        if diffuse_color is not None:
            # assume diffuse color is vec3, and has opacity
            mat.create_uniform("diffuse_color", [*diffuse_color, opacity])

        return mat

def process_line(line):
    if line.startswith('#include'):
        split = line.split(' ')
        include_file = split[1]
        include_text = read_file(include_file)
        return include_text  # (replaces line)
    return line

def shader_preprocessor(switches, source):
    # process each line looking for custom preprocessor directives (#include)
    lst = list(map(process_line, source.splitlines()))
    # add #defines for multicompile
    defines = ["#define {}".format(define) for define in switches]
    lst = defines + lst

    source = '\n'.join(lst)
    return source

class ProgramData:
    def __init__(self, vs_path, fs_path, defaults=(), name='default'):
        self.name = name
        self.vs_path = vs_path
        self.fs_path = fs_path
        self.defaults = defaults  # default uniform values
        R.programdata[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'ProgramData "{}"'.format(self.name)

class Program(Sorted):
    def __init__(self, programdata, switches):
        super().__init__(Program)
        # stores shader pair
        self.name = None
        self.programdata = programdata
        self.switches = switches
        self.program = None
        self.attributes = {}
        self.uniforms = {}
        self.uniform_blocks = {}

        # init
        self.name = self.instance_name(self.programdata, self.switches)
        log.basic.debug('creating Program "{}"...'.format(self.name))
        vs_source = self.get_source(self.programdata.vs_path)
        fs_source = self.get_source(self.programdata.fs_path)
        try:
            self.program = gfx.compile_program(vs_source, fs_source)
        except RuntimeError as e:
            log.basic.error('compilation error for program "{}"'.format(self.name))
            raise e
        self.attributes = gfx.get_program_attributes(self.program)
        self.uniforms = gfx.get_program_uniforms(self.program)
        self.uniform_blocks = gfx.get_program_uniform_blocks(self.program)
        for k, index in self.uniform_blocks.items():
            block = R.uniform_blocks[k]
            gfx.bind_program_uniform_block(self.program, index, block.index)
            log.basic.debug('bound uniform block "{}" from {} to {}'.format(k, index, block))

        R.programs[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'Program "{}"'.format(self.name)

    def delete(self):
        gfx.delete_program(self.program)
        log.basic.info('deleted ' + str(self))

    @staticmethod
    def get_program(programdata, switches):
        pdata = R.get_resource(R.programdata, programdata)
        instance_name = Program.instance_name(pdata, switches)
        if instance_name in R.programs:
            return R.programs[instance_name]
        # else create
        return Program(pdata, switches)

    @staticmethod
    def instance_name(programdata, switches):
        return programdata.name + str(Program.encode_switches(switches))

    @staticmethod
    def encode_switches(switches):
        id = 0
        for switch in switches:
            id |= SWITCH_MAP[switch]
        return id

    def get_source(self, path):
        source = read_file(path)
        source = shader_preprocessor(self.switches, source)
        if DEBUG_SHADERS:
            write_file(path + '.debug', source)
        return source

    def update(self):
        gfx.bind_program(self.program)

    def on_set_job(self, queue):
        queue.append(Job(self.update, "      change program {}".format(self.name)))

class Pass(Sorted):
    name = 'default'

    def __init__(self):
        super().__init__(Pass)

        # init
        R.passes[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'Pass "{}"'.format(self.name)

    def update(self):
        pass

    # all passes should define reset_state()?

    def delete(self):
        log.basic.info('deleted Pass "{}"'.format(self.name))

    def on_set_job(self, queue):
        queue.append(Job(self.update, "    change pass {}".format(self.name)))

class AnimationData:
    def __init__(self, name, duration, keys, tps):
        self.name = name
        self.duration = duration
        self.keys = keys
        self.tps = tps

        R.animationdata[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'AnimationData "{}"'.format(self.name)

    @staticmethod
    def create(animation):
        # creates multiple animations from assimp animation
        duration = int(animation.duration)  # can duration not be an int?
        tps = animation.tickspersecond
        for channel in animation.channels:
            name = channel.nodename.data.decode('utf-8')
            keys = []
            for i in range(len(channel.positionkeys)):
                time = channel.positionkeys[i].time  # all keys must have some time right? lol
                pos = channel.positionkeys[i].value
                hmm = channel.rotationkeys[i].value
                hmm[0] = -hmm[0]  # OK for some reason I have to reverse the x rotation...
                rot = Quaternion(hmm)
                # rot = Quaternion(channel.rotationkeys[i].value)
                scale = channel.scalingkeys[i].value
                keys.append(AnimationKey(time, pos, rot, scale))
            AnimationData(name, duration, keys, tps)

class AnimationKey:
    def __init__(self, time, position, rotation, scale):
        self.time = time
        self.position = position
        self.rotation = rotation
        self.scale = scale

class Node:
    def __init__(self, name, transformation, meshes, parent):
        self.name = name
        self.transformation = transformation
        self.meshes = meshes  # uhh, this is still the assimp type lol
        self.parent = parent
        self.children = []

    @staticmethod
    def create_r(parent, node):
        new = Node(node.name, node.transformation, node.meshes, parent)
        children = []
        for child in node.children:
            children.append(Node.create_r(new, child))
        new.children = children
        return new  # return topmost

class MeshData:
    def __init__(self, name, default_material, position_data, face_data, normal_data=None, color_data=None,
                 uv_data=None, bone_index_data=None, bone_weight_data=None):
        self.name = name
        self.vbo_size = 0
        self.attributes = {}  # vb attributes
        self.faces = None
        self.element_count = 0
        self.default_material = default_material

        # these shouldn't be here maybe?
        self.bones = None
        self.rootbone = None

        # init
        self.attributes[POSITION_ATTRIBUTE] = position_data
        self.faces = face_data
        if normal_data:
            self.attributes[NORMAL_ATTRIBUTE] = normal_data
        if color_data:
            self.attributes[COLOR_ATTRIBUTE] = color_data
        if uv_data:
            self.attributes[UV_ATTRIBUTE] = uv_data

        if bone_index_data:
            self.attributes[BONE_INDEX_ATTRIBUTE] = bone_index_data
            self.attributes[BONE_WEIGHT_ATTRIBUTE] = bone_weight_data

        self.vbo_size = sum([attr.size for attr in self.attributes.values()])
        # use first attribute to count number of elements (since all attributes should be the same)
        self.element_count = int(self.attributes[POSITION_ATTRIBUTE].data_len / POSITION_STRIDE)

        # R.meshdata[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'MeshData "{}"'.format(self.name)

    def draw(self, style=gfx.GL_TRIANGLES):
        gfx.draw_mesh(style, self.faces.data_len)

    def draw_instanced(self, instance_count, style=gfx.GL_TRIANGLES):
        gfx.draw_instanced(style, self.faces.data_len, instance_count)

    def delete(self):
        log.basic.debug('deleted ' + str(self))

class PooledMeshData(MeshData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mesh = None
        self.faces_offset = 0  # in bytes
        self.base_vertex = 0

        R.pooledmeshdata[self.name] = self

    def __str__(self):
        return 'PooledMeshData "{}"'.format(self.name)

    def set_mesh(self, mesh, faces_offset, base_vertex):
        self.mesh = mesh
        self.faces_offset = faces_offset
        self.base_vertex = base_vertex

    def draw(self, style=gfx.GL_TRIANGLES):
        gfx.draw_mesh_base_vertex(style, self.faces.data_len,
                                  self.faces_offset, self.base_vertex)

    def draw_instanced(self, instance_count, style=gfx.GL_TRIANGLES):
        gfx.draw_instanced_base_vertex(style,
                                       self.faces.data_len,
                                       self.faces_offset,
                                       instance_count,
                                       self.base_vertex)

# todo DynamicMesh, which is a ring buffer set to GL_STREAM_DRAW


class MeshGroup(Group):
    def __init__(self, attributes):
        super().__init__()
        self.meshes = []
        self.attributes = attributes

    def item_match(self, mesh):
        return collections.Counter(self.attributes) == collections.Counter(mesh.attributes.keys())

    @staticmethod
    def create(mesh):
        return MeshGroup(mesh.attributes.keys())

class Mesh:
    # can have multiple meshes concatenated into one vbo and ibo, for optimization
    def __init__(self, meshes, attributes, name='default'):
        self.name = name
        self.meshes = meshes
        self.attributes = attributes  # enabled mesh attributes
        self.attribute_start = {}  # start position of each attribute in vbo
        self.vbo = None
        self.ibo = None

        # init
        self.generate_name()
        self.init_buffers()

        R.meshes[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'Mesh "{}" with {} meshes'.format(self.name, len(self.meshes))

    def delete(self):
        gfx.delete_buffer(self.vbo)
        gfx.delete_buffer(self.ibo)
        log.basic.debug('deleted ' + str(self))

    def generate_name(self):
        # append attributes to name
        enc_attr = ""
        for attr in self.attributes:
            enc_attr += str(attr)[2]
        self.name = self.name + "|" + enc_attr

    def init_buffers(self):
        # initialize vbo
        vbo_size = 0
        for pooledmeshdata in self.meshes:
            vbo_size += pooledmeshdata.vbo_size

        with gfx.CreateVbo(vbo_size) as vbo:
            self.vbo = vbo.index
            # add data to vbo (VVVNNNCCCUUU)
            for attr_key in BUILTIN_ATTRIBUTES.keys():
                if attr_key not in self.attributes: continue
                self.attribute_start[attr_key] = vbo.offset
                for pooledmeshdata in self.meshes:
                    attr = pooledmeshdata.attributes[attr_key]
                    vbo.subdata(attr.size, attr.data)

        ibo_size = 0
        for pooledmeshdata in self.meshes:
            ibo_size += pooledmeshdata.faces.size

        with gfx.CreateIbo(ibo_size) as ibo:
            self.ibo = ibo.index
            base_vertex = 0
            for pooledmeshdata in self.meshes:
                pooledmeshdata.set_mesh(self, ibo.offset, base_vertex)
                ibo.subdata(pooledmeshdata.faces.size, pooledmeshdata.faces.data)
                base_vertex += pooledmeshdata.element_count

    @staticmethod
    def create_from_pooledmeshdata(meshes, name='default'):
        for group in Grouper.group_items(MeshGroup, meshes):
            Mesh(group.items, group.attributes, name=name)

class Vao(Sorted):
    # in libgdx this is a ModelInstance?
    def __init__(self, mesh, program):
        super().__init__(Vao)
        self.name = None
        self.mesh = mesh
        self.program = program
        self.vao = None

        # init
        self.name = self.construct_name(self.mesh, self.program)
        self.init_vao()
        R.vaos[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'Vao "{}"'.format(self.name)

    def delete(self):
        gfx.delete_vertex_array(self.vao)
        log.basic.info('deleted ' + str(self))

    @staticmethod
    def construct_name(mesh, program):
        return mesh.name + '.' + program.name

    def init_vao(self):
        with gfx.CreateVao() as vao:
            self.vao = vao.index
            vao.bind_vbo(self.mesh.vbo)
            self.enable_attributes(vao)
            vao.bind_ibo(self.mesh.ibo)

    def enable_attributes(self, vao):
        for attr_key, stride in BUILTIN_ATTRIBUTES.items():
            if attr_key not in self.program.attributes or \
               attr_key not in self.mesh.attributes:
                continue
            offset = self.mesh.attribute_start[attr_key]
            i = self.program.attributes[attr_key]
            vao.vtx_attr(i, stride, offset)

    def update(self):
        gfx.bind_vao(self.vao)

    @staticmethod
    def _find_vao(mesh, program):
        name = Vao.construct_name(mesh, program)
        if name in R.vaos:
            return R.vaos[name]

    @staticmethod
    def get_vao(mesh, program):
        vao = Vao._find_vao(mesh, program)
        if not vao:
            # create
            vao = Vao(mesh, program)
        return vao

    def on_set_job(self, queue):
        queue.append(Job(self.update, "        change vao {}".format(self.name)))

class Sampler:
    # we don't really need this as a class, but might as well be consistent
    def __init__(self, filter=LINEAR, wrapx=REPEAT, wrapy=REPEAT):
        self.name = None
        self.filter = filter
        self.wrapx = wrapx
        self.wrapy = wrapy
        self.sampler = None

        # init
        self.name = self.encode_value(self.filter, self.wrapx, self.wrapy)
        self.sampler = gfx.create_sampler(self.filter, self.wrapx, self.wrapy)
        R.samplers[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'Sampler "{}"'.format(self.name)

    @staticmethod
    def encode_value(filter, wrapx, wrapy):
        filter = FILTER_BINARY_MAP[filter]
        wrapx = WRAP_BINARY_MAP[wrapx]
        wrapy = WRAP_BINARY_MAP[wrapy]
        # lol, there are only 32 possible samplers you can make, might as well make them all...
        return filter << 4 | wrapx << 2 | wrapy

    @staticmethod
    def get_sampler(filtering, mir_x, mir_y, sampling):
        wrapx = MIRROR if mir_x else sampling
        wrapy = MIRROR if mir_y else sampling
        sampler_name = Sampler.encode_value(filtering, wrapx, wrapy)
        # if sampler doesn't exist, create
        if sampler_name not in R.samplers:
            # create
            return Sampler(filtering, wrapx, wrapy)
        return R.samplers[sampler_name]

class TextureData:
    # this also stores texture data (if you wanted to write to texture with CPU, maybe use something like this?)
    # todo tex needs to factor into sort order
    def __init__(self, name, path, alpha=False):
        self.name = name
        self.width = None
        self.height = None
        self.alpha = alpha
        self.data = None

        # init
        self.load_image(path)
        self.register_resource()
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'TextureData "{}"'.format(self.name)

    def register_resource(self):
        R.texturedata[self.name] = self

    def load_image(self, path):
        width, height, data = gfx.load_texture(path, self.alpha)
        self.width = width
        self.height = height
        self.data = data

    def delete(self):
        log.basic.info('deleted ' + str(self))

class ArrayTexture(TextureData):
    def __init__(self, name, path, alpha=False, sampler=DEFAULT_SAMPLER):
        super().__init__(name, path, alpha)
        self.t_array_id = None
        self.index = None
        self.sampler = R.get_resource(R.samplers, sampler)

    def __str__(self):
        return 'ArrayTexture "{}"'.format(self.name)

    def register_resource(self):
        R.arraytextures[self.name] = self

    def set_array(self, t_array_id, index):
        self.t_array_id = t_array_id
        self.index = index  # texture array index

    def delete(self):
        log.basic.info('deleted ' + str(self))

    @staticmethod
    def create(name='default', filepath=None, mir_x=False, mir_y=False,
               sampling=REPEAT, filtering=LINEAR, use_alpha=False):
        # todo could be split into Texture and Image?...
        filepath = abspath(filepath)  # accepts relative filepaths
        sampler = Sampler.get_sampler(filtering, mir_x, mir_y, sampling)
        return ArrayTexture(name=name, path=filepath, alpha=use_alpha, sampler=sampler)

class TextureGroup(Group):
    def __init__(self, width, height, alpha):
        super().__init__()
        self.width = width
        self.height = height
        self.alpha = alpha

    def item_match(self, texture):
        return texture.width == self.width and \
               texture.height == self.height and \
               texture.alpha == self.alpha

    @staticmethod
    def create(item):
        return TextureGroup(item.width, item.height, item.alpha)

class TextureArray:
    def __init__(self, width, height, alpha, textures, name):
        self.name = name
        self.tarray = None
        self.width = width
        self.height = height
        self.alpha = alpha
        self.textures = textures

        self.init_texture_array()
        R.texture_arrays[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'TextureArray {}x{}x{} {}'.format(self.width, self.height, len(self.textures),
                                                 "alpha" if self.alpha else "")

    def delete(self):
        gfx.delete_texture(self.tarray)
        log.basic.info('deleted ' + str(self))

    def init_texture_array(self):
        count = len(self.textures)
        with gfx.CreateTextureArray(self.width, self.height, count, self.alpha) as tarray:
            self.tarray = tarray.index
            t_index = 0
            for tex_info in self.textures:
                tex_info.set_array(self.tarray, t_index)
                tarray.subdata(t_index, tex_info.data)
                del tex_info.data  # don't need this anymore (unless readable texture?)
                t_index += 1

    @staticmethod
    def create_from_textures(textures, name='default'):
        for group in Grouper.group_items(TextureGroup, textures):
            TextureArray(group.width, group.height, group.alpha, group.items, name=name)

class Cubemap:
    def __init__(self, faces, name='default'):
        self.name = name
        self.id = None

        # init
        self.id = gfx.create_cubemap(faces)

        R.cubemaps[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'Cubemap "{}"'.format(self.name)

    def delete(self):
        gfx.delete_buffer(self.id)
        log.basic.info('deleted ' + str(self))

class RenderLayer(Sorted):
    def __init__(self, clear=True, clear_color=WHITE, target=None, mask=MASK_DEFAULT,
                 depth=0, name='default'):
        super().__init__(RenderLayer)
        self.name = name
        self.mask = mask
        self.cbuf = None
        self.dbuf = None
        self.fbuf = None
        self.width = None
        self.height = None
        self.render_texture = None  # always has a render texture?
        self.processors = []  # post processing stages
        self.depth = depth
        self.camera = None  # what if you wanted to render with multiple cameras but the same mask? can you do that?
        self.clear = clear
        self.clear_color = clear_color
        self.initialized = False

        R.render_layers[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'RenderLayer "{}"'.format(self.name)

    def delete(self):
        if self.initialized:
            gfx.delete_renderbuffer(self.cbuf)
            gfx.delete_renderbuffer(self.dbuf)
            gfx.delete_framebuffer(self.fbuf)
            gfx.delete_texture(self.render_texture)
            log.basic.info('deleted ' + str(self))

    def set_camera(self, camera, width, height):
        if self.initialized:
            log.basic.warning("don't add render layer to multiple cameras")
            return
        # if isinstance(camera, str):
        #    camera = find_with_name(R.entities, camera).get_component(entities.CameraComponent)
        self.camera = camera
        self.width = width
        self.height = height
        self.init_framebuffer()
        self.initialized = True

    def init_framebuffer(self):
        # todo disable depth buffer? and this should probably be all one gfx function
        self.cbuf = gfx.create_color_renderbuffer(self.width, self.height)
        self.dbuf = gfx.create_depth_renderbuffer(self.width, self.height)
        self.fbuf = gfx.create_framebuffer(self.cbuf, self.dbuf)
        # todo only works because we didn't unbind framebuffer lol
        self.render_texture = gfx.create_render_texture(self.width, self.height, True)

    def update(self):
        gfx.bind_framebuffer(self.fbuf)
        gfx.clear_framebuffer(self.width, self.height, self.clear, self.clear_color)

    def add_postprocessor(self, processor):
        processor = R.get_resource(R.post_processors, processor)
        self.processors.append(processor)

    def post_processing(self):
        gfx.set_overwrite_blending()  # always want? I guess this is why post processing should be a Pass
        #                                 wouldn't it be easier to just have process.update though?
        for process in self.processors:
            # huh, I'd have thought it wouldn't let you draw from and to the same render texture. nevermind
            process.draw(self.render_texture, self.fbuf)

    def on_set_job(self, queue):
        queue.append(Job(self.update, "  change render layer {}".format(self.name)))

    def on_unset_job(self, queue):
        queue.append(Job(self.post_processing, "  do post processing {}".format(self.name)))
        pass

class PostProcessor:
    name = 'passthrough'
    program_name = 'passthrough'
    quad = None

    def __init__(self):
        self.vao = None
        self.tex_uniform = None
        self.program = None

        # init
        R.post_processors[self.name] = self
        log.basic.info('created ' + str(self))

    def __str__(self):
        return 'PostProcessor "{}"'.format(self.name)

    def init(self):
        # todo should i use a material and stuff and do it the normal way instead of hacking it?
        if not PostProcessor.quad:
            PostProcessor.quad = R.pooledmeshdata[R.multimeshdata['quad'][0]]
        self.program = Program.get_program(self.program_name, DEFAULT_SWITCHES)
        self.vao = Vao.get_vao(PostProcessor.quad.mesh, self.program)
        info = self.program.uniforms[b"tex"]
        self.tex_uniform = gfx.Texture2DUniform(info.index, 0, None, R.samplers[0].sampler)

    def _bind(self, source, dest):
        gfx.bind_framebuffer(dest)
        gfx.bind_program(self.program.program)
        gfx.bind_vao(self.vao.vao)
        self.tex_uniform.texture_id = source  # source is a texture id, dest is a framebuffer id...
        self.tex_uniform.update()

    def draw(self, source, dest):
        self._bind(source, dest)
        PostProcessor.quad.draw()

    def delete(self):
        log.basic.info('deleted ' + str(self))

class Camera(Sorted):
    # this is only abstract, still have to implement yourself
    # (it's weird that I'm sorting by camera anyway...)
    def __init__(self):
        super().__init__(Camera)
        self.name = 'no name'  # this could be a problem

        # note: not adding to .cameras ={} because don't care
        log.basic.info('created Camera "{}"'.format(self.name))

    def __str__(self):
        return 'Camera "{}"'.format(self.name)

    def update_state(self):
        raise NotImplementedError()

    def on_set_job(self, queue):
        queue.append(Job(self.update_state, "change camera {}".format(self.name)))
