
import code
import logging
import threading
from pprint import pformat
from pprint import pprint

import pygame  # would be nice to abstract this dependency
from pygame.locals import *

from dreamengine.utils import *
from dreamengine.constants import *
from dreamengine.mathutils import *
import dreamengine.log as log
import dreamengine.basegraphics as gfx
from dreamengine.rendering import Pass, Vao, DrawJob, Resources, RenderLayer, UniformBlock, Camera, PostProcessor, Renderer, Object3D

E = None

def create_armature(mesh):
    mesh = E.pooledmeshdata[E.multimeshdata[mesh][0]]
    inverse_global = inverse(mesh.rootbone.transformation)
    bones = []  # can be sure will be added in indexed order?... (should map name to index)
    Entity3D.create_from_node_r(mesh.rootbone, bones, bones=mesh.bones, inverse_global=inverse_global)
    return bones

def interactive_console(vars):
    # (if I wanted an in-game console I'd use RestrictedPython)
    shell = code.InteractiveConsole(vars)
    shell.interact()

# todo BuiltinUniform(index, entity, func)?
class MVPUniform(gfx.Mat4Uniform):
    def __init__(self, index, entity):
        super().__init__(index, None)
        self.entity = entity

    def update(self):
        self.mat = self.entity.mvp.flatten()
        super().update()

class MVUniform(gfx.Mat4Uniform):
    def __init__(self, index, entity):
        super().__init__(index, None)
        self.entity = entity

    def update(self):
        self.mat = self.entity.mv.flatten()
        super().update()

class MUniform(gfx.Mat4Uniform):
    def __init__(self, index, entity):
        super().__init__(index, None)
        self.entity = entity

    def update(self):
        self.mat = self.entity.model.flatten()
        super().update()

class MPUniform(gfx.Mat4Uniform):
    def __init__(self, index, entity):
        super().__init__(index, None)
        self.entity = entity

    def update(self):
        self.mat = self.entity.mp.flatten()
        super().update()

class NormalMatrixUniform(gfx.Mat3Uniform):
    def __init__(self, index, entity):
        super().__init__(index, None)
        self.entity = entity

    def update(self):
        model = self.entity.model
        vm = mat4_2_mat3(model)
        vm = inverse(vm)
        vm = transpose(vm)
        self.mat = vm.flatten()
        super().update()

class BonesUniform(gfx.Mat4Uniform):
    def __init__(self, index, entity):
        super().__init__(index, None)
        self.entity = entity
        self.initialized = False
        self.bones = None

        # init
        armature = self.entity.get_component(ArmatureComponent)
        self.set_bones(armature.bones)
        # bones = rend.MeshData.bones

    def set_bones(self, bones):
        self.bones = bones  # a list of _entities_ that are arranged in a hierarchy and animated
        self.count = len(self.bones)
        self.initialized = True

    def update(self):
        if not self.initialized: return
        flattened = []
        for bone in self.bones:
            model = bone.inverse_global @ bone.model @ bone.offset
            flattened.append(model.flatten())
        # note: can save register space by uploading only quaternion and offset
        self.mat = np.concatenate(flattened)
        super().update()

class OpaquePass(Pass):
    name = 'opaque'

    def update(self):
        gfx.update_default_pass()

# does additive pass always need to come after opaque passes? probably...
class AdditivePass(Pass):
    name = 'additive'

    def update(self):
        gfx.update_additive_pass()

def distance_from_player(player_location, entity):
    t, r, s = entity.world_transform()
    return vec3_length(vec3_subtract(player_location, t))

class LightingBlock(UniformBlock):
    # not just lighting anymore...
    name = b"lighting"
    index = 0
    # everything should have stride 16 if 'shared' (?)
    size = 16 + SIZEOF_FOG + SIZEOF_DIRECTIONAL_LIGHT + (MAX_POINT_LIGHTS * SIZEOF_POINT_LIGHT)

    def update(self):
        # glGetIntegerv() tells us about GPU buffer block limitations
        # GL_MAX_UNIFORM_BUFFER_BINDINGS GL_MAX_UNIFORM_BLOCK_SIZE GL_MAX_VERTEX_UNIFORM_BLOCKS GL_MAX_FRAGMENT_UNIFORM_BLOCKS GL_MAX_GEOMETRY_UNIFORM_BLOCKS
        world = E.world
        dlight = E.dlight
        dlight_col = dlight.color
        dlight_pos = dlight.entity.position  # todo use normalised entity angle-axis rotation as direction
        dlight_ambient = dlight.ambient

        # I don't really need to do this because I'm not doing specular
        # view = E.maincamera.matrix
        # dlight_pos = np.resize(dlight_pos, (4,1))
        # lightPosition = view @ dlight_pos

        with gfx.UniformBlockUpdater(self.buf, self.index) as block:
            # wow, std140 is complicated
            #   vec3's are padded to vec4, unless you put a float after them
            block.subdata(4 * 4, E.t)
            block.subdata(3 * 4, world.ambient)
            block.subdata(1 * 4, world.fog_start)
            block.subdata(3 * 4, world.fog_color)
            block.subdata(1 * 4, world.fog_end)

            block.subdata(4 * 4, dlight_col)
            block.subdata(3 * 4, dlight_pos)
            block.subdata(1 * 4, dlight_ambient)

            # find four closest point lights
            player_location, r, s = E.player.world_transform()
            # hmm, it would probably be better to use camera position
            # but then would have to update lighting block multiple times per frame
            plights = list(E.plights)  # copy list
            plights.sort(key=lambda plight: distance_from_player(player_location, plight.entity))
            plights.sort(key=lambda plight: plight.priority)
            # todo should probably use coroutines to lerp lights on/off instead of suddenly changing
            #   or could be up to the user?
            # take the first four
            for plight in plights[:4]:
                t, s, r = plight.entity.world_transform()
                block.subdata(3 * 4, plight.color)
                block.subdata(1 * 4, plight.attenuation)
                block.subdata(3 * 4, t)
                block.subdata(1 * 4, plight.ambient)

class Entity:
    # you can use entity for singletons
    def __init__(self, name='default', components=()):
        self.name = name
        self.components = []

        # init
        for c in components:
            self.add_component(c)

        E.entities.append(self)
        log.basic.info('created ' + str(self))

    def __str__(self):
        return '{} "{}"'.format(self.__class__.__name__, self.name)
        # return 'Entity "{}" with {} components'.format(self.name, len(self.components))

    def delete(self):
        # should mark resources as not required anymore, so can unload?
        E.entities.remove(self)
        log.basic.info('deleted {} "{}"'.format(self.__class__.__name__, self.name))

    def add_component(self, comp):
        comp.entity = self
        self.components.append(comp)
        comp.init()

    def add_components(self, components):
        for component in components:
            self.add_component(component)

    def get_components(self, cls):
        l = []
        for c in self.components:
            if isinstance(c, cls):
                l.append(c)
        return l

    def get_component(self, cls):
        for c in self.components:
            if isinstance(c, cls):
                return c

    def update(self, dt):
        for c in self.components:
            c.update(dt)

class Entity3D(Entity, Object3D):
    def __init__(self, position=(0.0, 0.0, 0.0), rotation=(0, 0, 0), scale=(1.0, 1.0, 1.0),
                 parent=None, children=None, name='default', static=False, components=()):
        Object3D.__init__(self, position, rotation, scale, parent, children, name, static)
        Entity.__init__(self, name, components)

    def delete(self):
        Object3D.delete(self)
        Entity.delete(self)

    @staticmethod
    def create_from_node_r(node, entities, parent=None, bones=(), inverse_global=None, visualize=False, static=False):
        # create from assimp node
        # also creates animations, bones
        name = node.name
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        parent = parent if parent else E.root
        # trans = np.asmatrix(node.transformation)
        # t, s, r = decompose_matrix(trans)

        new = Entity3D(
            name=name,
            # position=t,
            # rotation=r,
            # scale=s,
            parent=parent,
            static=static)

        if visualize:
            # todo should be lines and shit
            #   (immediate mode)
            new.add_component(RenderComponent('small_octahedron', 'invert'))

        # todo would you really want to apply animations like this?
        if name in E.animationdata:
            anim = E.animationdata[name]
            new.add_component(AnimationComponent(anim, pingpong=True))

        if name in E.lightdata:
            light = E.lightdata[name]
            if light.type_ == 1:
                # directional
                new.add_component(DirectionalLightComponent(color=light.diffuse, ambient=light.ambient))
                new.position = light.direction
                # todo direction properly
                pass
            if light.type_ == 2:
                # point
                new.add_component(
                    PointLightComponent(color=light.diffuse, ambient=light.ambient, attenuation=light.attenuation))

        if name in E.cameradata:
            camera = E.cameradata[name]
            # hmm, adding default render layer?
            new.add_component(CameraComponent(fov=camera.fov, near=camera.near, far=camera.far, rlayers=['default']))

        for mesh in node.meshes:
            new.add_component(RenderComponent(E.unique_mesh_name(mesh)))
        # if len(node.meshes) > 0:
        #     # use name of first, RenderComponent knows how to get the other meshes
        #     mesh = node.meshes[0]
        #     new.add_components(RenderComponent.create(mesh.name))

        # todo this should be on BoneComponent or something?
        bone = find_with_name(bones, name)
        if bone:
            new.set_bone_mats(bone.offsetmatrix, inverse_global)

        entities.append(new)
        for child in node.children:
            Entity3D.create_from_node_r(child, entities, new, bones, inverse_global, visualize, static)
        return new  # return topmost entity

class Component:
    dt = 0.0  # adding this shortcut here

    def __init__(self):
        self.entity = None

    def init(self):
        pass

    def update(self, dt):
        pass

class CameraComponent(Component, Camera):
    # could probably abstract the maths
    def __init__(self, fov=math.radians(60), near=0.5, far=2000.0,
                 width=None, height=None,
                 ortho=False, orth_scale=1.0,
                 rlayers=()):
        Component.__init__(self)
        Camera.__init__(self)
        self.fov = fov
        self.near = near
        self.far = far
        self.ortho = ortho
        self.orth_scale = orth_scale
        self.fixed_size = True
        # a camera has dimensions!
        self.width = width or NATIVE_RESOLUTION[0]
        self.height = height or NATIVE_RESOLUTION[1]
        self.view = None
        self.projection = None  # the projection matrix

        # init
        # todo hmm we want width and height to be equal to screen size probably
        if not any(rlayers):
            RenderLayer().set_camera(self, self.width, self.height)  # should always have one i guess
        else:
            for rlayer in rlayers:
                rlayer = E.get_resource(E.render_layers, rlayer)
                rlayer.set_camera(self, self.width, self.height)

    def init(self):
        if not E.maincamera:
            E.maincamera = self
        self.name = self.entity.name

    def update_projection(self, width=None, height=None):
        self.view = inverse(self.entity.model)
        if not self.fixed_size:
            # this won't really work because framebuffers need to be resized
            # probably easier to delete/reset all cameras and render layers
            self.width = width
            self.height = height
        if not self.ortho:
            self.projection = perspective(self.fov, self.width / self.height, self.near, self.far)
        else:
            w = (self.width / 2) * self.orth_scale
            h = (self.height / 2) * self.orth_scale
            self.projection = ortho(-w, w, -h, h, self.near, self.far)

    def update_state(self):
        self.update_projection()
        E.active_camera = self
        E.root.update_mvp_r(self)  # todo should pass mask so entity can choose not to update itself

class ArmatureComponent(Component):
    def __init__(self, bones):
        super().__init__()
        self.bones = bones

class RenderComponent(Component):
    # todo inherit MeshRenderer, Component?
    builtin_uniforms = {  # links keywords to class (all have constructor (index, entity) ...)
        MVP_UNIFORM: MVPUniform,
        MV_UNIFORM: MVUniform,
        M_UNIFORM: MUniform,
        MP_UNIFORM: MPUniform,
        NORMALMATRIX_UNIFORM: NormalMatrixUniform,
        BONES_UNIFORM: BonesUniform
    }

    # unity has MeshComponent, would that be useful? i think that would be worse
    def __init__(self, mesh='default', material=None, instanced=False, color=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        # todo ability to override per-object material properties from here?
        self.visible = True
        self.meshdata = None
        self.material = None
        self.vao = None
        self.instanced = instanced  # maybe this is a material thing, but it's stupid anyway
        self.uniforms = {}  # per-object uniforms
        self._color = None
        self.color_uniform = None

        # init
        self.color = color  # per-object color
        try:
            self.meshdata = E.get_resource(E.pooledmeshdata, mesh)
        except KeyError:
            # 'mesh' should be a pooledmeshdata key, but since that is a bit confusing accept multimesh key too
            multi = E.multimeshdata[mesh]
            self.meshdata = E.pooledmeshdata[multi[0]] # only use first though
        if material is None:
            # use mesh 'default material'
            self.material = E.materials[self.meshdata.default_material]
        else:
            self.material = E.get_resource(E.materials, material)

    def set_color(self, color):
        self._color = color
        if self.color_uniform is not None:
            self.color_uniform.vec = color

    def get_color(self):
        return self._color

    color = property(get_color, set_color)

    def init(self):
        # create a vao for the combination of mesh and program
        prog = self.material.program
        self.vao = Vao.get_vao(self.meshdata.mesh, prog)
        # init object color uniform
        info = self.material.get_uniform_info('object_color')
        if info is not None:
            self.color_uniform = gfx.Vec4Uniform(info.index, self.color)
            self.uniforms['object_color'] = self.color_uniform
        # init builtin uniforms
        for key, cls in self.builtin_uniforms.items():
            if key in prog.uniforms:
                self.uniforms[key] = cls(prog.uniforms[key].index, self.entity)

        #log.basic.info('created render component ' + self.entity.name)

    def queue_draw(self):
        # maybe extract this into renderer.draw()?
        mat = self.material
        pass_ = mat.pass_
        vao = self.vao

        for rlayer in E.render_layers.values():
            if not mat.mask == rlayer.mask: continue
            cam = rlayer.camera
            func = self.draw_instanced if self.instanced else self.draw_mesh
            # note: order is significant!
            # (in 'approaching zero driver overhead' they have:
            #   render target -> pass                         -> material -> material instance -> vertex format  -> object
            #  (framebuffer   -> depth, blending, states, ... -> shaders  -> textures          -> vertex buffers -> object)
            #   i should definately try sorting by texture with higher order than program, to see if that works
            #   if it does, that means the optimal thing to do would be to recompile the shaders to use all texture units...
            sorters = [mat,  # uniforms
                       vao,  # vao
                       vao.program,  # shader program
                       pass_,  # gl state
                       rlayer,  # framebuffer
                       cam]  # mvp
            # todo should probably be Renderer.draw() or something
            DrawJob(func, sorters, '            draw job {}'.format(self.entity.name))

    def update_state(self):
        for uniform in self.uniforms.values():
            uniform.update()

    # couldn't you draw everything with just instancing?
    #   put every uniform into arrays of uniforms
    #   only store unique uniform values, and create an index buffer
    #   index by instance ID
    def draw_instanced(self):
        # this implementation is good if you want to control the positions of the instances programatically from the shader
        # todo use blank Entity3D as instances, contatenate matrices into a buffer and send as uniform, access with gl_InstanceID
        # InstanceComponent? with uniforms[] parameter?
        instance_count = 16
        # any use for glVertexAttribDivisor()?
        self.draw(self.meshdata.draw_instanced, instance_count, self.material.style)

    def draw_mesh(self):
        self.draw(self.meshdata.draw, self.material.style)

    def draw(self, func, *args):
        # if not self.frustrum_test(): return  # this actually slows it down lol
        self.update_state()
        if self.material.back:
            # draw back first
            gfx.enable_back()
            # gfx.glCullFace(gfx.GL_FRONT)
            func(*args)
            gfx.enable_front()  # leave in state we found it
            # gfx.glCullFace(gfx.GL_BACK)
        if self.material.front:
            func(*args)

    @staticmethod
    def create(mesh, materials=None, material=None, instanced=False):
        # since meshes with multiple materials are actually multiple meshes, they require multiple render components
        # in that case this function will return multiple render components
        multi = E.multimeshdata[mesh]
        if materials is not None:
            return [RenderComponent(meshdata, materials[i], instanced) for i, meshdata in enumerate(multi)]
        else:
            # use 'material'
            return [RenderComponent(meshdata, material, instanced) for meshdata in multi]


class AnimationComponent(Component):
    def __init__(self, data, pingpong=False):
        super().__init__()
        self.data = E.get_resource(E.animationdata, data)
        self.pingpong = pingpong
        # self.loop = loop
        self.speed = 1.0  # test

    def update(self, dt):
        d = self.data.duration
        t = E.t * self.data.tps * self.speed
        f = t % d
        if self.pingpong and t / d % 2 >= 1:
            f = d - f
        k1, k2, j = self.get_keys(f)
        t = lerp_vec3(k1.position, k2.position, j)
        r = k1.rotation.slerp(k2.rotation, j)
        s = lerp_vec3(k1.scale, k2.scale, j)
        mats = [translate(t), r.to_matrix(), scale(s)]
        self.entity.anim_mat = matrix_multiply(mats)

    def get_keys(self, t):
        l = len(self.data.keys) - 1
        for i in range(l):
            k1 = self.data.keys[i]
            if k1.time > t:
                k2 = self.data.keys[i - 1]
                f = AnimationComponent.get_dist(k1.time, k2.time, t)
                return k1, k2, f
        # todo shouldn't get here?
        last = self.data.keys[l - 1]
        # log.basic.warning("animation key not found")
        return last, last, 0.0

    @staticmethod
    def get_dist(a, b, c):
        # get the proportional distance of c between a and b
        d = c - a
        try:
            return 1 / ((b - a) / d)
        except ZeroDivisionError:
            return 0

class WorldComponent(Component):
    # hmm, how can make easy for user to override/extend?
    def __init__(self, ambient=(0.6, 0.6, 0.6)):
        super().__init__()
        self.ambient = ambient
        # plan is to only support 1 directional light and 4 point lights (heh heh)

        self.fog_color = (0.3, 0.31, 0.31)
        self.fog_start = 10
        self.fog_end = 100

        self.lblock = LightingBlock()

        E.world = self

    def update(self, dt):
        # we only really need to do this if the value changes, but it's pretty trivial
        self.lblock.update()

def World(*args, **kwargs):
    return Entity(
        name='world',
        components=[WorldComponent(*args, **kwargs)]
    )

class DreamError(Exception):
    pass

class DirectionalLightComponent(Component):
    def __init__(self, color=(0.5, 0.5, 0.5), ambient=0.04):
        super().__init__()
        self.color = color
        self.ambient = ambient

        # init
        if E.dlight is not None:
            raise DreamError("multiple directional light components not supported")
        E.dlight = self

def DirectionalLight(direction=(0.0, 0.0, 0.0), *args, **kwargs):
    return Entity3D(
        name="directional light",
        position = direction,
        components=[DirectionalLightComponent(*args, **kwargs)]
    )

class PointLightComponent(Component):
    def __init__(self, color=(0.5, 0.5, 0.5), ambient=0.0, attenuation=0.1, priority=0):
        super().__init__()
        self.color = color
        self.ambient = ambient
        self.attenuation = attenuation  # rename 'radius'?
        self.priority = priority
        E.plights.append(self)

def PointLight(*args, **kwargs):
    return Entity3D(
        name='point light',
        components=[PointLightComponent(*args, **kwargs)]
    )

class Engine(Resources, Renderer):
    def __init__(self, logging_level=logging.WARNING):
        Resources.__init__(self)
        Renderer.__init__(self)
        global E
        E = self
        DrawJob.renderer = self

        log.init(logging_level)
        log.basic.info("starting")

        self.screen = None
        self.clock = None

        self.entities = []
        self.dlight = None
        self.plights = []
        self.world = None

        self.root = None
        self.active_camera = None
        self.maincamera = None

        self.frame = 0
        self.dt = 0  # delta time
        self.t = 0
        self.console_thread = None
        self.initialized = False

        # init
        self.resize(*SCREEN_SIZE)
        self.init_pygame()
        self.root = Entity3D(name="root")
        Object3D.root = self.root
        self.create_lights()
        World()
        self.sound_test()
        self.default_resources()
        self.init_rendering()
        self.load_resources(read_file("dreamengine/default_resources.json"))  # todo 'color cube' should just be cube

        # self.interactive_console()

    def init_pygame(self):
        pygame.init()
        # i guess using pygame limits me to not using a multisampling frame? i dunno
        # self.screen = pygame.display.set_mode((self.width, self.height), HWSURFACE | OPENGL | DOUBLEBUF | RESIZABLE)
        self.screen = pygame.display.set_mode((self.width, self.height), HWSURFACE | OPENGL | DOUBLEBUF)
        self.clock = pygame.time.Clock()

    def default_resources(self):
        OpaquePass()
        AdditivePass()

    def sound_test(self):
        # test sound
        # (could load 'songs' in json? yeah why not)
        # actually should load 'sounds', then load from SoundComponent (use for SFX too)
        pygame.mixer.init()
        sound = pygame.mixer.Sound("assets\\music.ogg")
        sound.play()
        sound.set_volume(0.2)

    def create_lights(self):
        # create four point lights that contribute no light to the scene
        # because we need minimum of four...
        for i in range(4):
            PointLight(color=(0.0, 0.0, 0.0), priority=1000)

    def interactive_console(self):
        # interactive console
        if log.basic.isEnabledFor(logging.DEBUG):
            vars = globals().copy()
            vars.update(locals())
            t = threading.Thread(target=interactive_console, args=(vars,))
            t.daemon = True
            t.start()
            self.console_thread = t

    def late_init(self):
        if not self.maincamera:
            log.basic.warning("maincamera not present in scene. creating default camera")
            Entity3D(components=[CameraComponent()])
        if not self.dlight:
            log.basic.warning('creating default directional light')
            DirectionalLight(direction=(0.2, 0.9, 0.5), color=(0.2, 0.2, 0.2))
        for name, rlayer in self.render_layers.items():
            if not rlayer.initialized:
                raise DreamError('render layer "{}" not initialized'.format(rlayer.name))
        self.init_post_processing()  # we have to do this after camera is created
        self.root.update_matrix_r(True)  # update static entity matrices once
        self.construct_queue()  # we only need to do this once because nothing is changing
        # (I can imagine this being another area of tedious optimization, caching bits that change...)
        self.initialized = True

    def mainloop(self):
        log.basic.info("entering mainloop")
        while True:
            if not self.initialized:
                self.late_init()
            self.t = pygame.time.get_ticks() / 1000
            self.handle_events()
            for entity in self.entities:
                entity.update(self.dt)
            self.root.update_matrix_r()
            self.draw()
            self.blit()
            pygame.display.flip()
            self.sleep()
            self.frame += 1

    def sleep(self):
        self.dt = self.clock.tick(FPS) / 1000
        Component.dt = self.dt
        #print(int(self.clock.get_fps()))

    def handle_events(self):
        pygame.event.pump()
        events = pygame.event.get()
        for e in events:
            if e.type == QUIT: self.quit()
            if e.type == VIDEORESIZE:
                size = e.dict['size']
                self.resize(*size)
            if e.type == KEYDOWN:
                if e.key == K_SPACE:
                    pass
                if e.key == K_F7:
                    # take a screenshot!
                    data = gfx.read_pixels(*NATIVE_RESOLUTION)
                    surface = pygame.image.fromstring(data, NATIVE_RESOLUTION, 'RGB', True)
                    pygame.image.save(surface, 'screenshot.png')
                    # todo F8 to take video
                    #   image data is stored in memory, when press F8 save as image sequence, then make gif
                    #   http://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python

    def construct_queue(self):
        self.draw_queue = []  # reset the queue
        for entity in self.entities:
            rend = entity.get_components(RenderComponent)
            for r in rend: r.queue_draw()
        self.sort_queue()

        if log.basic.isEnabledFor(logging.DEBUG):
            for job in self.draw_queue:
                log.basic.debug(job.desc)

    def quit(self):
        log.basic.info("quitting")
        # self.unload_resources()  # this is breaking at the mo
        pygame.quit()
        if self.console_thread:
            self.console_thread.join()  # closes thread (I think) (hmm, usually doesn't work)
        log.basic.info("finished")
        exit(0)  # or quit()?
