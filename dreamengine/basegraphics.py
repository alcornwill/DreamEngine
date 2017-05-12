# this is an abstraction between the engine and OpenGL
# should make it easier to move to C++
# todo it would be good if this was just base.py, then abstract numpy and pygame as well

# it needs to be more customizable, safer, and maybe contain more state
# (do I really need to use __enter__ and __exit__ so much? was only doing it because it was fun)

from pprint import pprint
import os
import pickle
import logging
import pyassimp as ass
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from ctypes import c_void_p
import numpy as np
import pygame

from dreamengine.constants import *
from dreamengine.utils import *
from dreamengine.mathutils import *
import dreamengine.log as log

null = c_void_p(0)

OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
OpenGL.FULL_LOGGING = False

DEFAULT_FORMAT = GL_RGB
DEFAULT_FORMAT_ALPHA = GL_RGBA
DEFAULT_INTERNAL_FORMAT = GL_RGB8
DEFAULT_INTERNAL_FORMAT_ALPHA = GL_RGBA8

def get_format(alpha):
    return DEFAULT_FORMAT_ALPHA if alpha else DEFAULT_FORMAT

def get_internal_format(alpha):
    return DEFAULT_INTERNAL_FORMAT_ALPHA if alpha else DEFAULT_INTERNAL_FORMAT

def init():
    # global state
    log.basic.debug("opengl version: {}.{}".format(glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))
    log.basic.debug("shader language version: {}".format(glGetString(GL_SHADING_LANGUAGE_VERSION)))

    glShadeModel(GL_SMOOTH)  # still not sure how to make flat shaded (haven't actually tried doing from blender)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)

    # glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])  # doesn't exist in current context?
    # glEnable(GL_MULTISAMPLE)  # don't need to do this on the window buffer right?

    glEnable(GL_CULL_FACE)
    glFrontFace(GL_CCW)
    glCullFace(GL_BACK)

    glEnable(GL_LINE_STIPPLE)
    glLineStipple(1, 0x0F0F)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(2)

    # (could be done in pass, but when would you not want this enabled?)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glDepthFunc(GL_LEQUAL)
    glDepthRange(0, 1)

def update_default_pass():
    glDisable(GL_BLEND)

    # doesn't seem to do anything (I'm probably using it wrong)
    # glEnable(GL_MULTISAMPLE) if USE_MULTISAMPLE else glDisable(GL_MULTISAMPLE)
    # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

# hmm for single line opengl calls like this, maybe better to just expose API
def enable_blending():
    # will probably want more control over blending equation
    glEnable(GL_BLEND)
    glBlendEquation(GL_FUNC_ADD)  # always this? confused

def set_additive_blending():
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

def set_overwrite_blending():
    glBlendFunc(GL_SRC_ALPHA, GL_ZERO)

def update_additive_pass():
    # glEnablei(GL_BLEND, 0) # i is the framebuffer?
    enable_blending()
    set_additive_blending()

def update_back_face_pass():
    glCullFace(GL_FRONT)

def enable_front():
    glCullFace(GL_BACK)

def enable_back():
    glCullFace(GL_FRONT)

def depth_test(value):
    func = glEnable if value else glDisable
    func(GL_DEPTH_TEST)

def delete_buffer(buf):
    glDeleteBuffers(1, [buf])

def delete_vertex_array(vao):
    glDeleteVertexArrays(1, [vao])

def delete_texture(texture):
    glDeleteTextures(1, [texture])

def delete_framebuffer(framebuffer):
    glDeleteFramebuffers(1, [framebuffer])

def delete_renderbuffer(renderbuffer):
    glDeleteRenderbuffers(1, [renderbuffer])

def delete_program(program):
    glDeleteProgram(program)

def draw_instanced_base_vertex(style, length, offset, instance_count, base_vertex):
    glDrawElementsInstancedBaseVertex(style,
                                      length,
                                      GL_UNSIGNED_SHORT,
                                      c_void_p(offset),
                                      instance_count,
                                      base_vertex)

def draw_mesh_base_vertex(style, length, offset, base_vertex):
    glDrawElementsBaseVertex(style,
                             length,
                             GL_UNSIGNED_SHORT,
                             c_void_p(offset),
                             base_vertex)

def draw_instanced(style, length, instance_count):
    glDrawElementsInstanced(style,
                            length,
                            GL_UNSIGNED_SHORT,
                            null,
                            instance_count)

def draw_mesh(style, length):
    glDrawElements(style,
                   length,
                   GL_UNSIGNED_SHORT,
                   null)

def get_program_attributes(program):
    # get attributes
    attributes = {}
    count = glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES)
    # there are a few other cool things you can do with glGetProgramiv()
    # https://www.khronos.org/registry/OpenGL-Refpages/es2.0/xhtml/glGetProgramiv.xml
    bufSize = glGetProgramiv(program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH)
    for i in range(count):
        length = GLsizei()
        size = GLint()
        type_ = GLenum()
        name = (GLchar * bufSize)()
        glGetActiveAttrib(program, i, bufSize, length, size, type_, name)
        # even though we asked for attribute at location i, it's complete LIES
        index = glGetAttribLocation(program, name.value)
        attributes[name.value] = index
        # types: GL_FLOAT GL_FLOAT_VEC2 GL_FLOAT_VEC3 GL_FLOAT_VEC4 GL_FLOAT_MAT2 GL_FLOAT_MAT3 GL_FLOAT_MAT4
        # there are more, it's not documented well
        log.basic.debug('found attribute "{}"'.format(name.value))
    return attributes

class UniformInfo:
    def __init__(self, index, type_, t_pos=-1):
        self.index = index
        self.type = type_
        self.t_pos = t_pos

def get_program_uniforms(program):
    # get uniforms
    uniforms = {}
    t_pos_count = 0
    count = glGetProgramiv(program, GL_ACTIVE_UNIFORMS)
    for i in range(count):
        # lol, they made this API pythonic, but not the attributes one, and didn't even document it
        ret = glGetActiveUniform(program, i)
        name = ret[0]
        size = ret[1]  # is this just the 'length' for arrays?
        type_ = ret[2]
        index = glGetUniformLocation(program, name)
        if type_ == GL_SAMPLER_2D or type_ == GL_SAMPLER_2D_ARRAY:
            unif = UniformInfo(index, type_, t_pos_count)
            t_pos_count += 1  # not sure this is reliable way to get tex_index?
        else:
            unif = UniformInfo(index, type_)
        uniforms[name] = unif
        # gl_type = get_gl_type(type_)
        # from the type we should be able to get the stride
        log.basic.debug('found uniform "{}" index {} size {}'.format(name, index, size))
    return uniforms

def get_program_uniform_blocks(program):
    uniform_blocks = {}
    num_blocks = glGetProgramiv(program, GL_ACTIVE_UNIFORM_BLOCKS)
    for b in range(num_blocks):
        name_len = GLint()
        glGetActiveUniformBlockiv(program, b, GL_UNIFORM_BLOCK_NAME_LENGTH, name_len)
        name_len = name_len.value
        block_name = glGetActiveUniformBlockName(program, b, name_len)
        decoded_name = np.array(block_name).tobytes()[:-1]
        idx = glGetUniformBlockIndex(program, block_name)

        # num_uniforms = glGetActiveUniformBlockiv(program, idx, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS)
        # log.basic.debug(num_uniforms)
        # # it doesn't like this
        # # indices, hmm = glGetActiveUniformBlockiv(program, idx, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES)
        # # log.basic.debug(indices)
        # size = glGetActiveUniformBlockiv(program, idx, GL_UNIFORM_BLOCK_DATA_SIZE)
        # log.basic.debug(size)

        uniform_blocks[decoded_name] = idx
        log.basic.debug('found uniform block "{}" index {}'.format(decoded_name, idx))
    return uniform_blocks

def bind_program_uniform_block(program, index, block):
    glUniformBlockBinding(program, index, block)

def bind_program(program):
    glUseProgram(program)

def bind_vao(vao):
    glBindVertexArray(vao)

def load_texture(path, hasAlpha=False):
    surface = pygame.image.load(path)
    if hasAlpha:
        data = pygame.image.tostring(surface, 'RGBA', True)
    else:
        data = pygame.image.tostring(surface, 'RGB', True)
    width, height = surface.get_rect().size
    width = width
    height = height
    del surface
    return width, height, data

def compile_program(vs_source, fs_source):
    return compileProgram(
        compileShader(vs_source, GL_VERTEX_SHADER),
        compileShader(fs_source, GL_FRAGMENT_SHADER)
    )

def create_uniform_buffer(index, size):
    buf = glGenBuffers(1)
    glBindBufferBase(GL_UNIFORM_BUFFER, index, buf)
    glBufferData(GL_UNIFORM_BUFFER, size, null, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_UNIFORM_BUFFER, 0)
    return buf

def create_texture(width, height, alpha, data):
    # not using
    internalFormat = get_internal_format(alpha)
    format = get_format(alpha)
    index = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, index)
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    return index

def create_render_texture(width, height, alpha):
    internalFormat = get_internal_format(alpha)
    format = get_format(alpha)
    index = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, index)
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, null)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    # glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, index)
    # glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, SAMPLES, internalFormat, width, height, GL_TRUE)
    # glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    # glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, index, 0)
    # glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, index, 0)
    draw_buffers = [GL_COLOR_ATTACHMENT0]
    glDrawBuffers(1, draw_buffers)

    glBindTexture(GL_TEXTURE_2D, 0)
    # glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
    return index

SAMPLER_FILTER_MAP = {
    NEAREST: GL_NEAREST,
    LINEAR: GL_LINEAR
}

SAMPLER_WRAP_MAP = {
    REPEAT: GL_REPEAT,
    MIRROR: GL_MIRRORED_REPEAT,
    CLAMP_EDGE: GL_CLAMP_TO_EDGE,
    CLAMP_BORDER: GL_CLAMP_TO_BORDER
}

def create_sampler(filter, wraps, wrapt):
    # convert to gl types
    filter = SAMPLER_FILTER_MAP[filter]
    wraps = SAMPLER_WRAP_MAP[wraps]
    wrapt = SAMPLER_WRAP_MAP[wrapt]

    sampler = glGenSamplers(1)
    # are there actually any other options?
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, filter)
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, filter)
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, wraps)
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, wrapt)
    return sampler

def create_color_renderbuffer(width, height):
    cbuf = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, cbuf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height)
    # glRenderbufferStorageMultisample(GL_RENDERBUFFER, SAMPLES, GL_RGBA, width, height)
    return cbuf

def create_depth_renderbuffer(width, height):
    dbuf = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, dbuf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
    # glRenderbufferStorageMultisample(GL_RENDERBUFFER, SAMPLES, GL_DEPTH_COMPONENT, width, height)
    return dbuf

def create_framebuffer(cbuf, dbuf):
    # probably should be CreateFramebuffer, but can't be botehred
    fbuf = glGenFramebuffers(1)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbuf)
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, cbuf)
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, dbuf)
    # glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
    return fbuf

def read_pixels(width, height):
    data = glReadPixels(0, 0, width, height, DEFAULT_FORMAT, GL_UNSIGNED_BYTE)
    return np.array(data).tostring()

def clear():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

def bind_framebuffer(fbuf):
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbuf)

def clear_framebuffer(width, height, clear, clear_color):
    glViewport(0, 0, width, height)
    if clear:
        glClearColor(*clear_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

def blit_framebuffer(source, w1, h1, dest, w2, h2):
    glBindFramebuffer(GL_READ_FRAMEBUFFER, source)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dest)
    glViewport(0, 0, w2, h2)

    # copy to window buffer
    # todo variable aspect ratio (but fixed height)
    #   would have to make all framebuffers again I think
    glBlitFramebuffer(0, 0, w1, h1,
                      0, 0, w2, h2,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST)
    # glInvalidateFramebuffer() after flip()?

def get_basic_uniform_type(enum):
    # this will always be a type switch because can't derive an enum
    if enum == GL_FLOAT_VEC3:
        return Vec3Uniform
    elif enum == GL_FLOAT_VEC4:
        return Vec4Uniform
    elif enum == GL_FLOAT:
        return FloatUniform
    elif enum == GL_INT:
        return IntUniform
    elif enum == GL_FLOAT_MAT4:
        return Mat4Uniform
    elif enum == GL_FLOAT_MAT3:
        return Mat3Uniform

# hmm i don't think Uniform should be a class, just functions with maybe tiny bit of state
# hmm
#   for name, init, update in UNIFORM_MAP:
#       cls = type(name, (Uniform), {})
#       cls.__init__ = types.MethodType(init, cls)
#       cls.update = types.MethodType(update, cls)
# use metaprogramming?
# there are 33 glUniformXxx variants... maybe don't need all of them
# then derivations won't need to use opengl and can go in rendering.py
class Uniform:
    # hmm, should this have methods for bind_texture, set_uniform, ... ?
    def __init__(self, uniform_index):
        self.uniform_index = uniform_index

    def update(self):
        pass

class Texture2DUniform(Uniform):
    def __init__(self, uniform_index, t_pos, tex, sampler):
        super().__init__(uniform_index)
        self.tex_unit = t_pos
        from dreamengine.rendering import R
        tex = R.get_resource(R.texturedata, tex)
        sampler = R.get_resource(R.samplers, sampler)
        self.texture_id = tex  # how tf does that work
        self.sampler = sampler

    def update(self):
        glUniform1i(self.uniform_index, self.tex_unit)
        glActiveTexture(GL_TEXTURE0 + self.tex_unit)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBindSampler(self.tex_unit, self.sampler)
        # need to unbind things as well?

class Texture2DArrayUniform(Uniform):
    def __init__(self, uniform_index, t_pos, t_array_id, sampler):
        super().__init__(uniform_index)
        self.tex_unit = t_pos
        self.t_array_id = t_array_id
        self.sampler = sampler

    def update(self):
        glUniform1i(self.uniform_index, self.tex_unit)
        glActiveTexture(GL_TEXTURE0 + self.tex_unit)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.t_array_id)
        glBindSampler(self.tex_unit, self.sampler)

class CubemapUniform(Uniform):
    def __init__(self, uniform_index, t_pos, cubemap):
        super().__init__(uniform_index)
        from dreamengine.rendering import R
        self.tex_unit = 0  # todo
        cubemap = R.get_resource(R.cubemaps, cubemap)
        self.cubemap_id = cubemap.id

    def update(self):
        glUniform1i(self.uniform_index, self.tex_unit)
        glActiveTexture(GL_TEXTURE0 + self.tex_unit)
        # we should have TexStateChangeJob so can do this minimum number of times
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemap_id)
        # todo uhh why isn't there a sampler? do cubemaps not have them?

class Vec4Uniform(Uniform):
    def __init__(self, uniform_index, vec):
        super().__init__(uniform_index)
        self.vec = vec

    def update(self):
        try:
            glUniform4f(self.uniform_index, *self.vec)
        except TypeError:
            print(self.vec)

class Vec3Uniform(Uniform):
    def __init__(self, uniform_index, vec):
        super().__init__(uniform_index)
        self.vec = vec

    def update(self):
        glUniform3f(self.uniform_index, *self.vec)

class FloatUniform(Uniform):
    def __init__(self, uniform_index, val):
        super().__init__(uniform_index)
        self.val = val

    def update(self):
        glUniform1f(self.uniform_index, self.val)

class IntUniform(Uniform):
    def __init__(self, uniform_index, val):
        super().__init__(uniform_index)
        self.val = val

    def update(self):
        glUniform1i(self.uniform_index, self.val)

class Mat4Uniform(Uniform):
    def __init__(self, uniform_index, mat=None):
        super().__init__(uniform_index)
        self.mat = mat
        self.count = 1

    def update(self):
        glUniformMatrix4fv(self.uniform_index, self.count, GL_TRUE, self.mat)

class Mat3Uniform(Uniform):
    def __init__(self, uniform_index, mat=None):
        super().__init__(uniform_index)
        self.mat = mat
        self.count = 1

    def update(self):
        glUniformMatrix3fv(self.uniform_index, self.count, GL_TRUE, self.mat)

class UniformBlockUpdater:
    def __init__(self, buffer, index):
        self.enum = GL_UNIFORM_BUFFER
        self.buf = buffer
        self.index = index
        self.offset = 0

    def __enter__(self):
        glBindBufferBase(self.enum, self.index, self.buf)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        glBindBuffer(self.enum, 0)

    def subdata(self, stride, data):
        if not isinstance(data, np.ndarray):  # this creates a lot of numpy arrays...
            data = np.array(data, 'f')
        glBufferSubData(self.enum, self.offset, stride, data)
        self.offset += stride
        # if I was really clever I would implement std140 properly
        # so you give it the type and it works out the stride, offset,
        # keeps track of alignment

class CreateBuffer:
    def __init__(self, enum, size):
        self.enum = enum
        self.option = GL_STATIC_DRAW
        self.index = None
        self.size = size
        self.offset = 0

    def __enter__(self):
        self.index = glGenBuffers(1)
        glBindBuffer(self.enum, self.index)
        # this isn't really 'updating' is it. it's initializing
        glBufferData(self.enum, self.size, null, self.option)
        return self

    def __exit__(self, something, something_else, another_thing):
        glBindBuffer(self.enum, 0)

    def subdata(self, size, data):
        glBufferSubData(self.enum, self.offset, size, data)
        self.offset += size

class CreateVbo(CreateBuffer):
    def __init__(self, size):
        super().__init__(GL_ARRAY_BUFFER, size)

class CreateIbo(CreateBuffer):
    def __init__(self, size):
        super().__init__(GL_ELEMENT_ARRAY_BUFFER, size)

class CreateVao:
    def __init__(self):
        self.index = None

    def __enter__(self):
        self.index = glGenVertexArrays(1)
        glBindVertexArray(self.index)
        return self

    def __exit__(self, something, something_else, another_thing):
        glBindVertexArray(0)

    def bind_vbo(self, vbo):
        glBindBuffer(GL_ARRAY_BUFFER, vbo)

    def vtx_attr(self, index, stride, offset):
        # TODO!! enum parameter so don't have to be GL_FLOAT...
        #   but then you have to specify that all the way back in AttributeData or something?
        glEnableVertexAttribArray(index)
        glVertexAttribPointer(index, stride, GL_FLOAT, False, 0, c_void_p(offset))

    def bind_ibo(self, ibo):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)

class CreateTextureArray:
    def __init__(self, width, height, count, alpha):
        self.index = None
        self.width = width
        self.height = height
        self.count = count
        self.format = GL_RGBA if alpha else GL_RGB
        self.internalFormat = GL_RGBA8 if alpha else GL_RGB8
        self.mipmaps = MIPMAPS if is_power2(width) and is_power2(height) else 0

    def __enter__(self):
        self.index = glGenTextures(1)
        # glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.index)
        # glTexStorage3D(GL_TEXTURE_2D_ARRAY, MIPMAPS, self.format, self.width, self.height, self.count)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, self.mipmaps, self.internalFormat, self.width, self.height, self.count,
                     0, self.format, GL_UNSIGNED_BYTE, null)
        # glGenerateMipmap(GL_TEXTURE_2D_ARRAY)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, self.mipmaps)
        # glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_GENERATE_MIPMAP, GL_TRUE)
        return self

    def __exit__(self, something, something_else, another_thing):
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def subdata(self, t_index, data):
        # wtf, if you increase mipmaps this fucks up 'invalid value'
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                        0,
                        0, 0, t_index,
                        self.width, self.height, 1,
                        self.format, GL_UNSIGNED_BYTE,
                        data)

def create_cubemap(faces):
    # parameters should probably be 'width, height, data[]'?
    format = get_format(False)
    internalFormat = get_internal_format(False)
    cubemap_id = glGenBuffers(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_id)
    first = faces[0]
    # glTexStorage2D(GL_TEXTURE_CUBE_MAP,
    #                MIPMAPS,
    #                GL_RGB,
    #                first.width, first.height)
    for i, face in enumerate(faces):
        target = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i
        # glTexSubImage2D(target,
        #                 0,
        #                 0, 0,
        #                 first.width, first.height,
        #                 GL_RGB,
        #                 GL_UNSIGNED_BYTE,
        #                 face.data)
        glTexImage2D(target,
                     MIPMAPS,
                     internalFormat,
                     first.width, first.height,
                     0, format,
                     GL_UNSIGNED_BYTE,
                     face.data)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, MIPMAPS)
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
    return cubemap_id

class BoneWeight:
    def __init__(self, index=0, weight=0):
        self.index = index
        self.weight = weight

class BoneData:
    # for pickling
    def __init__(self, indices, weights):
        self.indices = indices
        self.weights = weights

def cached_process(func, origin_path, cache_path, invalidate):
    # (this could probably be cleverer. a class or decorator?)
    # save the result of a process to file and in load subsequent calls
    if os.path.isfile(cache_path):
        origin_edit_time = os.path.getmtime(origin_path)
        cache_edit_time = os.path.getmtime(cache_path)
        if not invalidate and \
                not origin_edit_time > cache_edit_time and \
                os.path.isfile(cache_path):
            log.basic.debug('cached data loaded from "{}"'.format(cache_path))
            return pickle.load(open(cache_path, 'rb'))
    # else do the function and save the pickle
    result = func()
    pickle.dump(result, open(cache_path, 'wb'))
    log.basic.debug('cached data saved to "{}"'.format(cache_path))
    return result

class MeshLoader:
    # load model with pyassimp and convert into data we can send to GPU
    # (wanted to abstract from pyassimp a bit)
    # could potentially load a lot more than just the mesh...

    class AttributeData:
        def __init__(self, stride, data):
            self.stride = stride  # size in bytes of one attribute element
            self.data = data
            self.data_len = len(data)
            # dtype should probably be a parameter
            self.size = len(data) * data.dtype.itemsize  # the size of the whole attribute

    def __init__(self, model_path):
        self.model_path = model_path
        self.scene = None

    def __enter__(self):
        self.scene = ass.load(self.model_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # todo disabling this for now!
        # ass.release(self.scene)
        pass

    def get_bones(self, mesh, num_verts):
        if not any(mesh.bones):
            return None, None

        # check if already cached bone data
        #   ok I implemented this because I thoughT it was the bottleneck, but after profiling I think it's just pyassimp
        #   so it might be worth getting rid of caching all together, but it probably is faster
        bone_data_path = self.model_path + '.' + mesh.name + ".p"
        bone_data = cached_process(lambda: MeshLoader.construct_bone_data(mesh.bones, num_verts),
                                   self.model_path, bone_data_path, INVALIDATE_BONE_DATA)

        bone_index_data = MeshLoader.AttributeData(BONE_INDEX_STRIDE, bone_data.indices)
        bone_weight_data = MeshLoader.AttributeData(BONE_WEIGHT_STRIDE, bone_data.weights)

        return bone_index_data, bone_weight_data

    @staticmethod
    def construct_bone_data(bones, num_verts):
        log.basic.debug('reconstructing bone data...')
        if len(bones) > MAX_BONES:
            log.basic.warning("more than {} bones detected: {}".format(MAX_BONES, len(bones)))
        vertex_weights = [[] for x in range(num_verts)]
        for i, bone in enumerate(bones):
            for weight in bone.weights:
                # note: bones indexed by their order in list (maybe associate with name somehow)
                vertex_weights[weight.vertexid].append(BoneWeight(i, weight.weight))
        for i in range(num_verts):
            vw = vertex_weights[i]
            l = len(vw)
            if l > MAX_BONE_WEIGHTS_PER_VERTEX:
                log.basic.warning(
                    "more than {} bone weights detected: {} \r\n limit the bone weights per vertex in your 3D modelling application".format(
                        MAX_BONE_WEIGHTS_PER_VERTEX, l))
                # wonder if assimp can do that in post processing?
            elif l < MAX_BONE_WEIGHTS_PER_VERTEX:
                # hmm, guess I need to fill with zeros?
                d = MAX_BONE_WEIGHTS_PER_VERTEX - l
                vw.extend([BoneWeight() for i in range(d)])
                # assume sum of weights is less than 1
        indices = []
        weights = []
        for vw in vertex_weights:
            index = (vw[0].index, vw[1].index, vw[2].index, vw[3].index)
            indices.append(np.array(index, np.int))  # TODO!! this should be np.uint8
            weight = (vw[0].weight, vw[1].weight, vw[2].weight, vw[3].weight)
            weights.append(np.array(weight, 'f'))  # doesn't really need to be float because only 0.0 - 1.0?

        return BoneData(np.concatenate(indices), np.concatenate(weights))

    def get_data(self, mesh):
        position_data = MeshLoader.AttributeData(POSITION_STRIDE, mesh.vertices.flatten())
        face_data = MeshLoader.AttributeData(1, mesh.faces.flatten().astype(np.uint16))
        normal_data = None
        color_data = None
        uv_data = None
        if mesh.normals.any():
            normal_data = MeshLoader.AttributeData(NORMAL_STRIDE, mesh.normals.flatten())
        if mesh.colors.any():
            color_data = MeshLoader.AttributeData(COLOR_STRIDE, mesh.colors.flatten())
        if mesh.texturecoords.any():
            data = mesh.texturecoords[0]  # only supports 1 uv component!
            uv_data = MeshLoader.AttributeData(UV_STRIDE, np.delete(data, 2, 1).flatten())  # drop third dimension!

        bone_index_data, bone_weight_data = self.get_bones(mesh, face_data.data_len)

        return position_data, face_data, normal_data, color_data, uv_data, bone_index_data, bone_weight_data

    @staticmethod
    def print_node_r(node, space):
        log.basic.debug(space + node.name)
        for child in node.children:
            MeshLoader.print_node_r(child, space + "  ")

        # switch (type) {
        #     CASE(GL_FLOAT, 1, GLfloat);
        #     CASE(GL_FLOAT_VEC2, 2, GLfloat);
        #     CASE(GL_FLOAT_VEC3, 3, GLfloat);
        #     CASE(GL_FLOAT_VEC4, 4, GLfloat);
        #     CASE(GL_INT, 1, GLint);
        #     CASE(GL_INT_VEC2, 2, GLint);
        #     CASE(GL_INT_VEC3, 3, GLint);
        #     CASE(GL_INT_VEC4, 4, GLint);
        #     CASE(GL_UNSIGNED_INT, 1, GLuint);
        #     CASE(GL_UNSIGNED_INT_VEC2, 2, GLuint);
        #     CASE(GL_UNSIGNED_INT_VEC3, 3, GLuint);
        #     CASE(GL_UNSIGNED_INT_VEC4, 4, GLuint);
        #     CASE(GL_BOOL, 1, GLboolean);
        #     CASE(GL_BOOL_VEC2, 2, GLboolean);
        #     CASE(GL_BOOL_VEC3, 3, GLboolean);
        #     CASE(GL_BOOL_VEC4, 4, GLboolean);
        #     CASE(GL_FLOAT_MAT2, 4, GLfloat);
        #     CASE(GL_FLOAT_MAT2x3, 6, GLfloat);
        #     CASE(GL_FLOAT_MAT2x4, 8, GLfloat);
        #     CASE(GL_FLOAT_MAT3, 9, GLfloat);
        #     CASE(GL_FLOAT_MAT3x2, 6, GLfloat);
        #     CASE(GL_FLOAT_MAT3x4, 12, GLfloat);
        #     CASE(GL_FLOAT_MAT4, 16, GLfloat);
        #     CASE(GL_FLOAT_MAT4x2, 8, GLfloat);
        #     CASE(GL_FLOAT_MAT4x3, 12, GLfloat);
